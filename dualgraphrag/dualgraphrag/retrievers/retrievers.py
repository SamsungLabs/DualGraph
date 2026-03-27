# mypy: disable-error-code="method-assign"
import asyncio
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from itertools import chain
from typing import Any, List

import pandas as pd
from pydantic_ai import (
    Agent,
    AgentRunResult,
    ModelRetry,
    RunContext,
    capture_run_messages,
)
from pytablewriter import MarkdownTableWriter
from rdflib import URIRef, Variable

from ..base import (
    LLM,
    LOGGER,
    BaseKVStorage,
    RetrievalResult,
    Retriever,
    Tokenizer,
    VectorStorage,
)
from ..configurations import NaiveQueryConfig, SparqlQueryConfig, TKGQueryConfig
from ..storage import RDFoxCSHandle, collapse_prefix
from ..utils import (
    build_conversation,
    stringify_table_list,
    truncate_table_list_by_token_size,
)

L = logging.getLogger(LOGGER)


class BuildContextFromChunks:

    def __init__(self, *, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    async def __call__(
        self,
        chunks: list[dict],
        params,
    ) -> tuple[str, list[dict]]:
        context_parts: list[list[str]] = [
            ["id", "content"],
        ]
        for i, chunk in enumerate(chunks):
            context_parts.append(
                [
                    str(i),
                    chunk["content"],
                ]
            )
        context_parts = truncate_table_list_by_token_size(
            data=context_parts,
            tokenizer=self.tokenizer,
            collate_func=", ".join,
            max_token_size=params.chunks_max_token_size,
        )
        context = stringify_table_list(
            data=context_parts,
            table_format=params.chunks_table_format,
            mark_format=True,
        )
        return context, chunks[: len(context_parts) - 1]


class RetrieveRelevantChunks:

    def __init__(self, *, chunks_vdb: VectorStorage, chunks_db: BaseKVStorage):
        self.chunks_vdb = chunks_vdb
        self.chunks_db = chunks_db

    async def __call__(self, query: list[str], params) -> list[dict]:
        """

        Args:
            query: In enriched format, so a list with query and optionally
                   relevant entities.

        """
        chunk_ids_dists_ = await asyncio.gather(
            *[
                self.chunks_vdb.query(
                    query=q,
                    top_k=params.retrieve_top_k,
                    meta_filter=None,
                )
                for q in query
            ]
        )
        L.debug("Retrieved %d chunks (before deduplication)", len(chunk_ids_dists_))
        chunk_ids_dists = self._deduplicate_retrieved_chunks(
            chunks=list(chain(*chunk_ids_dists_)),
            top_k=params.retrieve_top_k,
        )
        L.debug("Retrieved %d chunks (after deduplication)", len(chunk_ids_dists))

        # Retrieve chunks from key-value storage
        chunks = await asyncio.gather(
            *[self.chunks_db.get_by_id(c["id"]) for c in chunk_ids_dists]
        )
        for chunk, chunk_id_dist in zip(chunks, chunk_ids_dists, strict=True):
            assert chunk is not None
            if "id" in chunk:
                assert chunk["id"] == chunk_id_dist["id"]
            else:
                chunk["id"] = chunk_id_dist["id"]
            chunk["distance"] = chunk_id_dist["distance"]

        return chunks  # type: ignore[return-value]

    def _deduplicate_retrieved_chunks(
        self, chunks: list[dict], top_k: int
    ) -> list[dict]:
        # Handle deduplication after using multiple alternative queries
        df = pd.DataFrame.from_records(chunks)
        df = df.groupby(["id"], as_index=False).min()
        df = df.sort_values(by="distance", ascending=True)  # type: ignore
        return df.to_dict(orient="records")[:top_k]


class ClosedBookRetriever(Retriever):
    async def __call__(self, query: str, params: Any, **kwargs) -> RetrievalResult:
        # pylint: disable=unused-argument
        return RetrievalResult(context=["EMPTY"])


@dataclass
class NaiveRetriever(Retriever):

    tokenizer: Tokenizer
    chunks_vdb: VectorStorage
    text_chunks: BaseKVStorage

    async def __call__(
        self, query: str, params: NaiveQueryConfig, **kwargs
    ) -> RetrievalResult:
        context = {}

        chunks = await RetrieveRelevantChunks(
            chunks_vdb=self.chunks_vdb, chunks_db=self.text_chunks
        )([query], params)

        context["Sources"], used_chunks = await BuildContextFromChunks(
            tokenizer=self.tokenizer
        )(chunks, params)

        return RetrievalResult(
            context=[
                "\n".join(f"-----{k}-----\n{v}\n" for k, v in context.items() if v)
            ],
            metadata={
                "anchor_chunks": [c["id"] for c in used_chunks],
            },
        )


@dataclass
class TKGRetriever(Retriever):
    tokenizer: Tokenizer
    entities_vdb: VectorStorage
    chunks_vdb: VectorStorage
    text_chunks: BaseKVStorage
    entity_base: pd.DataFrame

    async def __call__(
        self, query: str, params: TKGQueryConfig, **kwargs
    ) -> RetrievalResult:
        entity_ids = [
            e["entity_id"]
            for e in await self.entities_vdb.query(
                query=query,
                top_k=params.retrieve_top_k_ents,
            )
        ]
        entities = self.entity_base.loc[entity_ids].to_dict(orient="records")

        chunk_ids: list[str] = self._select_chunks(
            entities,
            chunks_top_k=params.retrieve_top_k_chunks,
            entities_top_k=params.retrieve_top_k_ents,
        )
        chunks: list[dict] = []
        for pos, chunk in enumerate(await self.text_chunks.get_by_ids(chunk_ids)):
            assert isinstance(chunk, dict)
            chunk["normalized_order"] = pos / len(chunk_ids)
            chunks.append(chunk)

        context, used_chunks = await BuildContextFromChunks(tokenizer=self.tokenizer)(
            chunks,
            params,
        )

        anchor_chunks = [c["id"] for c in used_chunks]
        anchor_nodes = [
            e["name"]
            for e in entities
            if any(x in anchor_chunks for x in e["chunk_id"])
        ]

        return RetrievalResult(
            context=[context],
            metadata={
                "anchor_chunks": anchor_chunks,
                "anchor_nodes": anchor_nodes,
            },
        )

    def _select_chunks(
        self, entities: list[dict], chunks_top_k: int, entities_top_k: int
    ) -> list[str]:
        chunk_counter = Counter(
            [chunk_id for e in entities for chunk_id in e["chunk_id"]]
        )
        chunk_scores = {}
        for entity_pos, entity in enumerate(entities):
            for chunk_id in entity["chunk_id"]:
                if chunk_id in chunk_scores:
                    continue
                chunk_scores[chunk_id] = (
                    chunk_counter[chunk_id]
                    / sum(chunk_counter.values())
                    * (1 - (entity_pos / entities_top_k))
                )
        return [
            c for c, _ in sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        ][:chunks_top_k]


@dataclass
class SparqlRetriever(Retriever):
    tokenizer: Tokenizer
    entities_vdb: VectorStorage
    chunks_vdb: VectorStorage
    text_chunks: BaseKVStorage
    entity_base: pd.DataFrame
    specs_vdb: VectorStorage
    features_vdb: VectorStorage
    categories_vdb: VectorStorage
    rdfoxhandle: RDFoxCSHandle | None
    prompts: dict[str, str]
    llm: LLM

    async def __call__(
        self, query: str, params: SparqlQueryConfig, **kwargs
    ) -> RetrievalResult:
        assert self.rdfoxhandle is not None
        sparql_generation_context: str = kwargs.get("sparql_generation_context", "")
        metadata: dict[str, Any] = kwargs.get("metadata", {})

        prompt = self.prompts["sparql_kg_query_generation_prompt_template"].format(
            query=query,
            additional_schema_information=sparql_generation_context,
        )
        metadata["sparql_generation_prompt"] = prompt

        responses, _ = await self.llm.get_n_completions(
            build_conversation(prompt=prompt),
            n=params.n_generated_queries,
            extra_body={"use_beam_search": params.use_beam_search},
            log_data={"phase": "sparql query generation"},
        )
        metadata["sparql_llm_responses"] = responses

        def _extract_sparql_query(output: str) -> str | None:
            if match := re.search(
                r"<SPARQL>((?:(?!</SPARQL>).)*)</SPARQL>", output, flags=re.DOTALL
            ):
                return match.group(1).strip()
            return None

        sparql_queries: list[str | None] = [_extract_sparql_query(r) for r in responses]

        outputs: list[str | None] = await self._run_sparql_queries(
            queries=sparql_queries, result_limit=params.query_result_limit
        )
        metadata["sparql_rdfox_responses"] = outputs
        outputs = await self._format_sparql_query_responses(
            responses=outputs, result_limit=params.query_result_limit
        )

        context_: list[str] = [
            "To gather information for answering the question relevant "
            "knowledge graph was queried. Below are the queries and "
            f"received answers (limited to {params.query_result_limit} results):"
        ]
        for sparql_query, outputstr in zip(sparql_queries, outputs, strict=True):
            if outputstr is None:
                continue
            context_.append(f"Query:\n{sparql_query}\n\nAnswer:\n{outputstr}\n###")

        if len(context_) == 1:
            context = "No information could be retrieved from knowledge graph"
            should_fallback = True
        else:
            context = "\n\n".join(context_)
            should_fallback = False

        metadata["should_fallback_to_tkg"] = should_fallback

        return RetrievalResult(
            context=[context],
            metadata=metadata,
        )

    async def retrieve_sparql_generation_context(
        self,
        query: str,
        params: SparqlQueryConfig,
        metadata: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        specs: list[dict[str, str]] = [
            json.loads(s["nodes"])
            for s in await self.specs_vdb.query(
                query=query,
                top_k=params.retrieve_top_k_specs,
            )
        ]
        metadata["specs_from_vdb"] = specs

        feats: list[dict[str, str]] = [
            json.loads(f["nodes"])
            for f in await self.features_vdb.query(
                query=query,
                top_k=params.retrieve_top_k_features,
            )
        ]
        metadata["features_from_vdb"] = feats

        categories: list[dict[str, str]] = [
            json.loads(c["nodes"])
            for c in await self.categories_vdb.query(
                query=query,
                top_k=params.retrieve_top_k_categories,
            )
        ]
        metadata["categories_from_vdb"] = categories

        async def _retrieve_node_types(node: str) -> list[str]:
            assert self.rdfoxhandle is not None
            query = f"SELECT DISTINCT ?type\nWHERE {{\n    {node} a ?type .\n}}"

            result = self.rdfoxhandle.parse_result(await self.rdfoxhandle.query(query))

            types = []
            for binding in result.bindings:
                ref = binding[Variable("type")]
                assert isinstance(ref, URIRef)
                types.append(ref.fragment)
            return types

        entities: list[dict[str, Any]] = []
        for e in await self.entities_vdb.query(
            query=query,
            top_k=params.retrieve_top_k_entities,
            meta_filter={"in_triple": 1},
        ):
            node = f"skg:{e['rdfox_entity_id']}"
            types = await _retrieve_node_types(node)
            types = [
                f"skgt:{t}" for t in types if t not in {"UTKG_Entity", "SKG_Entity"}
            ]
            assert types
            entities.append({"node": node, "types": types})
        metadata["entities_from_vdb"] = entities

        lines = []
        lines.append("Distinct specification patterns:")
        lines.extend(
            [
                "- ?product skg:hasSpec ?spec .\n"
                f"  ?spec skg:inSection {n['section']} ;\n"
                f"        skg:inEntry {n['entry']} ;\n"
                f"        skg:hasValue {n['value']} ."
                for n in specs
            ]
        )
        lines.append("Additional shortcut patterns for popular features:")
        lines.extend([f"- ?product skg:hasFeature {n['feature']} ." for n in feats])
        lines.append("Categories:")
        lines.extend([f"- {n['category']} a skgt:Category ." for n in categories])
        lines.append("Other nodes:")
        for n in entities:
            joiner = " ;\n  " + " " * len(n["node"]) + " a "
            lines.append(
                joiner.join([f"- {n['node']} a {n['types'][0]}"] + n["types"][1:])
                + " ."
            )

        return "\n".join(lines), metadata

    async def _run_sparql_queries(
        self,
        queries: list[str | None],
        result_limit: int,
    ) -> list[str | None]:

        async def _run_query(query: str) -> str | None:
            assert self.rdfoxhandle is not None
            try:
                output = await self.rdfoxhandle.query(
                    query,
                    extra_params={
                        "offset": "0",
                        "limit": str(result_limit),
                    },
                    response_format="application/sparql-results+json",
                )
                if output.status_code != 200:
                    raise RuntimeError(output.text)
                return output.text
            except Exception as e:  # pylint: disable=broad-exception-caught
                L.warning(
                    "Failed to run RDFox query:\n%s\nDue to following error:\n%s",
                    query,
                    repr(e),
                )
                return None

        outputs: list[str | None] = []
        for query in queries:
            if query is None:
                outputs.append(None)
                continue
            outputs.append(await _run_query(query))
        return outputs

    async def _format_sparql_query_responses(
        self,
        responses: list[str | None],
        result_limit: int,
    ) -> list[str | None]:

        def _parse_response(response: str) -> tuple[list[str], list[list[str]]] | None:
            try:
                data = json.loads(response)
                vars_ = data["head"]["vars"]
                rows = []
                for binding in data["results"]["bindings"]:
                    row = []
                    for var in vars_:
                        if var not in binding:
                            row.append("")
                        elif binding[var]["type"] == "uri":
                            row.append(
                                collapse_prefix(
                                    URIRef(binding[var]["value"]),
                                    prefixes=["rdf", "skg", "skgt"],
                                )
                            )
                        else:
                            row.append(binding[var]["value"])
                    rows.append(row)
                return vars_, rows
            except Exception as e:  # pylint: disable=broad-exception-caught
                L.warning(
                    "Failed to parse RDFox response:\n%s\nDue to following error:\n%s",
                    response,
                    repr(e),
                )
                return None

        def _format_response(vars_: list[str], rows: list[list[str]]) -> str:
            writer = MarkdownTableWriter(
                table_name="",
                headers=vars_,
                value_matrix=rows,
                margin=1,
            )
            return writer.dumps().strip()

        # Parsing and formatting algorithm:
        # 1. Query produced error -> discarded
        # 2. Query produced too many results (limit) -> discarded
        # 3. Query produced no results and other queries
        #    got discarded or produced no results -> discarded

        parsed_responses: list[tuple[list[str], list[list[str]]] | None] = []
        for response in responses:
            if response is None:
                parsed_responses.append(None)
                continue
            parsed_response = _parse_response(response)
            if parsed_response is None:
                parsed_responses.append(None)
                continue
            _, rows = parsed_response
            if len(rows) >= result_limit:
                L.warning(
                    "Skipped RDFox response:\n"
                    "Due to exceeded limit on number of results (probably noise)",
                )
                L.debug(
                    response,
                )
                parsed_responses.append(None)
                continue
            parsed_responses.append(parsed_response)

        if all(len(pr[1]) == 0 for pr in parsed_responses if pr is not None):
            L.warning("All remaining RDFox response are empty")
            return [None] * len(parsed_responses)

        return [pr if pr is None else _format_response(*pr) for pr in parsed_responses]


@dataclass
class AgenticSPARQLDeps:
    params: SparqlQueryConfig
    rdfoxhandle: RDFoxCSHandle


@dataclass
class AgenticSparqlRetriever(Retriever):
    tokenizer: Tokenizer
    entities_vdb: VectorStorage
    chunks_vdb: VectorStorage
    text_chunks: BaseKVStorage
    entity_base: pd.DataFrame
    specs_vdb: VectorStorage
    features_vdb: VectorStorage
    categories_vdb: VectorStorage
    rdfoxhandle: RDFoxCSHandle | None
    prompts: dict[str, str]
    llm: LLM
    agent: Agent[AgenticSPARQLDeps, str]

    # def __post_init__(self):
    #     # AgenticSparqlRetriever._extract_sparql_query = self.agent.tool_plain(
    #     #     AgenticSparqlRetriever._extract_sparql_query
    #     # )
    #     # AgenticSparqlRetriever.run_sparql_query = self.agent.tool(
    #     #     AgenticSparqlRetriever.run_sparql_query
    #     # )

    #     AgenticSparqlRetriever.validate_sparql = self.agent.output_validator(
    #         AgenticSparqlRetriever.validate_sparql
    #     )

    def __post_init__(self):
        AgenticSparqlRetriever.validate_sparql = self.agent.output_validator(
            AgenticSparqlRetriever.validate_sparql
        )  # type: ignore

    async def __call__(
        self, query: str, params: SparqlQueryConfig, **kwargs
    ) -> RetrievalResult:
        assert self.rdfoxhandle is not None
        sparql_generation_context: str = kwargs.get("sparql_generation_context", "")
        instructions = f"""ADDITIONAL SCHEMA INFORMATION:
        Below is the list of some nodes present in the graph that may be relevant for provided
        question, but this is only a selection so you do not have to use all of them, also there
        may exist some other relevant nodes in the graph not present in that list.
        {sparql_generation_context}"""

        metadata: dict[str, Any] = kwargs.get("metadata", {})

        deps = AgenticSPARQLDeps(params=params, rdfoxhandle=self.rdfoxhandle)

        with capture_run_messages() as messages:
            agent_run_result: AgentRunResult = await self.agent.run(
                query, deps=deps, instructions=instructions
            )
            sparql_llm_responses: List[str] = [agent_run_result.output]

            metadata["sparql_llm_responses"] = sparql_llm_responses
            metadata["sparql_agent_messages"] = (
                messages  # agent_run_result.all_messages()
            )

            sparql_queries: list[str | None] = [
                AgenticSparqlRetriever._extract_sparql_query(r)
                for r in sparql_llm_responses
            ]

            outputs: list[str | None] = await self._run_sparql_queries(
                sparql_queries, params.query_result_limit
            )
            metadata["sparql_rdfox_responses"] = outputs
            outputs = await self._format_sparql_query_responses(
                responses=outputs, result_limit=params.query_result_limit
            )

            context_: list[str] = [
                "To gather information for answering the question relevant "
                "knowledge graph was queried. Below are the queries and "
                f"received answers (limited to {params.query_result_limit} results):"
            ]
            for sparql_query, outputstr in zip(sparql_queries, outputs, strict=True):
                if outputstr is None:
                    continue
                context_.append(f"Query:\n{sparql_query}\n\nAnswer:\n{outputstr}\n###")

            if len(context_) == 1:
                context = "No information could be retrieved from knowledge graph"
                should_fallback = True
            else:
                context = "\n\n".join(context_)
                should_fallback = False

            metadata["should_fallback_to_tkg"] = should_fallback

            return RetrievalResult(
                context=[context],
                metadata=metadata,
            )

    # pylint: disable=line-too-long
    @staticmethod
    def _extract_sparql_query(output: str) -> str | None:
        """
        Extracts a SPARQL query enclosed within <SPARQL> tags from the given string.

        This method searches for the first occurrence of a substring delimited by
        <SPARQL> and </SPARQL> tags in the input string. If found, it returns the
        stripped content inside the tags; otherwise, returns None.

        Parameters:
            output (str): The string from which to extract the SPARQL query.

        Returns:
            str | None: The extracted SPARQL query as a string (without surrounding tags),
                        or None if no matching pattern is found.

        Example Usage:
            >>> result = SomeClass._extract_sparql_query("Some text <SPARQL>SELECT * WHERE { ?s ?p ?o }</SPARQL>")
            >>> print(result)
            "SELECT * WHERE { ?s ?p ?o }"

        Exceptions:
            None: This method does not raise any exceptions.

        Notes:
            - The search is case-sensitive and uses DOTALL flag to match across newlines.
            - Only the first occurrence of the pattern is extracted.
            - Returns None immediately if the pattern is not found.
        """
        if output is None:
            return None
        if match := re.search(
            r"<SPARQL>((?:(?!</SPARQL>).)*)</SPARQL>", output, flags=re.DOTALL
        ):
            return match.group(1).strip()
        return None

    async def _run_sparql_queries(
        self, queries: list[str | None], result_limit: int
    ) -> list[str | None]:

        outputs: list[str | None] = []
        for query in queries:
            if query is None:
                outputs.append(None)
                continue
            outputs.append(
                await self._run_sparql_query(
                    query, result_limit=result_limit, rdfoxhandle=self.rdfoxhandle
                )
            )
        return outputs

    @staticmethod
    async def _run_sparql_query(
        query: str, result_limit: int, rdfoxhandle: RDFoxCSHandle | None
    ):
        assert rdfoxhandle is not None
        try:
            output = await rdfoxhandle.query(
                query,
                extra_params={
                    "offset": "0",
                    "limit": str(result_limit),
                },
                response_format="application/sparql-results+json",
            )
            if output.status_code != 200:
                raise ModelRetry(output.text)
            return output.text
        except Exception as e:  # pylint: disable=broad-exception-caught
            L.warning(
                "Failed to run RDFox query:\n%s\nDue to following error:\n%s",
                query,
                repr(e),
            )
            raise ModelRetry(repr(e)) from e

    @staticmethod
    async def validate_sparql(ctx: RunContext[AgenticSPARQLDeps], query: str) -> str:
        L.debug("Validating SPARQL query: %s", query)
        if not query or query.strip() == "":
            L.warning("Raising ModelRetry due to empty query!")
            raise ModelRetry("Generated empty query!")

        extracted_query = AgenticSparqlRetriever._extract_sparql_query(query)

        if not extracted_query or extracted_query.strip() == "":
            L.warning("Raising ModelRetry due to missing <SPARQL> tags!")
            raise ModelRetry("No <SPARQL> tags found in the generated response!")

        query_result = await AgenticSparqlRetriever._run_sparql_query(
            extracted_query, ctx.deps.params.query_result_limit, ctx.deps.rdfoxhandle
        )
        if not query_result or query_result.strip() == "":
            L.warning("Raising ModelRetry due to empty query result!")
            raise ModelRetry("Got empty result from running the SPARQL query.")

        try:
            _ = AgenticSparqlRetriever._parse_response(query_result)
        except Exception as e:
            L.warning(
                "Raising ModelRetry due to error during parsing of SPARQL query: %s!",
                repr(e),
            )
            raise ModelRetry(
                "During parsing of SPARQL query response following exception was encountered:"
                f" {repr(e)}"
            ) from e

        return query

    @staticmethod
    async def run_sparql_query(
        ctx: RunContext[AgenticSPARQLDeps], query: str
    ) -> str | None:
        return await AgenticSparqlRetriever._run_sparql_query(
            query, ctx.deps.params.query_result_limit, ctx.deps.rdfoxhandle
        )

    @staticmethod
    def _parse_response(response: str) -> tuple[list[str], list[list[str]]] | None:
        data = json.loads(response)
        vars_ = data["head"]["vars"]
        rows = []
        for binding in data["results"]["bindings"]:
            row = []
            for var in vars_:
                if var not in binding:
                    row.append("")
                elif binding[var]["type"] == "uri":
                    row.append(
                        collapse_prefix(
                            URIRef(binding[var]["value"]),
                            prefixes=["rdf", "skg", "skgt"],
                        )
                    )
                else:
                    row.append(binding[var]["value"])
            rows.append(row)
        return vars_, rows

    @staticmethod
    def _format_response(vars_: list[str], rows: list[list[str]]) -> str:
        writer = MarkdownTableWriter(
            table_name="",
            headers=vars_,
            value_matrix=rows,
            margin=1,
        )
        return writer.dumps().strip()

    async def _format_sparql_query_responses(
        self,
        responses: list[str | None],
        result_limit: int,
    ) -> list[str | None]:
        # Parsing and formatting algorithm:
        # 1. Query produced error -> discarded
        # 2. Query produced too many results (limit) -> discarded
        # 3. Query produced no results and other queries
        #    got discarded or produced no results -> discarded

        parsed_responses: list[tuple[list[str], list[list[str]]] | None] = []
        for response in responses:
            if response is None:
                parsed_responses.append(None)
                continue
            try:
                parsed_response = AgenticSparqlRetriever._parse_response(response)
            except Exception as e:  # pylint: disable=broad-exception-caught
                L.warning(
                    "Failed to parse RDFox response:\n%s\nDue to following error:\n%s",
                    response,
                    repr(e),
                )
                parsed_response = None
            if parsed_response is None:
                parsed_responses.append(None)
                continue
            _, rows = parsed_response
            if len(rows) >= result_limit:
                L.warning(
                    "Skipped RDFox response:\n"
                    "Due to exceeded limit on number of results (probably noise)",
                )
                L.debug(response)
                parsed_responses.append(None)
                continue
            parsed_responses.append(parsed_response)

        if all(len(pr[1]) == 0 for pr in parsed_responses if pr is not None):
            L.warning("All remaining RDFox response are empty")
            return [None] * len(parsed_responses)

        return [
            pr if pr is None else AgenticSparqlRetriever._format_response(*pr)
            for pr in parsed_responses
        ]
