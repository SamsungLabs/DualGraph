# pylint: disable=no-member

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from rdflib import Literal, URIRef, Variable
from tqdm.asyncio import tqdm

from .alignement import do_alignment
from .base import (
    LLM,
    LOGGER,
    BaseKVStorage,
    Chunk,
    Chunker,
    ContentHandler,
    Embedder,
    Prescience,
    QueryResult,
    RetrievalResult,
    Router,
    StorageNameSpace,
    Tokenizer,
    VectorStorage,
)
from .chunkers import init_chunker
from .configurations import GraphRAGConfig, QueryConfig, RetrievalMethod
from .embedders import init_embedder
from .llms import init_llm
from .retrievers import (
    AgenticSPARQLDeps,
    AgenticSparqlRetriever,
    ClosedBookRetriever,
    NaiveRetriever,
    SparqlRetriever,
    TKGRetriever,
)
from .routers import init_router
from .storage import (
    JsonKVStorage,
    MongoKVStorage,
    NameCache,
    RDFoxCSHandle,
    dump_entities_to_rdfox,
    get_vector_storage,
    knowledge_to_triples,
    make_rdfox_ids,
)
from .tokenizers import init_tokenizer
from .usage_monitors import UsageMonitor
from .utils import (
    TimeLogger,
    build_conversation,
    compute_mdhash_id,
    get_run_id,
    get_unique_id,
    is_retrieval_result_empty,
    limit_async_func_call_asyncio,
    merge_retrieval_results,
    parse_json_list,
    remove_file_if_exists,
)

L = logging.getLogger(LOGGER)

MAX_ASYNC_QUERY = 16

KEY_SET = set(["entity_name", "entity_description"])


class ExtractedEntity(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    chunk_id: list[str]
    model_config = ConfigDict(extra="forbid")


@dataclass
class DualGraphRAG:
    # pylint: disable=too-many-instance-attributes

    id: str
    working_dir: Path
    config: GraphRAGConfig

    prompts: dict[str, str]

    usage_monitor: UsageMonitor | None

    tokenizer: Tokenizer
    chunker: Chunker
    embedder: Embedder
    llm: LLM
    router: Router | None

    entities_vdb: VectorStorage
    chunks_vdb: VectorStorage
    text_chunks: BaseKVStorage

    specs_vdb: VectorStorage
    features_vdb: VectorStorage
    categories_vdb: VectorStorage

    entity_base: pd.DataFrame

    rdfoxhandle: RDFoxCSHandle

    closedbook_retriever: ClosedBookRetriever
    naive_retriever: NaiveRetriever
    tkg_retriever: TKGRetriever
    sparql_retriever: SparqlRetriever
    agentic_sparql_retriever: AgenticSparqlRetriever

    @classmethod
    async def in_working_dir(
        cls,
        working_dir: Path,
        config: GraphRAGConfig,
        reset: bool = False,
        run_name: str | None = None,
    ):
        # pylint: disable=too-many-locals
        if reset:
            remove_file_if_exists(working_dir / "kv_store_text_chunks.json")
            remove_file_if_exists(working_dir / "entities.pqt")

        stage = os.environ.get("GRAPHRAG_STAGE", "_")
        run_id = get_run_id(working_dir, run_name, run_mode=stage)
        usage_monitor = UsageMonitor()

        prompts = {}
        prompt_paths = config.general.prompts.model_dump()
        for key, prompt_path in prompt_paths.items():
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompts[key] = f.read()

        tokenizer = await init_tokenizer(config.general.tokenizer)
        chunker = await init_chunker(
            config.index.chunker,
            tokenizer=tokenizer,
        )

        llm_response_cache = (
            await MongoKVStorage.from_environ_credentials(
                namespace="llm_response_cache", run_id=run_id
            )
            if config.general.use_cache_llm
            else None
        )
        llm = await init_llm(
            config.general.llm,
            cache=llm_response_cache,
            usage_monitor=usage_monitor,
        )

        embedder_response_cache = (
            await MongoKVStorage.from_environ_credentials(
                namespace="embedder_response_cache", run_id=run_id
            )
            if config.general.use_cache_embedder
            else None
        )
        embedder = await init_embedder(
            config.general.embedder,
            cache=embedder_response_cache,
        )

        router = await init_router(llm, config.query.router)

        entities_vdb = await get_vector_storage(
            config.general.vector_storage_name
        ).in_working_dir(
            working_dir=working_dir,
            namespace="entities",
            embedder=embedder,
            meta_fields={
                "entity_id": "string",
                "rdfox_entity_id": "string",
                "in_triple": "bool",
            },
            reset=reset,
        )
        chunks_vdb = await get_vector_storage(
            config.general.vector_storage_name
        ).in_working_dir(
            working_dir=working_dir,
            namespace="chunks",
            embedder=embedder,
            reset=reset,
        )
        text_chunks = JsonKVStorage(
            namespace="text_chunks",
            global_config={"working_dir": working_dir.absolute()},
        )

        specs_vdb = await get_vector_storage(
            config.general.vector_storage_name
        ).in_working_dir(
            working_dir=working_dir,
            namespace="specs",
            embedder=embedder,
            meta_fields={
                "nodes": "string",
            },
            reset=reset,
        )
        features_vdb = await get_vector_storage(
            config.general.vector_storage_name
        ).in_working_dir(
            working_dir=working_dir,
            namespace="features",
            embedder=embedder,
            meta_fields={
                "nodes": "string",
            },
            reset=reset,
        )
        categories_vdb = await get_vector_storage(
            config.general.vector_storage_name
        ).in_working_dir(
            working_dir=working_dir,
            namespace="categories",
            embedder=embedder,
            batch_size=config.general.embedder.batch_size,
            meta_fields={
                "nodes": "string",
            },
            reset=reset,
        )

        entity_base = (
            pd.read_parquet(working_dir / "entities.pqt")
            if (working_dir / "entities.pqt").exists()
            else pd.DataFrame()
        )

        closedbook_retriever = ClosedBookRetriever()

        naive_retriever = NaiveRetriever(
            tokenizer=tokenizer, chunks_vdb=chunks_vdb, text_chunks=text_chunks
        )

        tkg_retriever = TKGRetriever(
            tokenizer=tokenizer,
            entities_vdb=entities_vdb,
            chunks_vdb=chunks_vdb,
            text_chunks=text_chunks,
            entity_base=entity_base,
        )

        _cfg = config.general.graph_storage
        rdfoxhandle = RDFoxCSHandle(
            rdfox_cert_verify=bool(_cfg.rdfox_cert_verify),
            rdfox_dstore=str(_cfg.rdfox_dstore),
            rdfox_url=str(_cfg.rdfox_url),
            rdfox_user=str(_cfg.rdfox_user),
            rdfox_passphrase=str(_cfg.rdfox_passphrase),
            rdfox_graph="",
        )
        await rdfoxhandle.connect()

        sparql_retriever = SparqlRetriever(
            tokenizer=tokenizer,
            entities_vdb=entities_vdb,
            chunks_vdb=chunks_vdb,
            text_chunks=text_chunks,
            entity_base=entity_base,
            specs_vdb=specs_vdb,
            features_vdb=features_vdb,
            categories_vdb=categories_vdb,
            prompts=prompts,
            llm=llm,
            rdfoxhandle=rdfoxhandle,
        )
        agentic_sparql_retriever = AgenticSparqlRetriever(
            tokenizer=tokenizer,
            entities_vdb=entities_vdb,
            chunks_vdb=chunks_vdb,
            text_chunks=text_chunks,
            entity_base=entity_base,
            specs_vdb=specs_vdb,
            features_vdb=features_vdb,
            categories_vdb=categories_vdb,
            prompts=prompts,
            llm=llm,
            rdfoxhandle=rdfoxhandle,
            agent=Agent[AgenticSPARQLDeps, str](
                OpenAIChatModel(
                    config.general.llm.model,
                    settings=ModelSettings(
                        max_tokens=int(config.general.llm.max_token_size * 0.5)
                    ),
                    provider=OpenAIProvider(
                        base_url=config.general.llm.base_urls[0],
                        api_key=config.general.llm.api_keys[0][0],
                    ),
                ),
                deps_type=AgenticSPARQLDeps,
                retries=config.query.agentic.sparql_retries,
                instructions=prompts[
                    "agentic_sparql_kg_query_generation_prompt_template"
                ],
            ),
        )

        return cls(
            id=run_id,
            working_dir=working_dir,
            config=config,
            prompts=prompts,
            usage_monitor=usage_monitor,
            tokenizer=tokenizer,
            chunker=chunker,
            embedder=embedder,
            llm=llm,
            router=router,
            entities_vdb=entities_vdb,
            chunks_vdb=chunks_vdb,
            text_chunks=text_chunks,
            specs_vdb=specs_vdb,
            features_vdb=features_vdb,
            categories_vdb=categories_vdb,
            entity_base=entity_base,
            rdfoxhandle=rdfoxhandle,
            closedbook_retriever=closedbook_retriever,
            naive_retriever=naive_retriever,
            tkg_retriever=tkg_retriever,
            sparql_retriever=sparql_retriever,
            agentic_sparql_retriever=agentic_sparql_retriever,
        )

    @TimeLogger.async_time_log(log_category="block")
    async def insert(self, input_: Iterable[str], handler: ContentHandler):
        # pylint: disable=too-many-statements
        _ncache = NameCache()

        docs = []
        injects: list[Prescience] = []
        for item_id in input_:
            for metadata in handler.get_metadatas(item_id):
                docs.append(handler.get_content(metadata))
            injects.extend(handler.get_prescience(item_id))
        L.info("Loaded content for %d documents", len(docs))

        if self.config.general.graph_storage.rdfox_drop_on_store:
            await self.rdfoxhandle.drop_old()
            L.info("Dropped RDFox storage")

        if len(injects) > 0:
            entities_in_triples = set()
            all_triples = set()
            all_datalogs = set()
            for knowledge in injects:
                triples, datalogs, entities = knowledge_to_triples(knowledge, _ncache)
                all_triples.update(triples)
                all_datalogs.update(datalogs)
                entities_in_triples.update(entities)
            await self.rdfoxhandle.upsert(list(all_triples))
            L.info(
                "Exported %d triples from structured data to RDFox instance",
                len(all_triples),
            )
            await self.rdfoxhandle.push_datalog("\n".join(all_datalogs))
            L.info("Applied %d datalog rules to RDFox instance", len(all_datalogs))
        else:
            entities_in_triples = set()

        chunks: list[Chunk] = [
            Chunk(
                id=get_unique_id(prefix="chunk-"),
                content=c,
                doc_id=str(i),
            )
            for i, cs in enumerate(await self.chunker(docs))
            for c in cs
        ]
        L.info("Chunked documents into %d chunks", len(chunks))

        inserting_chunks = {
            c.id: {"content": c.content, "id": c.id, "doc_id": c.doc_id} for c in chunks
        }
        L.info("Inserting chunks to KV Storage...")
        await TimeLogger.async_time_log(log_category="block", tag="chunk_upsert")(
            self.text_chunks.upsert
        )(inserting_chunks)
        inserting_chunks_vdb = [
            {"item": c.id, "content": c.content, "id": c.id, "doc_id": c.doc_id}
            for c in chunks
        ]
        L.info("Inserting chunks for VectorRAG...")
        await TimeLogger.async_time_log(log_category="block", tag="vdb_chunk_upsert")(
            self.chunks_vdb.upsert
        )(inserting_chunks_vdb)

        L.info("Entity extraction...")
        results = await tqdm.gather(
            *[self._index_chunk(c) for c in chunks], desc="Entity extraction"
        )
        output_df = pd.DataFrame(
            [e.model_dump() for entities in results for e in entities]
        )
        L.info("Extracted %d entities", len(output_df))

        def join_descriptions(description_list: Iterable[str]) -> str:
            # pylint: disable=unsubscriptable-object
            joined = " ".join((desc for desc in description_list))
            # Trim too long texts
            tokens = self.tokenizer.encode([joined])[0]
            limit = int(0.95 * self.config.general.llm.max_token_size)
            if len(tokens) > limit:
                joined = self.tokenizer.decode([tokens[:limit]])[0]
            return joined

        self.entity_base = output_df.groupby("name").agg(
            {"description": join_descriptions, "chunk_id": "sum"}
        )
        self.entity_base["name"] = self.entity_base.index
        self.entity_base = await self._summarize_descriptions(self.entity_base)
        L.info("Extracted %d unique entities", len(self.entity_base))

        self.entity_base = make_rdfox_ids(
            self.entity_base, prefix="skg", _ncache=_ncache
        )
        L.info("Generated entity IDs for RDFox")

        L.info("Upserting entities into VDB...")
        data_for_vdb = [
            {
                "item": compute_mdhash_id(row["rdfox_id"]),
                "content": row["description"],
                "entity_id": entity_id,
                "rdfox_entity_id": row["rdfox_id"],
                "in_triple": f"skg:{row['rdfox_id']}" in entities_in_triples,
            }
            for entity_id, row in self.entity_base.iterrows()
        ]
        L.info(
            "Aligned %d entities out of %d TKG entities and %d SKG entities",
            sum(v["in_triple"] for v in data_for_vdb),
            len(self.entity_base),
            len(entities_in_triples),
        )
        await TimeLogger.async_time_log(
            log_category="block", tag="entity_vdb_upsertion"
        )(self.entities_vdb.upsert)(data_for_vdb)

        await dump_entities_to_rdfox(self.entity_base, self.rdfoxhandle)
        L.info("Exported entities to RDFox instance")

        if self.config.index.do_contrastive_alignment:
            L.info("Started alignement of SKG and TKG using contrastive aligner")
            aligned_df = await self._align_entities_contrastively()
            L.info("Aligned %d entities", len(aligned_df))
            L.info("Upserting aligned entities into VDB...")
            data_for_vdb = [
                {
                    "item": compute_mdhash_id(skg_id),
                    "content": self.entity_base[
                        self.entity_base["rdfox_id"] == row["aligned"]
                    ]["description"].tolist()[0],
                    "entity_id": self.entity_base[
                        self.entity_base["rdfox_id"] == row["aligned"]
                    ]["name"].tolist()[0],
                    "rdfox_entity_id": skg_id,
                    "in_triple": True,
                }
                for skg_id, row in aligned_df.iterrows()
            ]
            await TimeLogger.async_time_log(
                log_category="block", tag="entity_vdb_upsertion"
            )(self.entities_vdb.upsert)(data_for_vdb)

        await self._populate_specs_vdb()
        await self._populate_features_vdb()
        await self._populate_categories_vdb()

        await self._insert_done()

    async def _insert_done(self):
        self.entity_base.to_parquet(self.working_dir / "entities.pqt")

        storages = [
            self.entities_vdb,
            self.chunks_vdb,
            self.text_chunks,
            self.specs_vdb,
            self.features_vdb,
            self.categories_vdb,
        ]
        tasks = [
            cast(StorageNameSpace, s).index_done_callback()
            for s in storages
            if s is not None
        ]
        await asyncio.gather(*tasks)

    async def _index_chunk(self, chunk: Chunk) -> list[ExtractedEntity]:
        extraction_prompt = self.prompts["extraction_prompt_template"]
        filled_prompt = extraction_prompt.format(input_text=chunk.content)
        out, _ = await self.llm(
            build_conversation(prompt=filled_prompt),
            log_data={"phase": "entity extraction"},
        )
        if out is None:
            out = ""
        parsed_out = parse_json_list(out)
        output_entities = []
        for dict_ in parsed_out:
            try:
                entity = ExtractedEntity(
                    name=dict_["entity_name"],
                    description=dict_["entity_description"],
                    chunk_id=[chunk.id],
                )
                output_entities.append(entity)
            except (ValidationError, KeyError, TypeError):
                pass
        return output_entities

    @TimeLogger.async_time_log(log_category="block")
    async def _summarize_descriptions(self, entity_base: pd.DataFrame) -> pd.DataFrame:
        entities_to_summarize = entity_base[
            entity_base["description"].apply(len)
            > self.config.index.extraction.summarization_threshold
        ]

        results = await tqdm.gather(
            *[
                self._summarize_single_description(
                    str(entity_id), row["description"], row["chunk_id"]
                )
                for entity_id, row in entities_to_summarize.iterrows()
            ],
            desc="Description summarization",
        )
        summarized_entities = [e.model_dump() for e in results]
        for entity_dict in summarized_entities:
            entity_name = entity_dict["name"]
            description = entity_dict["description"]
            entity_base.loc[entity_name, "description"] = description
        return entity_base

    async def _summarize_single_description(
        self, entity_name: str, entity_description: str, chunk_id: list[str]
    ) -> ExtractedEntity:
        summarization_prompt = self.prompts["summarization_prompt_template"]
        formatted_prompt = summarization_prompt.format(
            entity=entity_name,
            description=entity_description,
        )
        conversation_history = build_conversation(prompt=formatted_prompt)
        out, _ = await self.llm(
            conversation_history, log_data={"phase": "description summarization"}
        )
        if out is None:
            out = ""
        parsed_out = parse_json_list(out)
        try:
            return ExtractedEntity(
                name=entity_name,
                description=parsed_out[0] if len(parsed_out) > 0 else "",
                chunk_id=chunk_id,
            )
        except (ValidationError, IndexError):
            L.warning(
                "Failed to summarize entity description, fallback onto truncation"
            )
            return ExtractedEntity(
                name=entity_name,
                description=entity_description[
                    : self.config.index.extraction.summarization_threshold
                ],
                chunk_id=chunk_id,
            )

    async def _populate_specs_vdb(self) -> None:
        assert self.rdfoxhandle is not None

        query = """
            SELECT DISTINCT ?section ?sectionName ?entry ?entryName ?value ?valueName
            WHERE {
                ?spec skg:inSection ?section ;
                      skg:inEntry ?entry ;
                      skg:hasValue ?value .
                ?section skg:hasName ?sectionName .
                ?entry skg:hasName ?entryName .
                ?value skg:hasName ?valueName .
            }
            """
        result = self.rdfoxhandle.parse_result(await self.rdfoxhandle.query(query))

        def _linearize_spec(names: dict[str, str]) -> str:
            return (
                f"In the product specification, the \"{names['entryName']}\" entry "
                f"in the \"{names['sectionName']}\" section "
                f"has the value \"{names['valueName']}\""
            )

        specs = []
        for i, binding in enumerate(result.bindings):
            nodes = {}
            for var in ["section", "entry", "value"]:
                ref = binding[Variable(var)]
                assert isinstance(ref, URIRef)
                nodes[var] = f"skg:{ref.fragment}"
            names = {}
            for var in ["sectionName", "entryName", "valueName"]:
                ref = binding[Variable(var)]
                assert isinstance(ref, Literal)
                names[var] = ref.value
            specs.append(
                {
                    "item": str(i),
                    "nodes": json.dumps(nodes),
                    "content": _linearize_spec(names),
                }
            )

        L.info("Upserting %d specs into VDB...", len(specs))
        await TimeLogger.async_time_log(
            log_category="block", tag="specs_vdb_upsertion"
        )(self.specs_vdb.upsert)(specs)

    async def _populate_features_vdb(self) -> None:
        assert self.rdfoxhandle is not None
        query = """
            SELECT DISTINCT ?feature ?featureName
            WHERE {
                ?product skg:hasFeature ?feature .
                ?feature skg:hasName ?featureName .
            }
        """

        result = self.rdfoxhandle.parse_result(await self.rdfoxhandle.query(query))

        def _linearize_feature(names: dict[str, str]) -> str:
            return f"The product has \"{names['featureName']}\" feature"

        features = []
        for i, binding in enumerate(result.bindings):
            ref = binding[Variable("feature")]
            assert isinstance(ref, URIRef)
            nodes = {"feature": f"skg:{ref.fragment}"}
            ref = binding[Variable("featureName")]
            assert isinstance(ref, Literal)
            names = {"featureName": ref.value}
            features.append(
                {
                    "item": str(i),
                    "nodes": json.dumps(nodes),
                    "content": _linearize_feature(names),
                }
            )

        L.info("Upserting %d features into VDB...", len(features))
        await TimeLogger.async_time_log(
            log_category="block", tag="features_vdb_upsertion"
        )(self.features_vdb.upsert)(features)

    async def _populate_categories_vdb(self) -> None:
        assert self.rdfoxhandle is not None
        query = """
            SELECT DISTINCT ?category ?categoryName
            WHERE {
                ?category a skgt:Category ;
                          skg:hasName ?categoryName .
            }
        """

        result = self.rdfoxhandle.parse_result(await self.rdfoxhandle.query(query))

        def _linearize_category(names: dict[str, str]) -> str:
            return names["categoryName"]

        categories = []
        for i, binding in enumerate(result.bindings):
            ref = binding[Variable("category")]
            assert isinstance(ref, URIRef)
            nodes = {"category": f"skg:{ref.fragment}"}
            ref = binding[Variable("categoryName")]
            assert isinstance(ref, Literal)
            names = {"categoryName": ref.value}
            categories.append(
                {
                    "item": str(i),
                    "nodes": json.dumps(nodes),
                    "content": _linearize_category(names),
                }
            )

        L.info("Upserting %d categories into VDB...", len(categories))
        await TimeLogger.async_time_log(
            log_category="block", tag="categories_vdb_upsertion"
        )(self.categories_vdb.upsert)(categories)

    @TimeLogger.async_time_log(log_category="block")
    async def query(
        self,
        queries: list[str],
        params: QueryConfig,
        contexts_gt: list[list[str]] | None = None,
    ) -> list[QueryResult]:
        _ = contexts_gt
        return await tqdm.gather(
            *[self._query_single(q, params) for q in queries], desc="Querying"
        )

    # pylint: disable=too-many-branches
    @limit_async_func_call_asyncio(max_calls=MAX_ASYNC_QUERY)
    @TimeLogger.async_time_log(log_category="block")
    async def _query_single(
        self,
        query: str,
        params: QueryConfig,
    ) -> QueryResult:

        if (
            params.use_retrieval_routing
            and "sparql" in params.router.retrieval_routing_methods
            or "sparql" in params.retrieval_methods
        ):
            sparql_generation_context, metadata = (
                await self.sparql_retriever.retrieve_sparql_generation_context(
                    query,
                    params.sparql,
                    metadata={},
                )
            )
        else:
            sparql_generation_context, metadata = "", {}

        retrieval_methods: list[RetrievalMethod]
        if params.use_retrieval_routing:
            assert self.router is not None
            retrieval_methods = await self.router(query, sparql_generation_context)  # type: ignore
        else:
            retrieval_methods = params.retrieval_methods

        L.debug("retrieval_methods: %s", retrieval_methods)
        retrieval_results = []
        should_fallback_to_tkg = False
        if params.mock_retrieval:
            retrieval_results.append(await self.closedbook_retriever(query, None))
        else:
            for retrieval_method in retrieval_methods:
                match retrieval_method:
                    case "tkg":
                        _retrieval_result = await self.tkg_retriever(query, params.tkg)
                    case "sparql":
                        _retrieval_result = await self.sparql_retriever(
                            query,
                            params.sparql,
                            sparql_generation_context=sparql_generation_context,
                            metadata=metadata,
                        )
                        should_fallback_to_tkg = _retrieval_result.metadata[
                            "should_fallback_to_tkg"
                        ]
                    case "agentic_sparql":
                        _retrieval_result = await self.agentic_sparql_retriever(
                            query,
                            params.sparql,
                            sparql_generation_context=sparql_generation_context,
                            metadata=metadata,
                        )
                        should_fallback_to_tkg = _retrieval_result.metadata[
                            "should_fallback_to_tkg"
                        ]
                    case "naive":
                        _retrieval_result = await self.naive_retriever(
                            query, params.naive
                        )
                    case "closed_book":
                        _retrieval_result = await self.closedbook_retriever(query, None)
                    case _:
                        raise RuntimeError(
                            f"Unsupported retrieval method: {retrieval_methods[0]}"
                        )
                _retrieval_result.metadata["retrieval_method"] = retrieval_method
                retrieval_results.append(_retrieval_result)

        if (
            params.fallback_to_tkg
            and should_fallback_to_tkg
            and len(retrieval_methods) == 1
            and retrieval_methods[0] == "sparql"
        ):
            L.info("No results from SPARQL query for: %s. Falling back to TKG", query)
            retrieval_results.append(
                RetrievalResult(
                    context=[
                        "Because no information was retrieved from knowledge graph, "
                        "here are matching text chunks from source webpage:"
                    ],
                    metadata={"fallback_to_tkg": True},
                )
            )
            retrieval_results.append(await self.tkg_retriever(query, params.tkg))

        retrieval_result = merge_retrieval_results(retrieval_results)

        if is_retrieval_result_empty(retrieval_result) or params.mock_retrieval:
            if not params.mock_retrieval:
                L.warning("Could not retrieve context for query: %s", query)
            return QueryResult(
                answer=self.prompts["no_data_response"],
                retrieval_result=retrieval_result,
                metadata={
                    "retrieval_methods": retrieval_methods,
                },
            )

        querying_prompt = self.prompts["querying_prompt_template"]
        filled_prompt = querying_prompt.format(
            question=query, context="\n".join(retrieval_result.context)
        )
        conversation = build_conversation(
            prompt=filled_prompt,
            max_token_size=int(self.config.general.llm.max_token_size * 0.95),
            tokenizer=self.tokenizer,
        )
        answer, _ = await self.llm(
            conversation,
            log_data={"phase": "response generation"},
        )

        return QueryResult(
            answer=answer,
            retrieval_result=retrieval_result,
            metadata={
                "retrieval_methods": retrieval_methods,
            },
        )

    async def _align_entities_contrastively(self):
        query = """
            SELECT ?S ?P ?O
            WHERE {
                ?S ?P ?O .
                ?S a skgt:SKG_Entity .
            }
        """

        result = await self.rdfoxhandle.query(
            query,
            response_format="application/n-triples",
        )
        assert result.is_success

        with open(self.working_dir / "skg.nt", "w", encoding="utf-8") as f:
            f.write(result.text)

        aligned_df = await do_alignment(
            tkg_entities=self.entity_base,
            llm_func=self.llm,
            embedder=self.embedder,
            working_dir=self.working_dir,
        )
        return aligned_df
