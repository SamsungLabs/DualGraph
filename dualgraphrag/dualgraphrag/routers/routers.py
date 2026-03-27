import json
import logging
import re
from typing import Any, get_args

from pydantic import BaseModel

from ..base import LLM, LOGGER, Router
from ..configurations.literals import RetrievalMethod
from ..configurations.query_config import RetrievalRouterConfig
from ..utils import build_conversation, parse_json_dict

L = logging.getLogger(LOGGER)

URI_PATTERN = re.compile("<https://samsung.com/AIC-Warsaw/SKG#([^>]+)>")


class RetrievalMethodModel(BaseModel):
    retrieval_methods: list[RetrievalMethod]


class BaselineLLMRouter(Router):
    def __init__(self, llm: LLM, config: RetrievalRouterConfig):
        self.llm: LLM = llm
        self.config = config

        # pylint: disable=line-too-long
        self.retrieval_methods = {
            # "closed_book": {
            #     "description": "no retrieval at all. Parametric knowledge of the LLM is enough to answer the user query. No need to retrieve additional context. Use only when the query is simple and the LLM is expected to have enough knowledge to answer it. Don't use for questions which require access to up-to-date information or specific details from the source material.",
            #     "example": """
            #     Example input: "Where are Samsung headquaters located?"
            #     Example output: {"retrieval_methods":["closed_book"]}
            #     """,
            # },
            # "naive": {
            #     "description": "vector RAG - use embedding similarity to retrieve text chunks relevant for answering the query. This method is useful for queries which can be answerable just by reading text content from up to 10 pages. It retrieves text chunks that are most similar to the query based on their embeddings.",
            #     "example": """
            # Example input: "How does Now brief feature work?"
            # Example output: {{"retrieval_methods":["naive"]}}""",
            # },
            "tkg": {
                "description": "Textual knowledge graph - use entities extracted from query to match most similar entities and related text chunks extracted from text during indexing. Use for queries which don't require a comprehensive list of products or filtering based on values. Good for general questions about features or products, which can't be answered by querying structural knowledge graph with SPARQL queries.",
                "examples": [
                    {
                        "input": "Which Samsung phones have latest AI features?",
                        "output": "tkg",
                    },
                    {
                        "input": "What is the most important feature of Galaxy S24?",
                        "output": "tkg",
                    },
                    {
                        "input": "What is the main difference between Galaxy Fold and Flip phones?",
                        "output": "tkg",
                    },
                ],
            },
            "sparql": {
                "description": "Generate a SPARQL query to query the Knowledge Graph to provide a context for answering the user question. Useful for agregate or listing queries, for example asking for total count of specified entities, average, minimal or maximum value or list of entities meeting specified conditions, often based on numeric values comparison. The knowledge graph is constructed from structured data such as tables found in the source text.",
                "examples": [
                    {
                        "input": "Which Samsung phones have at least 50MP cameras?",
                        "output": "sparql",
                    },
                    {
                        "input": "Which fridges cost up to 1000 pounds and have double doors?",
                        "output": "sparql",
                    },
                    {
                        "input": "List phones under 500 pounds with at least 8GB of RAM and 128GB of storage.",
                        "output": "sparql",
                    },
                ],
            },
        }

        self.prompt_template = """
        You are an expert in selecting optimal retrieval method for a RAG system.
        The RAG system consist of a vector database with texts from {source} website and a Knowledge Graph containing entities, relations and attributes extracted from structures parts of the website such as tables.
        You will be given a user query and your task is to select optimal retrieval method to use for answering the user query.
        The RAG system can use following retrieval methods:
        {methods}
        {selection_hint}
        Here is the actual query:
        "{query}"
        """

        self.selection_hint = """
        Please select one retrieval method which will be best for answering the query.
        Don't take into account performance or cost of the retrieval method, just select the one which will be best for answering the query.
        Please provide the answer as a single word containing name of the selected retrieval method.
        """

        self.selection_hint_multiple = """
        Please select at least one method, but only as many methods as necessary to answer the query.
        Please provide the answer as a json dict containing one field called "retrieval_methods" with a value being a list of one or more of above method names.
        """

    extra_body: dict[str, Any]

    async def __call__(
        self, query: str, sparql_generation_context: str = "", **kwargs
    ) -> list[str]:
        if self.config.retrieval_routing_select_multiple:
            selection_hint = self.selection_hint_multiple
            json_schema = RetrievalMethodModel.model_json_schema()
            extra_body = {"guided_json": json_schema}
        else:
            selection_hint = self.selection_hint
            extra_body = {
                "guided_choice": [
                    str(m) for m in self.config.retrieval_routing_methods
                ]  # type: ignore[dict-item]
            }

        L.debug("Querying router with query: %s", query)
        L.debug("Extra body: %s", extra_body)

        if sparql_generation_context:
            self.retrieval_methods["sparql"][
                "node patterns related to the query found in the knowledge graph"
            ] = sparql_generation_context

        filled_prompt = self.prompt_template.format_map(
            {
                "query": query,
                "source": self.config.retrieval_routing_source,
                "nl": "\n",
                "methods": json.dumps(
                    {
                        k: self.retrieval_methods[k]
                        for k in self.config.retrieval_routing_methods
                    },
                    indent=4,
                ),
                "selection_hint": selection_hint,
            }
        )

        completion, _ = await self.llm(
            build_conversation(prompt=filled_prompt),
            log_data={"phase": "Retrieval method router"},
            extra_body=extra_body,
        )
        assert completion is not None

        if self.config.retrieval_routing_select_multiple:
            completion_dict = parse_json_dict(completion)
            retrievers = completion_dict.get("retrieval_methods", [])
            result = [r for r in retrievers if r in get_args(RetrievalMethod)]
        else:
            result = [completion.strip()]

        return result
