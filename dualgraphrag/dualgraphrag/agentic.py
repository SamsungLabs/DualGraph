# pylint: disable=no-member,too-many-lines

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List

from pydantic_ai import (
    Agent,
    AgentRunResult,
    ModelSettings,
    RunContext,
    UnexpectedModelBehavior,
    capture_run_messages,
)
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelRequest, ToolReturnPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from tqdm.asyncio import tqdm

from .base import QueryResult, RetrievalResult
from .configurations import GraphRAGConfig
from .configurations.query_config import (
    NaiveQueryConfig,
    QueryConfig,
    SparqlQueryConfig,
    TKGQueryConfig,
)
from .rag import DualGraphRAG
from .retrievers.retrievers import (
    AgenticSparqlRetriever,
    ClosedBookRetriever,
    NaiveRetriever,
    SparqlRetriever,
    TKGRetriever,
)

# from .unweaver import Unweaver
from .utils import TimeLogger, limit_async_func_call_asyncio

L = logging.getLogger(__name__)

MAX_ASYNC_QUERY = 16

TOOL_CONTEXT_LENGTH_LIMIT = 12000


@dataclass
class Deps:
    """Dependencies class for the agentic RAG system.

    This class holds all the dependencies needed by the agentic retrieval system,
    including various retrievers for different retrieval strategies and
    configuration parameters for each retrieval method.

    Attributes:
        naive_retriever: Retriever using vector similarity for text chunks
        closedbook_retriever: Retriever for closed-book question answering
        tkg_retriever: Retriever using Textual Knowledge Graph matching
        sparql_retriever: Retriever for generating SPARQL query context
        agentic_sparql_retriever: Retriever for executing SPARQL queries
        params_tkg: Configuration parameters for TKG retrieval
        params_sparql: Configuration parameters for SPARQL retrieval
        params_naive: Configuration parameters for naive retrieval
    """

    # pylint: disable=too-many-instance-attributes
    # Reason: All 8 dependencies are necessary for the agentic system's functionality
    # and represent distinct components that shouldn't be grouped arbitrarily.
    naive_retriever: NaiveRetriever
    closedbook_retriever: ClosedBookRetriever
    tkg_retriever: TKGRetriever
    sparql_retriever: SparqlRetriever
    agentic_sparql_retriever: AgenticSparqlRetriever
    params_tkg: TKGQueryConfig
    params_sparql: SparqlQueryConfig
    params_naive: NaiveQueryConfig


async def retrieve_naive(ctx: RunContext[Deps], question: str) -> List[str]:
    """
    Retrieve relevant text chunks using vector RAG
    Uses embedding similarity to retrieve text chunks relevant for answering the query.
    This method is useful for queries which can be answered just by reading text content
    from up to 10 pages.
    Most of the time it is not the best choice for questions asking for comprehensive list
      of products.
    It retrieves text chunks that are most similar to the query based on their embeddings.

    Args:
        question: natural language query to be searched in a vector database
    """
    L.debug("Calling retrieve_naive")
    result: RetrievalResult = await ctx.deps.naive_retriever(
        question, ctx.deps.params_naive
    )
    return result.context[:TOOL_CONTEXT_LENGTH_LIMIT]


async def retrieve_tkg(ctx: RunContext[Deps], question: str) -> List[str]:
    """
    Retrieve relevant text chunks using Textual knowledge graph -
    use entities extracted from query to match most similar entities and related text chunks
    extracted from text Samsung UK website during indexing.
    Use for queries which don't require a comprehensive list of products
    or filtering based on values.
    Good for general questions about features or products, which can't be answered
    by querying structural knowledge graph with SPARQL queries.

    Args:
        question: natural language query to be searched in a textual knowledge graph
    """
    L.debug("Calling retrieve_tkg")
    result: RetrievalResult = await ctx.deps.tkg_retriever(
        question, ctx.deps.params_tkg
    )
    return result.context[:TOOL_CONTEXT_LENGTH_LIMIT]


async def retrieve_sparql(ctx: RunContext[Deps], question: str) -> List[str]:
    """
    Retrieve relevant context by generating a SPARQL query to query the Knowledge Graph to provide
    a context for answering the user question.
    Useful for agregate or listing queries, for example asking for total count of specified
    entities, average, minimal or maximum value or a
    comprehensive list of products meeting specified conditions, often based on numeric values
    comparison.
    The knowledge graph is constructed from structured data such as tables found
    on the Samsung UK website.",

    Args:
        question: natural language query from a user.
        SPARQL query will be generated based on this question and result of running this query
        will be returned. Do NOT enter SPARQL query here!

    """
    L.debug("Calling retrieve_sparql")
    sparql_generation_context, metadata = (
        await ctx.deps.sparql_retriever.retrieve_sparql_generation_context(
            question,
            ctx.deps.params_sparql,
            metadata={},
        )
    )
    result: RetrievalResult = await ctx.deps.agentic_sparql_retriever(
        question,
        ctx.deps.params_sparql,
        sparql_generation_context=sparql_generation_context,
        metadata=metadata,
    )
    return result.context[:TOOL_CONTEXT_LENGTH_LIMIT]


@dataclass
class AgenticRAG(DualGraphRAG):

    agent_llm_model: ClassVar[OpenAIChatModel]
    agent: ClassVar[Agent[Deps, str]] = Agent(
        None,
        deps_type=Deps,
        output_type=str,
        defer_model_check=True,
        retries=3,
        tools=[retrieve_sparql, retrieve_tkg, retrieve_naive],
        instructions=(
            "You are a customer serivce agent which helps users decide which Samsung products to"
            " choose. You answer mostly using facts from Samsung's UK website and try to be as"
            " accurate and comprehensive as possible. Your answers will be evaluated for both"
            " precision and recall in terms of how well they match the ground truth answers. When"
            " asked to list all products meeting certain criteria, please try to list all matching"
            " products from the retrieved sources and only those. SPARQL retriever may be best for"
            " this kind of questions. For answering most of the questions you should use at least"
            " one of the tools provided to you. Preliminary results indicate that for most of the"
            " questions either retrieve_sparql or retrieve_tkg give the best results."
        ),
    )

    @classmethod
    async def in_working_dir(
        cls,
        working_dir: Path,
        config: GraphRAGConfig,
        reset: bool = False,
        run_name: str | None = None,
    ):
        result = await super().in_working_dir(working_dir, config, reset, run_name)
        cls.agent_llm_model = OpenAIChatModel(
            config.general.llm.model,
            settings=ModelSettings(max_tokens=64000),
            provider=OpenAIProvider(
                base_url=config.general.llm.base_urls[0],
                api_key=config.general.llm.api_keys[0][0],
            ),
        )
        # cls.agent = Agent(cls.agent_llm_model, deps_type=Deps)
        cls.agent = AgenticRAG.agent
        cls.agent.model = cls.agent_llm_model
        return result

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

    @limit_async_func_call_asyncio(max_calls=MAX_ASYNC_QUERY)
    @TimeLogger.async_time_log(log_category="block")
    async def _query_single(
        self,
        query: str,
        params: QueryConfig,
    ) -> QueryResult:

        deps: Deps = Deps(
            naive_retriever=self.naive_retriever,
            closedbook_retriever=self.closedbook_retriever,
            tkg_retriever=self.tkg_retriever,
            sparql_retriever=self.sparql_retriever,
            agentic_sparql_retriever=self.agentic_sparql_retriever,
            params_tkg=params.tkg,
            params_sparql=params.sparql,
            params_naive=params.naive,
        )

        # test_model = TestModel()
        with capture_run_messages() as messages:
            exc = None
            error = None

            try:
                answer = await self.agent.run(query, deps=deps)
            except (UnexpectedModelBehavior, ModelHTTPError) as e:
                L.error("An error occurred: %s", e)
                L.error("cause: %s", repr(e.__cause__))
                L.error("messages: %s", messages)
                exc = e
                error = e.__cause__
                answer = AgentRunResult("error")

            # L.info(test_model.last_model_request_parameters.function_tools)

            L.debug(answer.all_messages_json())

            retrieval_result = [
                str(m.parts[0].content)
                for m in answer.new_messages()
                if isinstance(m, ModelRequest)
                and len(m.parts) > 0
                and isinstance(m.parts[0], ToolReturnPart)
            ]

            metadata = {
                "retrieval_methods": ["agentic"],
                # "messages": answer.all_messages(),
                "messages": messages,
            }
            if exc is not None:
                metadata["exception"] = exc
            if error is not None:
                metadata["error"] = error

            return QueryResult(
                answer=answer.output,
                retrieval_result=RetrievalResult(context=retrieval_result),
                metadata=metadata,
            )


# from pydantic_ai.models.function import AgentInfo
# from pydantic_ai import ModelMessage, ModelResponse, TextPart
# @staticmethod
# def print_schema(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
#     for tool in info.function_tools:
#         print(tool)
#         print(tool.description)
#         print(tool.parameters_json_schema)
#     return ModelResponse(parts=[TextPart("foobar")])
