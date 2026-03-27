from ..base import LLM, Router
from ..configurations.query_config import RetrievalRouterConfig
from .routers import BaselineLLMRouter


async def init_router(llm: LLM, config: RetrievalRouterConfig) -> Router:
    match config.retrieval_routing_method:
        case "BaselineLLMRouter":
            return BaselineLLMRouter(llm, config)
        case _:
            raise ValueError(
                f'Unrecognized retrieval_routing_method \
                             "{config.retrieval_routing_method}" passed in RetrievalRouterConfig!'
            )
