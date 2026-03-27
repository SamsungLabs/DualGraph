from pydantic import Field

from . import defaults, sections
from .checkable_config import CheckableConfig
from .literals import RetrievalMethod


class NaiveQueryConfig(CheckableConfig):
    retrieve_top_k: int = defaults.NAIVE_RETRIEVE_TOP_K
    chunks_max_token_size: int = defaults.NAIVE_CHUNKS_MAX_TOKEN_SIZE
    chunks_table_format: str = defaults.NAIVE_CHUNKS_TABLE_FORMAT

    @staticmethod
    def get_checks():
        conflicts: list[str] = []
        warnings: list[str] = []
        subconfigs: list[str] = []
        return conflicts, warnings, subconfigs


class TKGQueryConfig(CheckableConfig):
    retrieve_top_k_ents: int = defaults.TKG_RETRIEVE_TOP_K_ENTS
    retrieve_top_k_chunks: int = defaults.TKG_RETRIEVE_TOP_K_CHUNKS
    chunks_max_token_size: int = defaults.TKG_CHUNKS_MAX_TOKEN_SIZE
    chunks_table_format: str = defaults.TKG_CHUNKS_TABLE_FORMAT

    @staticmethod
    def get_checks():
        conflicts: list[str] = []
        warnings: list[str] = []
        subconfigs: list[str] = []
        return conflicts, warnings, subconfigs


class SparqlQueryConfig(CheckableConfig):
    retrieve_top_k_specs: int = defaults.SPARQL_RETRIEVE_TOP_K_SPECS
    retrieve_top_k_features: int = defaults.SPARQL_RETRIEVE_TOP_K_FEATURES
    retrieve_top_k_categories: int = defaults.SPARQL_RETRIEVE_TOP_K_CATEGORIES
    retrieve_top_k_entities: int = defaults.SPARQL_RETRIEVE_TOP_K_ENTITIES

    n_generated_queries: int = defaults.SPARQL_N_GENERATED_QUERIES
    use_beam_search: bool = defaults.SPARQL_USE_BEAM_SEARCH
    query_result_limit: int = defaults.SPARQL_QUERY_RESULT_LIMIT

    @staticmethod
    def get_checks():
        conflicts: list[str] = []
        warnings: list[str] = []
        subconfigs: list[str] = []
        return conflicts, warnings, subconfigs


class RetrievalRouterConfig(CheckableConfig):
    retrieval_routing_method: str = defaults.RETRIEVAL_ROUTING_METHOD
    retrieval_routing_source: str = defaults.RETRIEVAL_ROUTING_SOURCE
    retrieval_routing_select_multiple: bool = defaults.RETRIEVAL_ROUTING_SELECT_MULTIPLE
    retrieval_routing_methods: list[RetrievalMethod] = (
        defaults.RETRIEVAL_ROUTING_METHODS
    )

    @staticmethod
    def get_checks():
        conflicts: list[str] = []
        warnings: list[str] = []
        subconfigs: list[str] = []
        return conflicts, warnings, subconfigs


class AgenticQueryConfig(CheckableConfig):
    rag_retries: int = defaults.AGENTIC_RAG_RETRIES
    sparql_retries: int = defaults.AGENTIC_SPARQL_RETRIES

    @staticmethod
    def get_checks():
        conflicts: list[str] = []
        warnings: list[str] = []
        subconfigs: list[str] = []
        return conflicts, warnings, subconfigs


class QueryConfig(CheckableConfig):
    use_retrieval_routing: bool = defaults.USE_RETRIEVAL_ROUTING
    mock_retrieval: bool = defaults.MOCK_RETRIEVAL
    fallback_to_tkg: bool = defaults.RETRIEVAL_FALLBACK_TO_TKG
    retrieval_methods: list[RetrievalMethod] = Field(
        default_factory=defaults.RETRIEVAL_METHODS.copy
    )

    # Always keep subconfig field names consistent with sections.py!
    naive: NaiveQueryConfig = Field(default_factory=NaiveQueryConfig)
    tkg: TKGQueryConfig = Field(default_factory=TKGQueryConfig)
    sparql: SparqlQueryConfig = Field(default_factory=SparqlQueryConfig)
    router: RetrievalRouterConfig = Field(default_factory=RetrievalRouterConfig)
    agentic: AgenticQueryConfig = Field(default_factory=AgenticQueryConfig)

    def __post_init__(self):
        assert len(self.retrieval_methods) == len(set(self.retrieval_methods))

    @staticmethod
    def get_checks():
        conflicts: list[str] = []
        warnings: list[str] = []
        subconfigs: list[str] = [
            sections.NAIVE,
            sections.TKG,
            sections.SPARQL,
            sections.ROUTER,
        ]
        return conflicts, warnings, subconfigs
