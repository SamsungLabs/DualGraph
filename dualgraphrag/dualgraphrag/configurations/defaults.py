from .literals import ChunkingStrategy, RetrievalMethod, SystemName, VectorStorageName

DEFAULT_KEY = "somekey-4321"

# MLflowConfig
DEFAULT_REMOTE_MLFLOW_URI: str = "http://IP_ADDRESS:PORT/"
DEFAULT_EXPERIMENT_NAME_INDEX: str = "DualGraph Indexing"
DEFAULT_EXPERIMENT_NAME_QUERY: str = "DualGraph Querying"


#### GENERAL CONFIG

USE_CACHE_LLM: bool = False
USE_CACHE_EMBEDDER: bool = False
USE_WHICH_SYSTEM: SystemName = "dualgraph"

VECTOR_STORAGE_NAME: VectorStorageName = "lancedb"

# LLMConfig
LLM_MAX_ASYNC: int = 8
LLM_BASE_URLS: list[str] = [
    "http://IP_ADDRESS:PORT1/v1",
    "http://IP_ADDRESS:PORT2/v1",
    "http://IP_ADDRESS:PORT3/v1",
    "http://IP_ADDRESS:PORT4/v1",
]
LLM_API_KEYS: list[list[str]] = [
    [DEFAULT_KEY],
    [DEFAULT_KEY],
    [DEFAULT_KEY],
    [DEFAULT_KEY],
]
LLM_MODEL: str = "openai/gpt-oss-120b"
LLM_TIMEOUT: int = 300
LLM_MAX_TOKEN_SIZE: int = 32768
LLM_POSTPROCESS_THINKING: bool = True
LLM_ENABLE_THINKING: bool = False

# GraphStorageConfig
GRAPH_RDFOX_CERT_VERIFY: bool = False
GRAPH_RDFOX_URL: str = "http://IP_ADDRESS:PORT"
GRAPH_RDFOX_DSTORE: str = "rdfox_dstore"
GRAPH_RDFOX_USER: str = ""
GRAPH_RDFOX_PASSPHRASE: str = ""
GRAPH_RDFOX_DROP_ON_STORE: bool = True

# TokenizerConfig
TOKENIZER_MODEL: str = "openai/gpt-oss-120b"

# EmbeddingConfig
EMBEDDER_MAX_ASYNC: int = 8
EMBEDDER_BASE_URLS: list[str] = [
    "http://IP_ADDRESS:PORT1",
    "http://IP_ADDRESS:PORT2",
    "http://IP_ADDRESS:PORT3",
    "http://IP_ADDRESS:PORT4",
]
EMBEDDER_API_KEYS: list[list[str]] = [
    [DEFAULT_KEY],
    [DEFAULT_KEY],
    [DEFAULT_KEY],
    [DEFAULT_KEY],
]
EMBEDDER_MODEL: str = "Qwen/Qwen3-Embedding-4B"
EMBEDDER_DIM: int = 2560
EMBEDDER_MAX_TOKEN_SIZE: int = 40960
EMBEDDER_BATCH_SIZE: int = 32
EMBEDDER_TIMEOUT: int = 300

# PromptPathsConfig
EXTRACTION_PROMPT = "prompts/extraction.txt"
QUERY_PROMPT = "prompts/query.txt"
SUMMARIZATION_PROMPT = "prompts/summarization.txt"
NO_DATA_RESPONSE = "prompts/no_data_response.txt"
SPARQL_KG_QUERY_GENERATION_PROMPT_TEMPLATE = "prompts/sparql_kg_query_generation.txt"
AGENTIC_SPARQL_KG_QUERY_GENERATION_PROMPT_TEMPLATE = (
    "prompts/agentic_sparql_kg_query_generation.txt"
)

#### QUERY CONFIG

USE_RETRIEVAL_ROUTING: bool = False
MOCK_RETRIEVAL: bool = False
RETRIEVAL_FALLBACK_TO_TKG: bool = True
RETRIEVAL_METHODS: list[RetrievalMethod] = ["tkg"]

# NaiveQueryConfig
NAIVE_RETRIEVE_TOP_K: int = 5
NAIVE_CHUNKS_MAX_TOKEN_SIZE: int = 4000
NAIVE_CHUNKS_TABLE_FORMAT: str = "csv"

# TKGQueryConfig
TKG_RETRIEVE_TOP_K_ENTS: int = 10
TKG_RETRIEVE_TOP_K_CHUNKS: int = 5
TKG_CHUNKS_MAX_TOKEN_SIZE: int = 4000
TKG_CHUNKS_TABLE_FORMAT: str = "csv"

# SparqlQueryConfig
SPARQL_RETRIEVE_TOP_K_SPECS: int = 5
SPARQL_RETRIEVE_TOP_K_FEATURES: int = 5
SPARQL_RETRIEVE_TOP_K_CATEGORIES: int = 5
SPARQL_RETRIEVE_TOP_K_ENTITIES: int = 5
SPARQL_N_GENERATED_QUERIES: int = 3
SPARQL_USE_BEAM_SEARCH: bool = False
SPARQL_QUERY_RESULT_LIMIT: int = 100

# RetrievalRouterConfig
RETRIEVAL_ROUTING_METHOD = "BaselineLLMRouter"
RETRIEVAL_ROUTING_SOURCE = "Samsung UK"
RETRIEVAL_ROUTING_SELECT_MULTIPLE = False
RETRIEVAL_ROUTING_METHODS: list[RetrievalMethod] = [
    "sparql",
    "tkg",
]

# Agentic
AGENTIC_RAG_RETRIES: int = 3
AGENTIC_SPARQL_RETRIES: int = 100


#### INDEX CONFIG

DO_CONTRASTIVE_ALIGNEMENT: bool = False

# ChunkingConfig
CHUNK_STRATEGY: ChunkingStrategy = "by_marker"
CHUNK_MAX_TOKEN_SIZE: int = 1500
CHUNK_OVERLAP_TOKEN_SIZE: int = 128

# ExtractionConfig
SUMMARIZATION_THRESHOLD: int = 4000
