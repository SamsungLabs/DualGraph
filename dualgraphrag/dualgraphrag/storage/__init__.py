from typing import Type

from ..base import VectorStorage
from ..configurations.literals import VectorStorageName
from .kv_json import JsonKVStorage
from .kv_mongo import MongoKVStorage
from .rdfox import (
    NameCache,
    RDFoxCSHandle,
    collapse_prefix,
    dump_entities_to_rdfox,
    knowledge_to_triples,
    make_rdfox_ids,
)
from .vdb_lancedb import LanceDBStorage

__all__ = [
    "JsonKVStorage",
    "LanceDBStorage",
    "MongoKVStorage",
    "NameCache",
    "RDFoxCSHandle",
    "knowledge_to_triples",
    "make_rdfox_ids",
    "collapse_prefix",
    "dump_entities_to_rdfox",
]


def get_vector_storage(vector_storage_name: VectorStorageName) -> Type[VectorStorage]:
    return {
        "lancedb": LanceDBStorage,
    }[vector_storage_name]
