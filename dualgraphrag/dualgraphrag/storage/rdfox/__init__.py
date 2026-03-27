import asyncio
import logging
from collections import defaultdict
from typing import Any, Literal

import pandas
from more_itertools import chunked
from rdflib import URIRef
from rdflib.query import Result
from rdflib.term import Literal as RDFLiteral

from ...base import LOGGER
from .cleaner import NameCache, clean_id_for_rdfox
from .knowledge import knowledge_to_triples
from .rdfoxhandle import (
    PREFIXES,
    RDFoxCSHandle,
    RDFoxCSHandleBadRequestError,
    collapse_prefix,
)

__all__ = [
    "clean_id_for_rdfox",
    "collapse_prefix",
    "knowledge_to_triples",
    "NameCache",
    "PREFIXES",
    "RDFoxCSHandle",
    "RDFoxCSHandleBadRequestError",
]

L = logging.getLogger(LOGGER)

RDFOX_UPSERTION_BATCH_SIZE = 1000


def make_rdfox_ids(
    base: pandas.DataFrame,
    prefix: Literal["tkg", "skg"],
    _ncache: NameCache | None = None,
) -> pandas.DataFrame:
    if _ncache is None:
        _ncache = NameCache()
    assert "name" in base.columns
    base["rdfox_id"] = base.apply(
        lambda row: clean_id_for_rdfox(row["name"], prefix, _ncache), axis=1
    )
    return base


async def dump_entities_to_rdfox(
    entities: pandas.DataFrame,
    rdfox_handle: RDFoxCSHandle,
):
    assert "rdfox_id" in entities.columns
    data_for_insertion = {
        str(row["rdfox_id"]): (
            str(entity_name),
            str(row["description"]),
        )
        for entity_name, row in entities.iterrows()
    }
    triples: list[str] = []
    for rid, (name, desc) in data_for_insertion.items():
        triples.extend(
            [
                f"skg:{rid} a skgt:UTKG_Entity .",
                f"skg:{rid} skg:hasName {RDFLiteral(name).n3()} .",
                f"skg:{rid} skg:hasDescription {RDFLiteral(desc).n3()} .",
            ]
        )
    for triples_chunk in chunked(triples, RDFOX_UPSERTION_BATCH_SIZE):
        await rdfox_handle.upsert(triples_chunk)
