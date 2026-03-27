import logging
from dataclasses import dataclass
from io import StringIO
from typing import Any, Iterable

import httpx
from rdflib import Graph, Namespace, URIRef
from rdflib.plugins.sparql.results import xmlresults
from rdflib.query import Result
from rdflib.term import Literal as RDFLiteral

from ...base import LOGGER

L = logging.getLogger(LOGGER)
logging.getLogger("httpx").setLevel(logging.WARNING)

PREFIXES: dict[str, Namespace] = {
    "tkg": Namespace("https://samsung.com/AIC-Warsaw/TKG#"),
    "struct": Namespace("https://samsung.com/AIC-Warsaw/struct#"),
    "chunk": Namespace("https://samsung.com/AIC-Warsaw/chunk#"),
    "rdf": Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    "graph": Namespace("https://samsung.com/AIC-Warsaw/graph#"),
    "skg": Namespace("https://samsung.com/AIC-Warsaw/SKG#"),
    "skgt": Namespace("https://samsung.com/AIC-Warsaw/SKGtype#"),
    "utkg": Namespace("https://samsung.com/AIC-Warsaw/UnweaverTKG#"),
    "schema": Namespace("http://schema.org/"),
}


def collapse_prefix(uri: URIRef, prefixes: list | None = None) -> str:
    graph = Graph()
    prefixes_ = prefixes if prefixes is not None else []
    for prefix in prefixes_:
        graph.bind(prefix, PREFIXES[prefix])
    return uri.n3(graph.namespace_manager)


class RDFoxCSHandleError(Exception):
    pass


class RDFoxCSHandleBadRequestError(RDFoxCSHandleError):
    pass


class RDFoxCSHandleQueryError(RDFoxCSHandleError):
    pass


_drop_executed_in_this_run: bool = False


@dataclass
class RDFoxCSHandle:
    rdfox_cert_verify: bool
    rdfox_dstore: str
    rdfox_url: str
    # Warning: authentication not yet available, these are placeholders
    rdfox_user: str
    rdfox_passphrase: str
    rdfox_graph: str

    def graph_name(self):
        return PREFIXES["graph"][self.rdfox_graph].n3()

    async def connect(self) -> bool:
        # Check if db exists, if not - create.
        # Upload roles and types - that doesn't hurt even if they already exist.
        async with httpx.AsyncClient(verify=self.rdfox_cert_verify) as client:
            headers = {
                "Accept": "application/sparql-results+xml",
                "Content-Type": "application/x-www-form-urlencoded",
            }
            response = await client.post(
                self.rdfox_url + f"/datastores/{self.rdfox_dstore}",
                headers=headers,
            )
        # 409 Created just means were working with an already existing dstore,
        # which is fine unless we were instructed to drop
        if not (response.is_success or response.status_code == 409):
            raise RuntimeError("RDFoxCSHandle: Cannot initiate datastore")
        return response.status_code == 409

    async def drop_old(self):
        # pylint: disable=global-statement
        global _drop_executed_in_this_run
        if _drop_executed_in_this_run:
            L.warning("RDFox was already dropped during this run. Skipping...")
            return
        _drop_executed_in_this_run = True
        L.info("Dropping current graph...")
        async with httpx.AsyncClient(verify=self.rdfox_cert_verify) as client:
            headers = {
                "Accept": "application/sparql-results+xml",
                "Content-Type": "application/x-www-form-urlencoded",
            }
            params = (
                {"graph": PREFIXES["graph"][self.rdfox_graph].n3()}
                if self.rdfox_graph
                else {"facts": "true", "axioms": "true"}
            )
            response = await client.delete(
                self.rdfox_url + f"/datastores/{self.rdfox_dstore}/content",
                params=params,
                headers=headers,
            )
            if not response.is_success:
                raise RuntimeError("RDFoxCSHandle: Cannot drop graph")

    async def query(
        self,
        query: str,
        extra_params: dict[str, str] | None = None,
        response_format: str = "application/sparql-results+xml",
    ) -> httpx.Response:
        async with httpx.AsyncClient(verify=self.rdfox_cert_verify) as client:
            headers = {
                "Accept": response_format,
                "Content-Type": "application/x-www-form-urlencoded",
            }
            params = {
                "query": query,
            }
            if extra_params:
                assert "query" not in extra_params
                params.update(extra_params)
            response = await client.get(
                self.rdfox_url + f"/datastores/{self.rdfox_dstore}/sparql",
                params=params,
                headers=headers,
                timeout=60,
            )
        return response

    async def _update(self, update: str) -> httpx.Response:
        async with httpx.AsyncClient(verify=self.rdfox_cert_verify) as client:
            headers = {
                "Accept": "application/sparql-results+xml",
                "Content-Type": "application/x-www-form-urlencoded",
            }
            params = {
                "update": update,
            }
            response = await client.post(
                self.rdfox_url + f"/datastores/{self.rdfox_dstore}/sparql",
                params=params,
                headers=headers,
            )
        return response

    async def _upsert(
        self, content: str, headers: dict[str, str] | None = None
    ) -> httpx.Response:
        async with httpx.AsyncClient(verify=self.rdfox_cert_verify) as client:
            if headers is None:
                headers = {"Content-Type": "text/turtle"}
            params = {"operation": "add-content-update-prefixes"}
            if self.rdfox_graph:
                params["default-graph"] = PREFIXES["graph"][self.rdfox_graph].n3()
            response = await client.patch(
                self.rdfox_url + f"/datastores/{self.rdfox_dstore}/content",
                content=content,
                params=params,
                headers=headers,
            )
        return response

    def parse_result(
        self,
        response: httpx.Response,
        nvars: int | None = None,
        nbindings: int | None = None,
    ) -> Result:
        if response.status_code == 400:
            raise RDFoxCSHandleBadRequestError(
                f"RDFoxCSHandleBadRequestError: Query failed: HTTP status {response.status_code}, "
                f"Response:\n{response.text}"
            )
        if not response.is_success:
            raise ConnectionError(
                f"RDFoxCSHandle: Query failed: HTTP status {response.status_code}, "
                f"Response:\n{response.text}"
            )
        result = xmlresults.XMLResultParser().parse(source=StringIO(response.text))
        improper_response = (
            result is None or result.bindings is None or result.vars is None
        )
        assert result.bindings is not None and result.vars is not None
        empty_response = (
            improper_response
            or nbindings is not None
            and len(result.bindings) != nbindings
        )
        improper_query = (
            improper_response or nvars is not None and len(result.vars) != nvars
        )
        if improper_query:
            raise RDFoxCSHandleQueryError(
                "Unexpected response. Reason: improper query"
                f"Response.text: {response.text}"
                f"Response.status_code: {response.status_code}"
            )
        if improper_response or empty_response:
            raise ValueError(
                "RDFoxCSHandle: Unexpected response, "
                f"reason: improper response: {improper_response}, "
                f"empty response: {empty_response} "
            )
        return result

    async def upsert(self, triples: list[str]):
        # First ensure prefixes are updated
        prefix_triples = [
            f"@prefix {prefix}: <{str(ns)}> ." for prefix, ns in PREFIXES.items()
        ]
        response = await self._upsert("\n".join(prefix_triples + triples))
        if not response.is_success:
            L.info("RDFoxCSHandle: Upsert response %s", response.text)
            raise ConnectionError(
                f"RDFoxCSHandle: Query failed: HTTP status {response.status_code}"
            )

    async def query_to_str_dict(
        self, query: str, extra_params: dict[str, str] | None = None
    ) -> dict[str, str]:
        response = await self.query(query, extra_params=extra_params)
        result = self.parse_result(response, nvars=2)
        assert result.vars is not None and result.bindings is not None
        retdict: dict[str, str] = {}
        for row in result.bindings:
            raw_key = row[result.vars[0]]
            raw_val = row[result.vars[1]]
            if not (
                isinstance(raw_key, (URIRef, RDFLiteral))
                and isinstance(raw_val, (URIRef, RDFLiteral))
            ):
                raise ValueError("RDFoxCSHandle: Wrong variable types in response")
            retdict[raw_key.toPython()] = raw_val.toPython()
        return retdict

    async def query_to_stripped_id_list(
        self, query: str, extra_params: dict[str, str] | None = None
    ) -> list[str]:
        response = await self.query(query, extra_params=extra_params)
        result = self.parse_result(response, nvars=1)
        assert result.vars is not None and result.bindings is not None
        refs = [row[result.vars[0]] for row in result.bindings]
        assert all(isinstance(r, URIRef) for r in refs)
        return [r.fragment if isinstance(r, URIRef) else "" for r in refs]

    async def query_to_triples(  # pylint: disable=too-many-positional-arguments
        self,
        query: str,
        allow_literal_tail: bool = True,
        allow_uri_tail: bool = True,
        strip_uri_tail: bool = False,
        strip_predicate: bool = True,
        strip_head: bool = False,
        literal_for_predicate: bool = False,
        extra_params: dict[str, str] | None = None,
    ) -> Iterable[tuple[str, str, Any]]:
        # pylint: disable=too-many-branches
        if not (allow_literal_tail or allow_uri_tail):
            raise ValueError("RDFoxCSHandle: Impossible combination of tail options")
        response = await self.query(query, extra_params=extra_params)
        result = self.parse_result(response, nvars=3)
        assert result.vars is not None and result.bindings is not None
        triples = []
        for row in result.bindings:
            head = row[result.vars[0]]
            pred = row[result.vars[1]]
            tail = row[result.vars[2]]
            if not isinstance(head, URIRef):
                raise ValueError("RDFoxCSHandle: Wrong variable types in response")
            tailstr: Any
            if isinstance(tail, URIRef):
                if not allow_uri_tail:
                    raise ValueError("RDFoxCSHandle: Wrong variable types in response")
                tailstr = tail.fragment if strip_uri_tail else tail.toPython()
            elif isinstance(tail, RDFLiteral):
                if not allow_literal_tail:
                    raise ValueError("RDFoxCSHandle: Wrong variable types in response")
                tailstr = tail.toPython()
            else:
                raise ValueError("RDFoxCSHandle: Wrong variable types in response")
            predstr: Any
            if isinstance(pred, URIRef):
                if literal_for_predicate:
                    raise ValueError("RDFoxCSHandle: Wrong variable types in response")
                predstr = pred.fragment if strip_predicate else pred.toPython()
            elif isinstance(pred, RDFLiteral):
                if not literal_for_predicate:
                    raise ValueError("RDFoxCSHandle: Wrong variable types in response")
                predstr = pred.toPython()
            else:
                raise ValueError("RDFoxCSHandle: Wrong variable types in response")

            triples.append(
                (
                    head.fragment if strip_head else head.toPython(),
                    predstr,
                    tailstr,
                )
            )
        return triples

    async def query_to_raw_result(
        self, query: str, extra_params: dict[str, str] | None = None
    ) -> Result:
        response = await self.query(query, extra_params=extra_params)
        try:
            result = self.parse_result(response)
        except RDFoxCSHandleError as e:
            L.error(
                "RDFoxCSHandleError exception caught during parsing result of following query: %s",
                query,
            )
            raise e
        assert result.vars is not None and result.bindings is not None
        return result

    async def push_datalog(self, ruleset: str):
        headers = {"Content-Type": "application/x.datalog"}
        response = await self._upsert(ruleset, headers=headers)
        if not response.is_success:
            L.error("RDFoxCSHandle: Push_datalog response %s", response.text)
            raise ConnectionError(
                f"RDFoxCSHandle: Query failed: HTTP status {response.status_code}"
            )
