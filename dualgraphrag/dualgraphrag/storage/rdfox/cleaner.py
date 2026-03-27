from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

CacheNamespace = Literal["tkg", "skg"]


@dataclass
class NameCache:
    id_map: dict[str, dict[str, str]] = field(default_factory=lambda: defaultdict(dict))
    ids: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    uniq_counter: int = 0

    def register(self, entry_id: str, name: str, ns: CacheNamespace):
        """Inserts an entry forcefully"""
        self.id_map[ns][name] = entry_id
        self.ids[ns].add(entry_id)


def clean_id_for_rdfox(
    name: str,
    ns: CacheNamespace,
    _ncache: NameCache,
    crop: int = 0,
    cached_only: bool = False,
    keep_colons: bool = False,
) -> str:
    if name in _ncache.id_map[ns]:
        return _ncache.id_map[ns][name]
    if cached_only:
        raise ValueError(f"Reference to unknown node {name} in RDFox cold storage")
    res_list = []
    underscore = False
    for char in name:
        if (keep_colons and char == ":") or (char.isalnum() and char.isascii()):
            res_list.append(char.lower())
            underscore = False
        elif (
            char.isspace()
            or char in "@,.;/\\|!&#*?+=_ -"
            or (not keep_colons and char == ":")
        ):
            if not underscore:
                res_list.append("_")
            underscore = True
    res = "".join(res_list)
    if len(res) > crop > 0:
        res = res[:crop]
    if len(res) > -crop > 0:
        res = res[crop:]
    if len(res) == 0 or res in _ncache.ids[ns]:
        res = f"{res}__{_ncache.uniq_counter}"
        _ncache.uniq_counter += 1
    _ncache.id_map[ns][name] = res
    _ncache.ids[ns].add(res)
    return res
