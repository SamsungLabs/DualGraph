import re
from typing import Any

from rdflib import Literal as RDFLiteral

from ...base import Prescience
from ...utils import get_unique_id
from .cleaner import NameCache, clean_id_for_rdfox

MAX_PROPERTY_ENTITY_NAME_LEN = 100
DEFAULT_SECTION = "Specifications"
DENORMALIZATION_DATALOG_RULES = [
    # Add a supertype to mark all nodes that aren't only in TKG
    "[?s, a, skgt:SKG_Entity] :- [?s, a, ?c], FILTER(?c != skgt:UTKG_Entity) .",
    #
    # # skg:hasFeature maps direct link to features, skipping skg:Spec
    # "[?p, skg:hasFeature, ?f] :- "
    # "[?p, skg:hasSpec, ?s], "
    # "[?s, skg:hasValue, ?f] .",
    #
    # # skg:hasFeature maps to Entry in table if entry has value "yes"
    "[?p, skg:hasFeature, ?f], [?f, rdf:type, skgt:Feature] :- "
    "[?p, skg:hasSpec, ?s], "
    "[?s, skg:inEntry, ?f], "
    "[?s, skg:hasValue, skg:yes] .",
    #
    # # skg:hasFeature maps to skg:8k_recording_support (created if not exists)
    # if some feature ID in skg:video_recording_resolution contains "8k"
    "[?p, skg:hasFeature, skg:8k_recording_support], "
    "[skg:8k_recording_support, rdf:type, skgt:Feature], "
    '[skg:8k_recording_support, skg:hasName, "8K Recording Support"] :- '
    "[?p, skg:hasSpec, ?s], "
    "[?s, skg:inEntry, skg:video_recording_resolution], "
    "[?s, skg:hasValue, ?f], "
    'FILTER(REGEX(str(?f), "8k")) .',
    #
    # # skg:hasFeature maps to skg:5g_support (created if not exists)
    # if at least one of the specific 5G features is present
    "[?p, skg:hasFeature, skg:5g_support], [skg:5g_support, rdf:type, skgt:Feature], "
    '[skg:5g_support, skg:hasName, "5G Support"] :- '
    "[?p, skg:hasSpec, ?s], [?s, skg:hasValue, ?f5g], "
    "FILTER (?f5g IN (skg:5g_sub6_fdd, skg:5g_sub6_tdd, skg:5g_sub6_sdl ) ) .",
    #
    # # skg:hasFeature maps to skg:4g_support (created if not exists)
    # if at least one of the specific 4G features is present
    "[?p, skg:hasFeature, skg:4g_support], [skg:4g_support, rdf:type, skgt:Feature], "
    '[skg:4g_support, skg:hasName, "4G Support"] :- '
    "[?p, skg:hasSpec, ?s], [?s, skg:hasValue, ?f4g], "
    "FILTER (?f4g IN (skg:4g_lte_fdd, skg:4g_lte_tdd ) ) .",
    #
    # # Price is a special property and should be easily found
    "[?product, skg:hasPrice, ?price] :- "
    "[?product, skg:hasSpec, ?spec], [?spec, skg:inEntry, ?entry], "
    '[?entry, skg:hasName, "Price"], [?spec, skg:hasValue, ?price] .',
    #
    # # Products also belong in categories
    "[?p, skg:belongs, ?c] :- [?p, skg:variantOf, ?pr], [?pr, skg:belongs, ?c] .",
]

KNOWN_POSTFIX_UNITS = {
    "%": "%",
    "mm": "mm",
    "V": "V",
    "W": "W",
    "w": "W",
    "Hz": "Hz",
    "kHz": "kHz",
    "℃": "°C",
    "°C": "°C",
    "Celsius": "°C",
    "kW": "kW",
    "Kw": "kW",
    "hr": "h",
    "Hours": "h",
    "h": "h",
    "hrs": "h",
    "A": "A",
    "DPI": "DPI",
    "gb": "GB",
    "GB": "GB",
    "Gbps": "Gbps",
    "Khz": "kHz",
    "l": "l",
    "L": "l",
    "Lumen": "lumen",
    "lumen": "lumen",
    "m": "m",
    "M": "m",
    "MB": "MB",
    "MB/s": "MB/s",
    "MHz": "MHz",
    "Mhz": "MHz",
    "MP": "MP",
    "Months": "month",
    "month": "month",
    "P/S": "P/S",
    "Pa": "Pa",
    "TB": "TB",
    "Wh": "Wh",
    "cd/m2": "cd/m2",
    "cd/㎡": "cd/m2",
    "cm": "cm",
    "g": "g",
    "grams": "g",
    "inch": "inch",
    '"': "inch",
    "kWh": "kWh",
    "kWh/100 cycles": "kWh/100 cycles",
    "kWh/year": "kWh/year",
    "kg": "kg",
    "kg/24hr": "kg/24h",
    "kg/24h": "kg/24h",
    "kwh/cycle": "kWh/cycle",
    "kWh/cycle": "kWh/cycle",
    "mA": "mA",
    "mAh": "mAh",
    "mGy": "mGy",
    "mGy/210sec (ISO/IEC7816-1)": "mGy",
    "min": "min",
    "ms": "ms",
    "nit": "nit",
    "ohm": "ohm",
    "pixels": "pixels",
    "rpm": "rpm",
    "sec+": "s",
    "s": "s",
    "secs": "s",
    "tb": "TB",
    "uA": "uA",
    "ℓ": "l",
    "Ω": "ohm",
    "㎜": "mm",
    "㎥/h": "m3/h",
}

KNOWN_POSTFIX_UNITS_RE = {
    r"^%\s*(?!~)\s.*": "%",
    r"^GB\s+": "GB",
    r"^Hz\s*\(": "Hz",
    r"^IOPS(\s|\*)": "IOPS",
    r"^L\s*(.*Cu\.ft)": "l",
    r"^MB/s(ec|\s)": "MB/s",
    r"^TB\s+": "TB",
    r"^V(\s[\/-:±]|:\s?[0-9]|\/)": "V",
    r"^W(\s|\()": "W",
    r"^Wh \(": "Wh,",
    r"^dB": "dB",
    r"^fps": "fps",
    r"^g\s\(": "g",
    r"^hours?\s": "h",
    r"^hr\s\(": "h",
    r"^kg\s(\(|\*)": "kg",
    r"^m\s+": "m",
    r"^mm(\s+|\(|\~)": "mm",
    r"^ms\s?\(": "ms",
    r"^year": "year",
    r"^\"(\s|\+|-)": "inch",
    r"^GBP": "GBP",
}


def _rawspecs_get_value_triples(
    feat_content: str, value_id: str, entry_id: str
) -> list[str]:
    content = feat_content.strip()
    # Handle unitless
    # OPTION: handle units from spec name?
    if re.match(r"^(-?[0-9]*\.?[0-9]+)$", content):
        value = float(content) if "." in content else int(content)
        return [f"skg:{value_id} skg:hasNumericValue {RDFLiteral(value).n3()} ."]
    # Handle dimensions.
    # Get anything that starts with dimensions followed by any text.
    # We don't analyze units, because they're always mm in dataset...
    # Unless it's unitless. In most cases that's resolutions, so we add
    # a simple filter for that case.
    mat = re.match(
        r"^([0-9]*\.?[0-9]+)\s*(?:\(L\)\s*)?[xX×∗\*]\s*"  # Line too long, eh...
        r"([0-9]*\.?[0-9]+)\s*(?:(?:\([WV]\))\s*)?(?:[xX×∗\*]\s*([0-9]*\.?[0-9]+))?",
        content,
    )
    if mat:
        dim1 = float(mat.group(1)) if "." in mat.group(1) else int(mat.group(1))
        dim2 = float(mat.group(2)) if "." in mat.group(2) else int(mat.group(2))
        ret = [
            f"skg:{value_id} skg:hasDim1 {RDFLiteral(dim1).n3()} .",
            f"skg:{value_id} skg:hasDim2 {RDFLiteral(dim2).n3()} .",
        ]
        if "esolution" not in entry_id:
            ret.append(f'skg:{value_id} skg:hasUnit "mm" .')
        if mat.group(3):
            dim3 = float(mat.group(3)) if "." in mat.group(3) else int(mat.group(3))
            ret.append(f"skg:{value_id} skg:hasDim3 {RDFLiteral(dim3).n3()} .")
        return ret
    # Handle simple postfix units
    # No prefix handling for now - no cases in dataset
    # Special handling for some units often followed by unrelated text,
    # but first grab the simple ones (faster)
    mat = re.match(r"^([0-9]*\.?[0-9]+)\s*(.+)$", content)
    if mat:
        maybe_unit = mat.group(2)
        unit = None
        if maybe_unit in KNOWN_POSTFIX_UNITS:
            unit = KNOWN_POSTFIX_UNITS[maybe_unit]
        else:
            for regexp, reunit in KNOWN_POSTFIX_UNITS_RE.items():
                if re.match(regexp, maybe_unit):
                    unit = reunit
                    break  # First match, we have no smarter tiebreaking rules
        if unit is not None:
            value = float(mat.group(1)) if "." in mat.group(1) else int(mat.group(1))
            return [
                f"skg:{value_id} skg:hasNumericValue {RDFLiteral(value).n3()} .",
                f"skg:{value_id} skg:hasUnit {RDFLiteral(unit).n3()} .",
            ]
    return []


def smart_split(value: str, split_on: str = ",") -> list[str]:
    parts = []
    current = []
    stack = []
    for char in value:
        if char in "([":
            stack.append(char)
            current.append(char)
        elif char == ")":
            if stack and stack[-1] == "(":
                stack.pop()
            current.append(char)
        elif char == "]":
            if stack and stack[-1] == "[":
                stack.pop()
            current.append(char)
        elif char in split_on:
            if not stack:  # Only split when outside parentheses
                parts.append("".join(current))
                current = []
            else:
                current.append(char)
        else:
            current.append(char)
    parts.append("".join(current))  # Add last part
    return parts


def _rawspecs_feature_parse(
    feature: str, value: str, prod_id: str, _ncache: NameCache
) -> tuple[list[str], list[str]]:
    entities: set[str] = set()
    triples: list[str] = []
    feature_place = feature.split(":", maxsplit=1)
    assert len(feature_place) in (1, 2)
    if len(feature_place) == 1:
        feature_place.insert(0, DEFAULT_SECTION)
    section_id = clean_id_for_rdfox(feature_place[0], "skg", _ncache)
    entry_id = clean_id_for_rdfox(feature_place[1], "skg", _ncache)
    entities.update({f"skg:{section_id}", f"skg:{entry_id}"})
    triples.extend(
        [
            f"skg:{section_id} a skgt:Section .",
            f"skg:{entry_id} a skgt:Entry .",
            f"skg:{section_id} skg:hasName {RDFLiteral(feature_place[0]).n3()} .",
            f"skg:{entry_id} skg:hasName {RDFLiteral(feature_place[1]).n3()} .",
        ]
    )
    # First remove thousands separators. This might break on lists of three-digit
    # integers if entered without spaces, but there's no way to handle that context-free
    value = re.sub(r"(\d),(?=\d{3})", r"\1", value)
    parts = smart_split(value, split_on=",")
    for value_content in parts:
        value_content = value_content.strip()
        if value_content:
            value_id = clean_id_for_rdfox(
                value_content,
                "skg",
                _ncache,
                crop=MAX_PROPERTY_ENTITY_NAME_LEN,
            )
            spec_id = get_unique_id(prefix="spec")
            entities.update({f"skg:{spec_id}", f"skg:{prod_id}", f"skg:{value_id}"})
            triples.extend(
                [
                    f"skg:{spec_id} a skgt:Spec .",
                    f"skg:{prod_id} skg:hasSpec skg:{spec_id} .",
                    f"skg:{spec_id} skg:inSection skg:{section_id} .",
                    f"skg:{spec_id} skg:inEntry skg:{entry_id} .",
                    f"skg:{spec_id} skg:hasValue skg:{value_id} .",
                    f"skg:{value_id} a skgt:Value .",
                    f"skg:{value_id} skg:hasName {RDFLiteral(value_content).n3()} .",
                ]
            )
            triples.extend(
                _rawspecs_get_value_triples(value_content, value_id, entry_id)
            )
    return triples, list(entities)


def _rawspecs_converter(
    knowledge: Any, _ncache: NameCache
) -> tuple[list[str], list[str], list[str]]:
    assert isinstance(knowledge, list)
    entities: set[str] = set()
    triples: set[str] = set()
    for data in knowledge:
        assert isinstance(data, dict)
        prod_range_name = data["prod_range"]
        assert isinstance(prod_range_name, str)
        prod_range_id = clean_id_for_rdfox(
            prod_range_name.replace("+", "plus"), "skg", _ncache
        )
        # It's better if range gets an id first.
        assert "name" in data and "prod_range" in data
        assert isinstance(data["name"], str)
        prod_name_parts = data["name"].split("<SEP>")
        prod_id = clean_id_for_rdfox(
            "__".join(prod_name_parts).replace("+", "plus"), "skg", _ncache
        )
        entities.add(f"skg:{prod_id}")
        triples.update(
            [
                f"skg:{prod_id} a skgt:Product .",
                f"skg:{prod_id} skg:hasName {RDFLiteral(prod_name_parts[0]).n3()} .",
            ]
        )
        entities.add(f"skg:{prod_range_id}")
        triples.update(
            [
                f"skg:{prod_range_id} a skgt:ProductRange .",
                f"skg:{prod_range_id} skg:hasName {RDFLiteral(prod_range_name).n3()} .",
                f"skg:{prod_id} skg:variantOf skg:{prod_range_id} .",
            ]
        )
        if "categories" in data:
            assert isinstance(data["categories"], list)
            for cat_name in data["categories"]:
                assert isinstance(cat_name, str)
                cat_id = clean_id_for_rdfox(cat_name, "skg", _ncache)
                entities.add(f"skg:{cat_id}")
                triples.update(
                    [
                        f"skg:{cat_id} a skgt:Category .",
                        f"skg:{cat_id} skg:hasName {RDFLiteral(cat_name).n3()} .",
                        f"skg:{prod_range_id} skg:belongs skg:{cat_id} .",
                    ]
                )
        if "specs" in data:
            for feature, value in data["specs"].items():
                triples_, entities_ = _rawspecs_feature_parse(
                    feature, value, prod_id, _ncache
                )
                triples.update(triples_)
                entities.update(entities_)
    return list(triples), DENORMALIZATION_DATALOG_RULES, list(entities)


converters = {
    "rawspecs": _rawspecs_converter,
}


def knowledge_to_triples(
    prescience: Prescience, _ncache: NameCache
) -> tuple[list[str], list[str], list[str]]:
    return converters[prescience.format](prescience.data, _ncache)
