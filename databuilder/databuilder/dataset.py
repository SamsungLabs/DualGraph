import argparse
import json
import os
import re
from dataclasses import dataclass
from os import makedirs
from pathlib import Path
from shutil import copyfile

from bs4 import BeautifulSoup

CHUNK_CHAR_LIMIT = 5000
CHUNK_MARKER = "[CHUNK_BREAK]"
FEATURE_MARKER = '# Feature from "{title}" page'
FEATURE_NEXT_MARKER = '# Feature from "{title}" page'
IMAGE_MARKER = "## Image description"
FAQ_MARKER = '## FAQ entry from "{title}" page'
SCRAPE_DIR_NAME = "scraped_htmls"
SPECS_FILE_NAME = "specs.json"
SPECS_MARKER = '# Specifications from "{title}" page'
SPECS_SECTION_MARKER = '## Specifications section "{section}"'


@dataclass
class ExtractionStats:
    extracted_rows: int
    total_rows: int
    extracted_chars: int
    total_chars: int


def generate_md(htmlfilepath: Path, specs: dict | None) -> ExtractionStats:
    # pylint: disable=too-many-locals,too-many-statements
    data = []
    with open(htmlfilepath, "r", encoding="utf-8") as f:
        content = f.read()

    soup = BeautifulSoup(content)
    all_lines = list(soup.stripped_strings)
    count_total_lines = len(all_lines)
    count_total_chars = sum(len(l) for l in all_lines)
    count_extracted_lines = 0
    count_extracted_chars = 0
    title = "no title" if soup.title is None else soup.title.string
    features = soup.find_all(
        "div", class_=lambda x: str(x).startswith(("hubble-feature-", "feature-"))
    )
    filtered_features = [
        feature
        for feature in features
        if not any(parent in features for parent in feature.parents)
    ]
    for feature in filtered_features:
        text = []
        blinds = feature.find_all(class_="blind")
        for blind in blinds:
            blind.decompose()
        strings = list(feature.stripped_strings)
        images = {str(img.attrs["alt"]) for img in feature.find_all("img")}
        fmarker = FEATURE_MARKER.format(title=title)
        textlen = len(fmarker)
        text.append(fmarker)
        for string in strings:
            count_extracted_lines += 1
            count_extracted_chars += len(string)
            if textlen > CHUNK_CHAR_LIMIT:
                text.append(CHUNK_MARKER)
                fnmarker = FEATURE_NEXT_MARKER.format(title=title)
                text.append(fnmarker)
                textlen = len(fnmarker)
            textlen += len(string)
            text.append(string)
        for image in images:
            # This is actually cheating, total does not count alt texts. Sue me.
            count_extracted_lines += 1
            count_extracted_chars += len(image)
            if textlen > CHUNK_CHAR_LIMIT:
                text.append(CHUNK_MARKER)
                fnmarker = FEATURE_NEXT_MARKER.format(title=title)
                text.append(fnmarker)
                textlen = len(fnmarker)
            text.append(IMAGE_MARKER)
            text.append(image)
            textlen += len(IMAGE_MARKER) + len(image)
        data.append("\n".join(text))
    textlen = 0
    if len(data) > 0:
        data.append(CHUNK_MARKER)
    faq_found = False
    for faq in soup.find_all("div", class_=lambda x: str(x).endswith("faq__accordion")):
        faq_found = True
        strings = list(faq.stripped_strings)
        count_extracted_lines += len(strings)
        count_extracted_chars += sum(len(s) for s in strings)
        faqmarker = FAQ_MARKER.format(title=title)
        textlen += len(faqmarker) + sum(len(string) for string in strings)
        data.append(faqmarker)
        data.append("\n".join(strings))
        if textlen > CHUNK_CHAR_LIMIT:
            data.append(CHUNK_MARKER)
            textlen = 0
    if specs:
        if len(data) > 0 and faq_found:
            data.append(CHUNK_MARKER)
        data.append(SPECS_MARKER.format(title=title))
        for section, entries in specs.items():
            count_extracted_lines += len(entries)
            count_extracted_chars += sum(len(s) for s in entries)
            data.append(SPECS_SECTION_MARKER.format(section=section))
            data.extend(entries)
    with open(Path(htmlfilepath).with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write("\n".join(data))
    return ExtractionStats(
        total_rows=count_total_lines,
        total_chars=count_total_chars,
        extracted_rows=count_extracted_lines,
        extracted_chars=count_extracted_chars,
    )


def parse_variant_features(featstrings: list[str]) -> dict:
    feats = {}
    for fstr in featstrings:
        pair = fstr.split(":", maxsplit=1)
        assert len(pair) == 2
        feats[f"Variant features:{pair[0]}"] = pair[1]
    return feats


def specs_reshape(specslist: list[dict]) -> dict:
    specs: dict[str, list[dict]] = {}
    for entry in specslist:
        assert "url_hash" in entry and "specs" in entry
        urlhash = entry["url_hash"]
        specdict = entry["specs"]
        if entry["price"].startswith("Total price:\n"):
            price_maybe = re.search(r"£([\d\.,]+)", entry["price"])
            if price_maybe:
                price = price_maybe.group(1).replace(",", "")
                specdict["Price"] = f"{price} GBP"
        if entry["model_code"]:
            specdict["Model Code"] = entry["model_code"]
        specdict.update(parse_variant_features(entry["variant_features"]))
        if urlhash not in specs:
            specs[urlhash] = []
        specs[urlhash].append(
            {
                "name": entry["name"],
                "categories": entry["categories"],
                "prod_range": entry["prod_range"],
                "specs": specdict,
            }
        )
    return specs


def specs_strings_dict(entries: list[dict]) -> dict[str, set[str]] | None:
    if not entries:
        return None
    ret = {}  # type: ignore
    for entry in entries:
        assert isinstance(entry["specs"], dict)
        for key, value in entry["specs"].items():
            assert isinstance(key, str)
            key_fields = key.split(":", maxsplit=1)
            if len(key_fields) == 2:
                section = key_fields[0]
                rest = key_fields[1]
            else:
                section = "UNNAMED"
                rest = key_fields[0]
            if section not in ret:
                ret[section] = set()
            ret[section].add(f"{rest}: {value}")
        if "UNNAMED" not in ret:
            ret["UNNAMED"] = set()
        ret["UNNAMED"].update(
            [
                f'Product Name: {entry["name"]}',
                f'Product Range: {entry["prod_range"]}',
                f'Product Categories: {", ".join(entry["categories"])}',
            ]
        )
    return ret


def convert_to_dataset(sourcedir: Path, destdir: Path):
    scrape_dirpath = sourcedir / SCRAPE_DIR_NAME
    specs_filepath = sourcedir / SPECS_FILE_NAME
    assert (
        scrape_dirpath.exists() and scrape_dirpath.is_dir() and specs_filepath.exists()
    )
    with open(specs_filepath, "r", encoding="utf-8") as f:
        specs = json.load(f)
        assert isinstance(specs, list)
    files = os.listdir(scrape_dirpath)
    files = [f for f in files if f.endswith(".html")]
    specs = specs_reshape(specs)
    stats: list[ExtractionStats] = []
    errorcount = 0
    for filenum, filename in enumerate(files):
        print(f"Processing file {filenum}/{len(files)}")
        htmlpath = scrape_dirpath / filename
        nameroot = Path(filename).stem
        dirname = destdir / f"file-{nameroot}"
        try:
            makedirs(dirname)
        except FileExistsError:
            # We already indexed this file
            continue
        copyfile(htmlpath, dirname / f"part-{nameroot}.html")
        stats.append(
            generate_md(
                dirname / f"part-{nameroot}.html",
                specs_strings_dict(specs.get(nameroot, None)),
            )
        )
        file_metadata = {
            "id": f"file-{nameroot}",
            "type": "fs:file",
            "content": [
                {
                    "id": f"part-{nameroot}",
                    "path": f"file-{nameroot}/part-{nameroot}.html",
                    "path_textualized": f"file-{nameroot}/part-{nameroot}.md",
                }
            ],
            "name": f"{nameroot}.html",
            "path": f"file-{nameroot}/part-{nameroot}.html",
        }
        if nameroot in specs:
            file_metadata["prescience"] = [
                {
                    "format": "rawspecs",
                    "path": f"file-{nameroot}/inject-{nameroot}.json",
                }
            ]
            with open(dirname / f"inject-{nameroot}.json", "w", encoding="utf-8") as f:
                json.dump(specs[nameroot], f, indent=4)
        with open(dirname / f"file-{nameroot}.json", "w", encoding="utf-8") as f:
            json.dump(file_metadata, f, indent=4)
    totals = ExtractionStats(
        total_rows=sum(s.total_rows for s in stats),
        total_chars=sum(s.total_chars for s in stats),
        extracted_rows=sum(s.extracted_rows for s in stats),
        extracted_chars=sum(s.extracted_chars for s in stats),
    )
    rp = 100.0 * totals.extracted_rows / totals.total_rows
    cp = 100.0 * totals.extracted_chars / totals.total_chars
    print(
        f"{len(stats)} HTMLs parsed, keeping on average "
        f"{rp}% rows or {cp}% characters; {errorcount} errors.\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert raw scrape data to dataset format"
    )

    parser.add_argument("input_path", type=Path, help="Directory to be scanned.")
    parser.add_argument(
        "output_path",
        type=Path,
        help="Directory to store outputs in.",
    )
    parsed_args = parser.parse_args()
    convert_to_dataset(parsed_args.input_path, parsed_args.output_path)
