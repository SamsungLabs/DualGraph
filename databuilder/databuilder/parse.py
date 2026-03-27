import glob
import json
from argparse import ArgumentParser
from pathlib import Path

import pandas
from bs4 import BeautifulSoup
from tqdm import tqdm


def parse_A(soup):
    specs = {}
    specs_div = soup.find("section", {"id": "specs"})
    spec_categories = specs_div.find_all("div", class_="pdd32-product-spec__item")
    for spec_category in spec_categories:
        spec_category_name_el = spec_category.find(
            "h3", class_="pdd32-product-spec__title"
        )
        assert spec_category_name_el is not None
        spec_category_name = spec_category_name_el.text.strip()

        spec_items = spec_category.find_all(
            "li", class_="pdd32-product-spec__content-item"
        )
        for spec_item in spec_items:

            spec_item_name_el = spec_item.find(
                "p", class_="pdd32-product-spec__content-item-title"
            )
            if spec_item_name_el is not None:
                spec_item_name = spec_item_name_el.text.strip()
                key_name = f"{spec_category_name}:{spec_item_name}"
            else:
                assert len(spec_items) == 1
                key_name = spec_category_name

            spec_content = spec_item.find(
                "p", class_="pdd32-product-spec__content-item-desc"
            )
            spec_item_value = spec_content.text.strip()
            specs[key_name] = spec_item_value

    return specs


def parse_B(soup):
    specs = {}
    specs_div = soup.find("div", {"class": "specification__table"})
    spec_categories = specs_div.find_all("div", class_="specification__row")
    for spec_category in spec_categories:
        spec_category_name_el = spec_category.find("h3")
        assert spec_category_name_el is not None
        spec_category_name = spec_category_name_el.text.strip()

        spec_items = spec_category.find_all("li")
        for spec_item in spec_items:

            spec_item_name_el = spec_item.find("div", class_="name")
            if spec_item_name_el is not None:
                spec_item_name = spec_item_name_el.text.strip()
                key_name = f"{spec_category_name}:{spec_item_name}"
            else:
                assert len(spec_items) == 1
                key_name = spec_category_name

            spec_content = spec_item.find("div", class_="detail")
            spec_item_value = spec_content.text.strip()
            specs[key_name] = spec_item_value
    return specs


def parse_C(soup):
    try:
        return parse_A(soup)
    except AttributeError:
        return {}


def parse(input_dir: Path):
    data = []
    json_files = list(glob.glob(str(input_dir / "*.json")))

    for file in tqdm(json_files):
        if file.endswith("specs.json") or file.endswith("metadata.json"):
            continue

        with open(file, "r", encoding="utf-8") as f:
            entry = json.load(f)

        categories = entry["url"].split("/")[4:-2]

        spec_html = entry["html"]
        soup = BeautifulSoup(spec_html)

        processed_entry = {
            "name": entry["variant_name"],
            "variant_features": entry["variant_features"],
            "categories": categories,
            "prod_range": entry["product_range"],
            "price": entry["price"],
            "model_code": entry["model_code"],
            "url_hash": entry["url_hash"],
        }
        match entry["layout_type"]:
            case "A":
                processed_entry["specs"] = parse_A(soup)
            case "B":
                processed_entry["specs"] = parse_B(soup)
            case "C":
                processed_entry["specs"] = parse_C(soup)

        data.append(processed_entry)

    df = pandas.DataFrame(data)
    deduplicated = df.drop_duplicates(["model_code", "name"])
    deduplicated.to_json(input_dir / "specs.json", orient="records", indent=2)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir", type=Path, help="Directory where htmls are saved"
    )
    args = parser.parse_args()
    parse(args.input_dir)


if __name__ == "__main__":
    main()
