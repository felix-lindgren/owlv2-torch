"""Write multi-class AI-TOD-v2 (xView part) subsets next to the vehicle-only ones.

The xView part annotates only four of AI-TOD's eight classes (airplane,
storage-tank, ship, vehicle); bridge, swimming-pool, person and wind-mill have
no instances in any split. A query for one of those would only ever be trained
as background on images that do contain such objects, so classes without a
training instance are dropped by default.

Writes into ``<root>/subsets``:
  all4_train_all.json  every train image, the kept classes
  all4_val200.json     the images of vehicle_val200.json, the kept classes, so
                       vehicle AP stays comparable to the vehicle-only runs
  all4_val.json        the whole val split (935 images), for the rare classes:
                       val200 holds only 7 airplanes, 20 ships and 31 tanks

    uv run python tools/aitod_class_subsets.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def subset(data: dict, image_ids: set[int] | None, category_ids: set[int], description: str) -> dict:
    images = [image for image in data["images"] if image_ids is None or image["id"] in image_ids]
    kept_ids = {image["id"] for image in images}
    return {
        "info": {**data.get("info", {}), "subset": description},
        "licenses": data.get("licenses", []),
        "images": images,
        "annotations": [
            annotation
            for annotation in data["annotations"]
            if annotation["image_id"] in kept_ids and annotation["category_id"] in category_ids
        ],
        "categories": [category for category in data["categories"] if category["id"] in category_ids],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("/mnt/datasets/sat/xview_aitod_v2"))
    parser.add_argument("--prefix", default="all4")
    args = parser.parse_args()

    train = json.loads((args.root / "train/annotations.json").read_text())
    val = json.loads((args.root / "val/annotations.json").read_text())
    counts = Counter(annotation["category_id"] for annotation in train["annotations"])
    category_ids = {category["id"] for category in train["categories"] if counts[category["id"]]}
    names = ", ".join(c["name"] for c in train["categories"] if c["id"] in category_ids)
    val200_ids = {
        image["id"]
        for image in json.loads((args.root / "subsets/vehicle_val200.json").read_text())["images"]
    }

    outputs = {
        "train_all": subset(train, None, category_ids, f"All train-split images; classes: {names}"),
        "val200": subset(val, val200_ids, category_ids, f"vehicle_val200 images; classes: {names}"),
        "val": subset(val, None, category_ids, f"All val-split images; classes: {names}"),
    }
    for name, data in outputs.items():
        path = args.root / "subsets" / f"{args.prefix}_{name}.json"
        path.write_text(json.dumps(data))
        per_class = Counter(annotation["category_id"] for annotation in data["annotations"])
        summary = ", ".join(f"{c['name']} {per_class[c['id']]}" for c in data["categories"])
        print(f"{path}: {len(data['images'])} images; {summary}")


if __name__ == "__main__":
    main()
