"""Write subsets of the complete AI-TOD-v2 (all five sources, eight classes).

``/mnt/datasets/sat/aitod_v2_full`` holds the released v2 annotations with images
from all five sources (see its README). Writes into ``<root>/subsets``:

  all8_train_all.json  the whole train split (11,214 images)
  all8_probe2k.json    2000 train images for quick probing
  all8_val600.json     600 val images for in-training evaluation

The sources differ in size and class mix (Airbus is almost all ships, VisDrone
alone has person, DIOR alone has wind-mill), and airplane, bridge, swimming-pool
and wind-mill have under 800 train instances each. A uniform sample would leave
wind-mill with a few dozen instances. The probe and val subsets therefore take a
fixed quota per source and fill it first with every image holding one of those
rare classes, then at random (seeded).

    uv run python tools/aitod_full_subsets.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

RARE_CLASSES = {"airplane", "bridge", "swimming-pool", "wind-mill"}
PROBE_QUOTAS = {"xview": 400, "airbus": 400, "visdrone": 300, "dota": 450, "dior": 450}
VAL_QUOTAS = {"xview": 120, "airbus": 120, "visdrone": 100, "dota": 130, "dior": 130}


def source_of(file_name: str) -> str:
    """The AI-TOD source dataset, from its crop naming scheme."""
    if re.match(r"^\d+__\d+_\d+\.png$", file_name):
        return "xview"
    if re.match(r"^P\d{4}", file_name):
        return "dota"
    if re.match(r"^[0-9a-f]{9}\.png$", file_name):
        return "airbus"
    if re.match(r"^\d{5}\.png$", file_name):
        return "dior"
    return "visdrone"


def stratified(data: dict, quotas: dict[str, int], seed: int) -> set[int]:
    names = {category["id"]: category["name"] for category in data["categories"]}
    classes_in = defaultdict(set)
    for annotation in data["annotations"]:
        classes_in[annotation["image_id"]].add(names[annotation["category_id"]])
    by_source = defaultdict(list)
    for image in sorted(data["images"], key=lambda image: image["id"]):
        by_source[source_of(image["file_name"])].append(image["id"])
    rng = random.Random(seed)
    chosen = set()
    for source, quota in quotas.items():
        ids = by_source[source]
        rare = [i for i in ids if classes_in[i] & RARE_CLASSES]
        rest = [i for i in ids if not classes_in[i] & RARE_CLASSES]
        rng.shuffle(rare)
        rng.shuffle(rest)
        chosen.update((rare + rest)[:quota])
    return chosen


def subset(data: dict, image_ids: set[int] | None, description: str) -> dict:
    images = [image for image in data["images"] if image_ids is None or image["id"] in image_ids]
    kept = {image["id"] for image in images}
    return {
        "info": {**(data.get("info") or {}), "subset": description},
        "licenses": data.get("licenses", []),
        "images": images,
        "annotations": [a for a in data["annotations"] if a["image_id"] in kept],
        "categories": data["categories"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("/mnt/datasets/sat/aitod_v2_full"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train = json.loads((args.root / "train/annotations.json").read_text())
    val = json.loads((args.root / "val/annotations.json").read_text())
    quota_text = lambda quotas: ", ".join(f"{s} {n}" for s, n in quotas.items())
    outputs = {
        "all8_train_all": subset(train, None, "All train-split images, all sources"),
        "all8_probe2k": subset(
            train, stratified(train, PROBE_QUOTAS, args.seed),
            f"Train probe, rare-class images first per source quota ({quota_text(PROBE_QUOTAS)}), seed {args.seed}",
        ),
        "all8_val600": subset(
            val, stratified(val, VAL_QUOTAS, args.seed),
            f"Val subset, rare-class images first per source quota ({quota_text(VAL_QUOTAS)}), seed {args.seed}",
        ),
    }
    (args.root / "subsets").mkdir(exist_ok=True)
    for name, data in outputs.items():
        path = args.root / "subsets" / f"{name}.json"
        path.write_text(json.dumps(data))
        names = {c["id"]: c["name"] for c in data["categories"]}
        per_class = Counter(names[a["category_id"]] for a in data["annotations"])
        sources = Counter(source_of(image["file_name"]) for image in data["images"])
        print(f"{path}: {len(data['images'])} images ({dict(sources)})")
        print("    " + ", ".join(f"{c['name']} {per_class[c['name']]}" for c in data["categories"]))


if __name__ == "__main__":
    main()
