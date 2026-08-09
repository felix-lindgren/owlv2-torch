"""Compare post-preprocessing object sizes for DIOR and tiled xView arms.

The comparison is restricted to the four frozen direct-overlap concepts and
uses the boxes after each dataset's actual square-pad/resize transform. Multiple
xView annotation files can be named with ``LABEL=PATH``. The closest arm is
selected by absolute log-median distance across concepts, not nominal GSD.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


DIRECT_CLASSES = ("Airplane", "Ship", "Storage tank", "Vehicle")


def clean_name(name: str) -> str:
    return " ".join(name.lower().replace("-", " ").replace("_", " ").split())


def summarize(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "p10": math.nan, "median": math.nan, "p90": math.nan, "below_16": math.nan}
    def percentile(q: float) -> float:
        return ordered[round(q * (len(ordered) - 1))]
    return {
        "count": len(ordered),
        "p10": percentile(0.10),
        "median": statistics.median(ordered),
        "p90": percentile(0.90),
        "below_16": sum(value < 16.0 for value in ordered) / len(ordered),
    }


def load_coco_sizes(path: Path, output_size: int, *, xview: bool) -> dict[str, list[float]]:
    with path.open(encoding="utf-8") as handle:
        coco = json.load(handle)
    if xview:
        conversion = coco.get("info", {}).get("xview_conversion", {})
        converted_size = conversion.get("output_size")
        if converted_size is not None and int(converted_size) != output_size:
            raise ValueError(
                f"{path} boxes are at output_size={converted_size}, requested {output_size}"
            )
    categories = {int(category["id"]): str(category["name"]) for category in coco["categories"]}
    images = {int(image["id"]): image for image in coco["images"]}
    direct_by_key = {clean_name(name): name for name in DIRECT_CLASSES}
    values: dict[str, list[float]] = defaultdict(list)
    for annotation in coco["annotations"]:
        if xview:
            concept = annotation.get("mapped_superclass")
        else:
            concept = direct_by_key.get(clean_name(categories[int(annotation["category_id"])]))
        if concept not in DIRECT_CLASSES:
            continue
        width, height = map(float, annotation["bbox"][2:4])
        if width <= 0 or height <= 0:
            continue
        if not xview:
            image = images[int(annotation["image_id"])]
            scale = output_size / max(float(image["width"]), float(image["height"]))
            width, height = width * scale, height * scale
        values[concept].append(math.sqrt(width * height))
    return values


def load_hf_dior_sizes(split: str, output_size: int) -> dict[str, list[float]]:
    from datasets import load_dataset

    dataset = load_dataset("HichTala/dior", split=split)
    class_names = dataset.features["objects"]["category"].feature.names
    direct_by_key = {clean_name(name): name for name in DIRECT_CLASSES}
    values: dict[str, list[float]] = defaultdict(list)
    for sample in dataset:
        scale = output_size / max(float(sample["width"]), float(sample["height"]))
        for bbox, category in zip(sample["objects"]["bbox"], sample["objects"]["category"]):
            concept = direct_by_key.get(clean_name(class_names[int(category)]))
            if concept is None:
                continue
            width, height = float(bbox[2]) * scale, float(bbox[3]) * scale
            if width > 0 and height > 0:
                values[concept].append(math.sqrt(width * height))
    return values


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("Expected LABEL=ANNOTATIONS.json")
    return name, Path(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xview", action="append", type=parse_named_path, required=True, metavar="LABEL=JSON")
    parser.add_argument("--dior-annotations", type=Path, help="Local DIOR COCO JSON; otherwise load HichTala/dior.")
    parser.add_argument("--dior-split", default="test")
    parser.add_argument("--output-size", type=int, default=960)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    if args.output_size <= 0:
        raise ValueError("output-size must be positive")

    dior = (
        load_coco_sizes(args.dior_annotations, args.output_size, xview=False)
        if args.dior_annotations
        else load_hf_dior_sizes(args.dior_split, args.output_size)
    )
    arms = {label: load_coco_sizes(path, args.output_size, xview=True) for label, path in args.xview}
    missing = {
        dataset_name: [name for name in DIRECT_CLASSES if not values[name]]
        for dataset_name, values in [("DIOR", dior), *arms.items()]
        if any(not values[name] for name in DIRECT_CLASSES)
    }
    if missing:
        raise ValueError(f"Every scale arm must contain all four direct classes: {missing}")
    result = {
        "measurement": "sqrt(box_area)_pixels_after_square_pad_and_resize",
        "output_size": args.output_size,
        "dior": {name: summarize(dior[name]) for name in DIRECT_CLASSES},
        "xview": {
            label: {name: summarize(values[name]) for name in DIRECT_CLASSES}
            for label, values in arms.items()
        },
    }
    distances = {}
    for label, values in arms.items():
        terms = []
        for name in DIRECT_CLASSES:
            target_median = result["dior"][name]["median"]
            source_median = result["xview"][label][name]["median"]
            if math.isfinite(target_median) and math.isfinite(source_median) and target_median > 0 and source_median > 0:
                terms.append(abs(math.log(source_median / target_median)))
        distances[label] = statistics.mean(terms)
    winner = min(distances, key=distances.get)
    result["selection"] = {
        "criterion": "mean absolute log ratio of four class medians",
        "distance_by_arm": distances,
        "baseline_arm": winner,
    }

    print(f"{'dataset':<14} {'class':<14} {'count':>9} {'p10':>8} {'median':>8} {'p90':>8} {'<16px':>8}")
    for dataset_name, summaries in [("DIOR", result["dior"]), *result["xview"].items()]:
        for name in DIRECT_CLASSES:
            row = summaries[name]
            print(f"{dataset_name:<14} {name:<14} {row['count']:>9} {row['p10']:>8.2f} {row['median']:>8.2f} {row['p90']:>8.2f} {row['below_16']:>8.1%}")
    print(f"Baseline scale: {winner} ({result['selection']['criterion']})")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
