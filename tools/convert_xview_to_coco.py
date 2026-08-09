"""Convert xView GeoJSON annotations into deterministic, tiled COCO data.

The public xView validation imagery is unlabelled, so this command first makes
a source-image train/holdout split and only then creates crops. Uniform crops
use centre ownership: every source box belongs to at most one grid crop. An
optional category-centred training stream deliberately resamples rare mapped
concepts and is marked separately in image metadata.

Example::

    uv run python tools/convert_xview_to_coco.py \
      --dataset-root /mnt/datasets/sat/xview \
      --output-dir /mnt/datasets/sat/xview_coco_640 \
      --tile-size 640 --output-size 960 \
      --category-crops-per-source 16 --render-samples 8
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from PIL import Image, ImageDraw
from tqdm import tqdm


XVIEW_CATEGORIES = {
    11: "Fixed-wing Aircraft", 12: "Small Aircraft", 13: "Cargo Plane",
    15: "Helicopter", 17: "Passenger Vehicle", 18: "Small Car", 19: "Bus",
    20: "Pickup Truck", 21: "Utility Truck", 23: "Truck", 24: "Cargo Truck",
    25: "Truck w/Box", 26: "Truck Tractor", 27: "Trailer",
    28: "Truck w/Flatbed", 29: "Truck w/Liquid", 32: "Crane Truck",
    33: "Railway Vehicle", 34: "Passenger Car", 35: "Cargo Car",
    36: "Flat Car", 37: "Tank car", 38: "Locomotive",
    40: "Maritime Vessel", 41: "Motorboat", 42: "Sailboat", 44: "Tugboat",
    45: "Barge", 47: "Fishing Vessel", 49: "Ferry", 50: "Yacht",
    51: "Container Ship", 52: "Oil Tanker", 53: "Engineering Vehicle",
    54: "Tower crane", 55: "Container Crane", 56: "Reach Stacker",
    57: "Straddle Carrier", 59: "Mobile Crane", 60: "Dump Truck",
    61: "Haul Truck", 62: "Scraper/Tractor", 63: "Front loader/Bulldozer",
    64: "Excavator", 65: "Cement Mixer", 66: "Ground Grader", 71: "Hut/Tent",
    72: "Shed", 73: "Building", 74: "Aircraft Hangar", 76: "Damaged Building",
    77: "Facility", 79: "Construction Site", 83: "Vehicle Lot", 84: "Helipad",
    86: "Storage Tank", 89: "Shipping container lot", 91: "Shipping Container",
    93: "Pylon", 94: "Tower",
}
IGNORED_TYPE_IDS = {75, 82}

# Frozen direct-label mapping used for exposure accounting and rare-class crop
# sampling. Context labels (for example cranes near harbors) are intentionally
# not called direct DIOR supervision.
XVIEW_TO_DIOR_DIRECT = {
    **{category_id: "Airplane" for category_id in (11, 12, 13)},
    **{category_id: "Vehicle" for category_id in (17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 32, 53, 60, 61, 62, 63, 64, 65, 66)},
    **{category_id: "Ship" for category_id in (40, 41, 42, 44, 45, 47, 49, 50, 51, 52)},
    86: "Storage tank",
}

SourceBox = tuple[int, tuple[float, float, float, float], int]


@dataclass(frozen=True)
class CropSpec:
    x: int
    y: int
    stream: str
    row: int = -1
    col: int = -1
    selected_superclass: str | None = None


def _find_geojson(dataset_root: Path) -> Path:
    candidates = (dataset_root / "train_labels" / "xView_train.geojson", dataset_root / "xView_train.geojson")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Could not find xView_train.geojson; checked " + ", ".join(map(str, candidates)))


def _contains_tiffs(path: Path) -> bool:
    return path.is_dir() and any(p.is_file() and p.suffix.lower() in {".tif", ".tiff"} for p in path.iterdir())


def _find_train_images(dataset_root: Path) -> Path:
    candidates = (dataset_root / "train_images", dataset_root / "train_images" / "train_images", dataset_root / "images" / "train")
    for candidate in candidates:
        if _contains_tiffs(candidate):
            return candidate
    raise FileNotFoundError("Could not find xView training TIFFs; checked " + ", ".join(map(str, candidates)))


def _image_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError as exc:
        raise ValueError(f"Expected a numeric xView image name, got {path.name}") from exc


def list_images(images_dir: Path) -> list[Path]:
    images = sorted((p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}), key=_image_sort_key)
    if not images:
        raise ValueError(f"No TIFF images found in {images_dir}")
    ids = [int(path.stem) for path in images]
    duplicates = [value for value, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate numeric image IDs: {duplicates[:5]}")
    return images


def split_train_val(images: list[Path], val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")
    if len(images) < 2:
        raise ValueError("At least two labeled images are required")
    count = max(1, min(len(images) - 1, round(len(images) * val_fraction)))
    indices = list(range(len(images)))
    random.Random(seed).shuffle(indices)
    val_indices = set(indices[:count])
    return ([p for i, p in enumerate(images) if i not in val_indices], [p for i, p in enumerate(images) if i in val_indices])


def read_source_boxes(geojson_path: Path) -> tuple[dict[str, list[SourceBox]], Counter[int], int]:
    print(f"Loading {geojson_path}...")
    with geojson_path.open(encoding="utf-8") as handle:
        features = json.load(handle).get("features")
    if not isinstance(features, list):
        raise ValueError(f"GeoJSON has no feature list: {geojson_path}")
    boxes_by_image: dict[str, list[SourceBox]] = defaultdict(list)
    ignored: Counter[int] = Counter()
    malformed = 0
    for source_index, feature in enumerate(tqdm(features, desc="Indexing xView annotations"), start=1):
        try:
            properties = feature["properties"]
            image_name = str(properties["image_id"])
            type_id = int(properties["type_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed GeoJSON feature at index {source_index - 1}") from exc
        if type_id in IGNORED_TYPE_IDS:
            ignored[type_id] += 1
            continue
        if type_id not in XVIEW_CATEGORIES:
            raise ValueError(f"Feature {source_index - 1} has unknown xView type_id={type_id}")
        try:
            values = tuple(float(value) for value in properties["bounds_imcoords"].split(","))
        except (KeyError, AttributeError, TypeError, ValueError):
            malformed += 1
            continue
        if len(values) != 4:
            malformed += 1
            continue
        boxes_by_image[image_name].append((type_id, values, int(properties.get("feature_id", source_index))))
    return dict(boxes_by_image), ignored, malformed


def uniform_crop_specs(width: int, height: int, tile_size: int) -> list[CropSpec]:
    """Non-overlapping grid with explicit right/bottom padding."""
    return [
        CropSpec(x=col * tile_size, y=row * tile_size, stream="uniform", row=row, col=col)
        for row in range(math.ceil(height / tile_size))
        for col in range(math.ceil(width / tile_size))
    ]


def direct_counts(boxes_by_image: dict[str, list[SourceBox]]) -> Counter[str]:
    return Counter(
        superclass
        for boxes in boxes_by_image.values()
        for category_id, (x1, y1, x2, y2), _ in boxes
        if (superclass := XVIEW_TO_DIOR_DIRECT.get(category_id)) is not None
        and x2 > x1 and y2 > y1
    )


def category_crop_specs(
    boxes: list[SourceBox], width: int, height: int, tile_size: int, count: int,
    superclass_counts: Counter[str], generator: random.Random,
) -> list[CropSpec]:
    candidates_by_superclass: dict[str, list[SourceBox]] = defaultdict(list)
    for box in boxes:
        superclass = XVIEW_TO_DIOR_DIRECT.get(box[0])
        if superclass is not None:
            candidates_by_superclass[superclass].append(box)
    if not candidates_by_superclass or count <= 0:
        return []
    superclasses = sorted(candidates_by_superclass)
    # Draw the concept first, then a box inside it. Weighting individual boxes
    # would leave total concept probability proportional to sqrt(frequency).
    superclass_weights = [1.0 / math.sqrt(superclass_counts[name]) for name in superclasses]
    specs: list[CropSpec] = []
    seen: set[tuple[int, int, str]] = set()
    attempts = 0
    while len(specs) < count and attempts < count * 20:
        attempts += 1
        superclass = generator.choices(superclasses, weights=superclass_weights, k=1)[0]
        category_id, (x1, y1, x2, y2), _ = generator.choice(candidates_by_superclass[superclass])
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        # Random placement around the selected centre prevents every centred
        # object appearing at the exact middle while guaranteeing inclusion.
        low_x, high_x = max(0.0, cx - tile_size), min(cx, max(0, width - tile_size))
        low_y, high_y = max(0.0, cy - tile_size), min(cy, max(0, height - tile_size))
        crop_x = int(round(generator.uniform(min(low_x, high_x), max(low_x, high_x))))
        crop_y = int(round(generator.uniform(min(low_y, high_y), max(low_y, high_y))))
        key = (crop_x, crop_y, superclass)
        if key in seen:
            continue
        seen.add(key)
        specs.append(CropSpec(crop_x, crop_y, "category_centered", selected_superclass=superclass))
    return specs


def remap_owned_boxes(
    boxes: list[SourceBox], crop: CropSpec, tile_size: int, output_size: int,
    min_visibility: float,
) -> tuple[list[dict[str, Any]], int]:
    """Assign by centre, clip at the crop, then rescale to the saved image."""
    output: list[dict[str, Any]] = []
    seam_clipped = 0
    scale = output_size / tile_size
    right, bottom = crop.x + tile_size, crop.y + tile_size
    for category_id, source_box, feature_id in boxes:
        x1, y1, x2, y2 = source_box
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        if not (crop.x <= cx < right and crop.y <= cy < bottom):
            continue
        clipped = (max(x1, crop.x), max(y1, crop.y), min(x2, right), min(y2, bottom))
        clipped_w, clipped_h = clipped[2] - clipped[0], clipped[3] - clipped[1]
        if clipped_w <= 0 or clipped_h <= 0:
            continue
        visibility = clipped_w * clipped_h / ((x2 - x1) * (y2 - y1))
        if visibility < min_visibility:
            continue
        if visibility < 1.0 - 1e-9:
            seam_clipped += 1
        local_x, local_y = (clipped[0] - crop.x) * scale, (clipped[1] - crop.y) * scale
        width, height = clipped_w * scale, clipped_h * scale
        output.append({
            "category_id": category_id,
            "bbox": [local_x, local_y, width, height],
            "area": width * height,
            "iscrowd": 0,
            "xview_feature_id": feature_id,
            "source_bbox": list(source_box),
            "visibility": visibility,
            "mapped_superclass": XVIEW_TO_DIOR_DIRECT.get(category_id),
        })
    return output, seam_clipped


def save_crop(source: Image.Image, crop: CropSpec, tile_size: int, output_size: int, destination: Path) -> tuple[int, int]:
    available_w = max(0, min(tile_size, source.width - crop.x))
    available_h = max(0, min(tile_size, source.height - crop.y))
    tile = Image.new("RGB", (tile_size, tile_size))
    if available_w and available_h:
        tile.paste(source.crop((crop.x, crop.y, crop.x + available_w, crop.y + available_h)), (0, 0))
    if output_size != tile_size:
        tile = tile.resize((output_size, output_size), Image.Resampling.BICUBIC)
    tile.save(destination, quality=95)
    return tile_size - available_w, tile_size - available_h


def _keep_empty(policy: str, fraction: float, generator: random.Random) -> bool:
    return policy == "keep" or (policy == "sample" and generator.random() < fraction)


def render_representatives(coco: dict, images_dir: Path, output_dir: Path, count: int) -> None:
    if count <= 0 or not coco["images"]:
        return
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        anns_by_image[int(annotation["image_id"])].append(annotation)
    images = coco["images"]
    candidates = [
        min(images, key=lambda item: len(anns_by_image[item["id"]])),
        max(images, key=lambda item: len(anns_by_image[item["id"]])),
    ]
    for key in ("pad_right", "seam_clipped_boxes"):
        matching = [item for item in images if item.get(key, 0) > 0 or (key == "pad_right" and item.get("pad_bottom", 0) > 0)]
        if matching:
            candidates.append(matching[0])
    remaining = sorted(images, key=lambda item: (-len(anns_by_image[item["id"]]), item["id"]))
    chosen = list({item["id"]: item for item in [*candidates, *remaining]}.values())[:count]
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in chosen:
        image = Image.open(images_dir / item["file_name"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        for ann in anns_by_image[item["id"]]:
            x, y, width, height = ann["bbox"]
            draw.rectangle((x, y, x + width, y + height), outline="red", width=2)
        draw.text((8, 8), f"{item['stream']} boxes={len(anns_by_image[item['id']])}", fill="yellow", stroke_width=2, stroke_fill="black")
        image.save(output_dir / item["file_name"])
    print(f"Rendered {len(chosen)} representative crops to {output_dir}")


def convert_split(
    output_dir: Path, split: str, source_images: Iterable[Path],
    boxes_by_image: dict[str, list[SourceBox]], *, tile_size: int = 960,
    output_size: int = 960, category_crops_per_source: int = 0,
    superclass_counts: Counter[str] | None = None, min_visibility: float = 0.2,
    empty_crop_policy: str = "keep", empty_crop_fraction: float = 0.1,
    seed: int = 0, limit: Optional[int] = None, overwrite: bool = False,
    render_samples: int = 0, materialize_images: bool = True,
) -> Path:
    source_images = list(source_images)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        source_images = source_images[:limit]
    split_dir, images_dir = output_dir / split, output_dir / split / "images"
    annotations_path = split_dir / "annotations.json"
    if annotations_path.exists() and not overwrite:
        raise FileExistsError(f"{annotations_path} exists; pass --overwrite")
    images_dir.mkdir(parents=True, exist_ok=True)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    exposure: dict[str, Counter[str]] = defaultdict(Counter)
    generator = random.Random(seed + (0 if split == "train" else 1_000_003))
    image_id = annotation_id = 1
    superclass_counts = superclass_counts or Counter()

    for source_path in tqdm(source_images, desc=f"Tiling xView {split}"):
        with Image.open(source_path) as opened:
            source_width, source_height = opened.size
            source = opened.convert("RGB") if materialize_images else None
        boxes = boxes_by_image.get(source_path.name, [])
        crop_specs = uniform_crop_specs(source_width, source_height, tile_size)
        if split == "train" and category_crops_per_source:
            crop_specs += category_crop_specs(boxes, source_width, source_height, tile_size, category_crops_per_source, superclass_counts, generator)
        for crop_index, crop in enumerate(crop_specs):
            mapped, seam_clipped = remap_owned_boxes(boxes, crop, tile_size, output_size, min_visibility)
            if not mapped and crop.stream == "uniform" and not _keep_empty(empty_crop_policy, empty_crop_fraction, generator):
                continue
            suffix = f"u_r{crop.row:02d}_c{crop.col:02d}" if crop.stream == "uniform" else f"c_{crop_index:04d}_{crop.selected_superclass.lower().replace(' ', '_')}"
            file_name = f"{source_path.stem}_{suffix}.jpg"
            available_w = max(0, min(tile_size, source_width - crop.x))
            available_h = max(0, min(tile_size, source_height - crop.y))
            pad_right, pad_bottom = tile_size - available_w, tile_size - available_h
            if materialize_images:
                assert source is not None
                save_crop(source, crop, tile_size, output_size, images_dir / file_name)
            images.append({
                "id": image_id, "width": output_size, "height": output_size,
                "file_name": file_name, "source_image": source_path.name,
                "source_image_id": int(source_path.stem), "crop_x": crop.x, "crop_y": crop.y,
                "source_tile_size": tile_size, "output_size": output_size,
                "resample_scale": output_size / tile_size, "stream": crop.stream,
                "selected_superclass": crop.selected_superclass,
                "pad_right": pad_right, "pad_bottom": pad_bottom,
                "seam_clipped_boxes": seam_clipped,
            })
            exposure[crop.stream]["crops"] += 1
            for ann in mapped:
                ann["id"], ann["image_id"] = annotation_id, image_id
                annotations.append(ann)
                annotation_id += 1
                exposure[crop.stream][ann.get("mapped_superclass") or "other"] += 1
            image_id += 1

    metadata = {
        "version": 2, "split": split, "seed": seed,
        "source_split_before_tiling": True, "tile_size": tile_size,
        "output_size": output_size, "resample_scale": output_size / tile_size,
        "ownership_rule": "box_center_in_half_open_crop", "min_box_visibility": min_visibility,
        "edge_policy": "zero_pad_right_bottom", "empty_crop_policy": empty_crop_policy,
        "empty_crop_fraction": empty_crop_fraction if empty_crop_policy == "sample" else None,
        "category_sampling": {
            "enabled": split == "train" and category_crops_per_source > 0,
            "crops_per_source": category_crops_per_source if split == "train" else 0,
            "rule": "inverse_sqrt_global_direct_superclass_frequency",
        },
        "direct_superclass_source_counts": dict(superclass_counts),
        "actual_exposure": {stream: dict(counts) for stream, counts in exposure.items()},
        "images_materialized": materialize_images,
    }
    coco = {
        "info": {"description": f"Tiled xView {split}", "xview_conversion": metadata},
        "licenses": [], "images": images, "annotations": annotations,
        "categories": [{"id": category_id, "name": name} for category_id, name in XVIEW_CATEGORIES.items()],
    }
    temporary = annotations_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(coco, handle, separators=(",", ":"))
        handle.write("\n")
    temporary.replace(annotations_path)
    with (split_dir / "conversion_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    if render_samples and not materialize_images:
        raise ValueError("--render-samples requires --materialize-images")
    render_representatives(coco, images_dir, split_dir / "previews", render_samples)
    print(f"Wrote {len(images)} crops and {len(annotations)} annotations to {split_dir}")
    print("Exposure: " + json.dumps(metadata["actual_exposure"], sort_keys=True))
    return annotations_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--geojson", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=960, help="Native source crop side.")
    parser.add_argument("--output-size", type=int, default=960, help="Saved square side/model input scale.")
    parser.add_argument("--min-box-visibility", type=float, default=0.2)
    parser.add_argument("--empty-crop-policy", choices=("keep", "drop", "sample"), default="keep")
    parser.add_argument("--empty-crop-fraction", type=float, default=0.1)
    parser.add_argument("--category-crops-per-source", type=int, default=0, help="Extra rare-class-centred training crops per source image.")
    parser.add_argument("--render-samples", type=int, default=0, help="Render sparse, dense, edge and seam crop previews per split.")
    parser.add_argument(
        "--materialize-images", action=argparse.BooleanOptionalAction, default=True,
        help="Disable for annotation-only 640/960 scale measurement passes.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    if args.tile_size <= 0 or args.output_size <= 0:
        raise ValueError("tile-size and output-size must be positive")
    if not 0.0 <= args.min_box_visibility <= 1.0:
        raise ValueError("min-box-visibility must be in [0, 1]")
    if not 0.0 <= args.empty_crop_fraction <= 1.0:
        raise ValueError("empty-crop-fraction must be in [0, 1]")
    if args.category_crops_per_source < 0:
        raise ValueError("category-crops-per-source must be non-negative")
    dataset_root = args.dataset_root.resolve()
    geojson = args.geojson.resolve() if args.geojson else _find_geojson(dataset_root)
    images_dir = args.images_dir.resolve() if args.images_dir else _find_train_images(dataset_root)
    source_images = list_images(images_dir)
    train_images, val_images = split_train_val(source_images, args.val_fraction, args.seed)
    boxes_by_image, ignored, malformed = read_source_boxes(geojson)
    source_names = {path.name for path in source_images}
    missing = sorted(set(boxes_by_image) - source_names)
    if missing:
        print(f"Skipping annotations for {len(missing)} missing source image(s): {missing[:5]}")
    if ignored:
        print("Skipped non-taxonomy labels: " + ", ".join(f"{key}={value}" for key, value in sorted(ignored.items())))
    if malformed:
        print(f"Skipped {malformed} malformed annotations")
    counts = direct_counts({name: boxes_by_image.get(name, []) for name in source_names})
    for split, split_images in (("train", train_images), ("val", val_images)):
        convert_split(
            args.output_dir.resolve(), split, split_images, boxes_by_image,
            tile_size=args.tile_size, output_size=args.output_size,
            category_crops_per_source=args.category_crops_per_source,
            superclass_counts=counts, min_visibility=args.min_box_visibility,
            empty_crop_policy=args.empty_crop_policy,
            empty_crop_fraction=args.empty_crop_fraction, seed=args.seed,
            limit=args.limit, overwrite=args.overwrite,
            render_samples=args.render_samples,
            materialize_images=args.materialize_images,
        )


if __name__ == "__main__":
    main()
