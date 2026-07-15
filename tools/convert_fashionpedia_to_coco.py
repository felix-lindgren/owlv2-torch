"""Convert Hugging Face Fashionpedia splits to a local COCO dataset.

The source dataset stores bounding boxes as Pascal VOC ``xyxy`` coordinates.
This script writes COCO ``xywh`` annotations and preserves the original encoded
image bytes, avoiding a lossy image decode/re-encode cycle.

Example:

    uv run python tools/convert_fashionpedia_to_coco.py \
        --output-dir data/fashionpedia_coco

The resulting training split can be used with ``torchvision.CocoDetection`` as:

    root=data/fashionpedia_coco/train/images
    annFile=data/fashionpedia_coco/train/annotations.json
"""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Optional

from PIL import Image as PILImage


FASHIONPEDIA_DATASET_ID = "detection-datasets/fashionpedia"
FORMAT_EXTENSIONS = {
    "BMP": ".bmp",
    "GIF": ".gif",
    "JPEG": ".jpg",
    "PNG": ".png",
    "TIFF": ".tiff",
    "WEBP": ".webp",
}


def xyxy_to_xywh(bbox: list[float] | tuple[float, ...]) -> list[float]:
    """Convert a Pascal VOC box to COCO coordinates."""
    if len(bbox) != 4:
        raise ValueError(f"Expected a four-value bbox, got {bbox!r}")
    x_min, y_min, x_max, y_max = (float(value) for value in bbox)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def _read_encoded_image(image_value: Any) -> tuple[bytes, str, tuple[int, int]]:
    """Return encoded bytes, suffix, and ``(width, height)`` for an HF image."""
    if isinstance(image_value, Mapping):
        encoded = image_value.get("bytes")
        source_path = image_value.get("path")
    elif isinstance(image_value, (str, Path)):
        encoded = None
        source_path = image_value
    else:
        raise TypeError(
            "Expected an undecoded Hugging Face image mapping with 'bytes' or "
            f"'path', got {type(image_value).__name__}"
        )

    if encoded is None:
        if not source_path:
            raise ValueError("Hugging Face image has neither encoded bytes nor a path")
        encoded = Path(source_path).read_bytes()
    else:
        encoded = bytes(encoded)

    with PILImage.open(BytesIO(encoded)) as image:
        image_format = image.format
        size = image.size
    if not image_format:
        raise ValueError("Could not determine the encoded image format")
    suffix = FORMAT_EXTENSIONS.get(image_format.upper(), f".{image_format.lower()}")
    return encoded, suffix, size


def _class_names(dataset: Any) -> list[str]:
    try:
        names = dataset.features["objects"]["category"].feature.names
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(
            "Dataset does not expose ClassLabel names at "
            "features['objects']['category']"
        ) from exc
    return [str(name) for name in names]


def convert_split(
    dataset: Any,
    split: str,
    output_dir: Path,
    *,
    limit: Optional[int] = None,
    overwrite: bool = False,
) -> Path:
    """Convert one already-loaded Fashionpedia split and return its JSON path."""
    from datasets import Image as HFImage
    from tqdm import tqdm

    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        dataset = dataset.select(range(min(limit, len(dataset))))

    split_dir = output_dir / split
    images_dir = split_dir / "images"
    annotations_path = split_dir / "annotations.json"
    if annotations_path.exists() and not overwrite:
        raise FileExistsError(
            f"{annotations_path} already exists; pass --overwrite to replace it"
        )
    images_dir.mkdir(parents=True, exist_ok=True)

    # Keep image payloads encoded so conversion does not alter JPEG pixels.
    dataset = dataset.cast_column("image", HFImage(decode=False))
    class_names = _class_names(dataset)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    seen_image_ids: set[int] = set()
    annotation_id = 1

    for sample in tqdm(dataset, desc=f"Converting Fashionpedia {split}"):
        image_id = int(sample["image_id"])
        if image_id in seen_image_ids:
            raise ValueError(f"Duplicate image_id {image_id} in split {split!r}")
        seen_image_ids.add(image_id)

        encoded, suffix, actual_size = _read_encoded_image(sample["image"])
        width, height = int(sample["width"]), int(sample["height"])
        if actual_size != (width, height):
            raise ValueError(
                f"Image {image_id} metadata is {width}x{height}, but encoded image "
                f"is {actual_size[0]}x{actual_size[1]}"
            )

        file_name = f"{image_id}{suffix}"
        image_path = images_dir / file_name
        if overwrite or not image_path.exists():
            image_path.write_bytes(encoded)
        images.append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": file_name,
        })

        objects = sample["objects"]
        bboxes = objects["bbox"]
        category_ids = objects["category"]
        areas = objects.get("area")
        if len(bboxes) != len(category_ids):
            raise ValueError(
                f"Image {image_id} has {len(bboxes)} boxes but "
                f"{len(category_ids)} category IDs"
            )
        if areas is not None and len(areas) != len(bboxes):
            raise ValueError(
                f"Image {image_id} has {len(bboxes)} boxes but {len(areas)} areas"
            )

        for object_index, (bbox, category_id) in enumerate(zip(bboxes, category_ids)):
            x, y, box_width, box_height = xyxy_to_xywh(bbox)
            if box_width <= 0.0 or box_height <= 0.0:
                continue
            category_id = int(category_id)
            if category_id < 0 or category_id >= len(class_names):
                raise ValueError(
                    f"Image {image_id} has category_id={category_id}, outside "
                    f"the valid range [0, {len(class_names)})"
                )
            area = (
                float(areas[object_index])
                if areas is not None
                else box_width * box_height
            )
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x, y, box_width, box_height],
                "area": area,
                "iscrowd": 0,
            })
            annotation_id += 1

    coco = {
        "info": {
            "description": (
                f"Fashionpedia {split} converted from {FASHIONPEDIA_DATASET_ID}"
            )
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category_id, "name": name}
            for category_id, name in enumerate(class_names)
        ],
    }

    # Replace the final JSON atomically so an interrupted conversion never leaves
    # a partially written annotation file that looks complete.
    temporary_path = annotations_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(coco, file, separators=(",", ":"))
        file.write("\n")
    temporary_path.replace(annotations_path)

    print(
        f"Wrote {len(images)} images and {len(annotations)} annotations to "
        f"{split_dir}"
    )
    return annotations_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory for split/images and split/annotations.json outputs.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Fashionpedia splits to convert (default: train val).",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert at most this many images per split (useful for smoke tests).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing annotations and image files.",
    )
    parser.add_argument(
        "--dataset-id",
        default=FASHIONPEDIA_DATASET_ID,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    from datasets import load_dataset

    args = parse_args()
    for split in args.splits:
        print(f"Loading {args.dataset_id} split={split}...")
        dataset = load_dataset(
            args.dataset_id,
            split=split,
            cache_dir=str(args.cache_dir) if args.cache_dir is not None else None,
        )
        annotations_path = convert_split(
            dataset,
            split,
            args.output_dir,
            limit=args.limit,
            overwrite=args.overwrite,
        )
        print(f"CocoDetection root:    {annotations_path.parent / 'images'}")
        print(f"CocoDetection annFile: {annotations_path}")


if __name__ == "__main__":
    main()
