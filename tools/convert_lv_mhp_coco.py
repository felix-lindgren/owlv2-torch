"""Convert LV-MHP-v1 person-parsing masks to COCO detection annotations.

LV-MHP-v1 stores one semantic PNG mask per person. For detection training, this
script emits one bounding box for every non-background category present in each
person mask. Multiple people wearing the same category therefore remain
separate COCO instances.

The dataset's ``train_list.txt`` covers training and validation together. By
default, a seeded 10% holdout becomes ``val`` and the official ``test_list.txt``
is left untouched. Pass ``--val-source test`` to instead use the official test
list as ``val`` and train on all 4,000 entries in ``train_list.txt``.

Example:

    uv run python tools/convert_lv_mhp_coco.py \
        --dataset-root /mnt/datasets/fashion/LV-MHP-v1 \
        --output-dir /mnt/datasets/fashion/lv_mhp_coco

The resulting files can be passed to ``prototype_train/train_text.py`` as:

    --train-annotations /mnt/datasets/fashion/lv_mhp_coco/train/annotations.json
    --train-images /mnt/datasets/fashion/lv_mhp_coco/train/images
    --val-annotations /mnt/datasets/fashion/lv_mhp_coco/val/annotations.json
    --val-images /mnt/datasets/fashion/lv_mhp_coco/val/images
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm


LV_MHP_CATEGORIES = {
    1: "hat",
    2: "hair",
    3: "sunglasses",
    4: "upper clothes",
    5: "skirt",
    6: "pants",
    7: "dress",
    8: "belt",
    9: "left shoe",
    10: "right shoe",
    11: "face",
    12: "left leg",
    13: "right leg",
    14: "left arm",
    15: "right arm",
    16: "bag",
    17: "scarf",
    18: "torso skin",
}


def read_image_list(path: Path) -> list[str]:
    """Read and validate an LV-MHP image list."""
    if not path.is_file():
        raise FileNotFoundError(f"Image list does not exist: {path}")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name]
    if not names:
        raise ValueError(f"Image list is empty: {path}")
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate entries in {path}: {duplicates[:5]}")
    return names


def split_train_val(
    image_names: list[str], val_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Make a stable random holdout while preserving source order in each split."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")
    if len(image_names) < 2:
        raise ValueError("At least two images are required for a train/val split")

    val_count = max(1, min(len(image_names) - 1, round(len(image_names) * val_fraction)))
    shuffled_indices = list(range(len(image_names)))
    random.Random(seed).shuffle(shuffled_indices)
    val_indices = set(shuffled_indices[:val_count])
    train = [name for index, name in enumerate(image_names) if index not in val_indices]
    val = [name for index, name in enumerate(image_names) if index in val_indices]
    return train, val


def mask_to_annotation(
    mask: np.ndarray,
    category_id: int,
    *,
    annotation_id: int,
    image_id: int,
    person_id: int,
    mask_file: str,
) -> dict[str, Any] | None:
    """Build one COCO annotation from one category in one person's mask."""
    ys, xs = np.nonzero(mask == category_id)
    if xs.size == 0:
        return None
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1],
        "area": int(xs.size),
        "iscrowd": 0,
        # Non-standard provenance fields are ignored by CocoDetection/COCOeval.
        "person_id": person_id,
        "source_mask": mask_file,
    }


def index_masks(annotations_dir: Path) -> dict[str, list[tuple[int, Path]]]:
    """Index masks by image stem and validate ``stem_count_person.png`` names."""
    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"Annotations directory does not exist: {annotations_dir}")

    indexed: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    declared_counts: dict[str, int] = {}
    for path in sorted(annotations_dir.glob("*.png")):
        try:
            image_stem, count_text, person_text = path.stem.rsplit("_", 2)
            declared_count = int(count_text)
            person_id = int(person_text)
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Expected mask name '<image>_<person-count>_<person-id>.png', got {path.name}"
            ) from exc
        previous_count = declared_counts.setdefault(image_stem, declared_count)
        if previous_count != declared_count:
            raise ValueError(f"Inconsistent person counts in masks for image {image_stem}")
        indexed[image_stem].append((person_id, path))

    for image_stem, masks in indexed.items():
        expected_count = declared_counts[image_stem]
        masks.sort()
        person_ids = [person_id for person_id, _ in masks]
        if len(masks) != expected_count or person_ids != list(range(1, expected_count + 1)):
            raise ValueError(
                f"Image {image_stem} declares {expected_count} people but has person IDs "
                f"{person_ids}"
            )
    return dict(indexed)


def _materialize_image(source: Path, destination: Path, mode: str, overwrite: bool) -> None:
    if destination.is_symlink() and not destination.exists():
        destination.unlink()
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            return
        destination.unlink()

    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "symlink":
        destination.symlink_to(os.path.relpath(source, start=destination.parent))
    elif mode != "none":
        raise ValueError(f"Unknown image mode: {mode}")


def convert_split(
    dataset_root: Path,
    output_dir: Path,
    split: str,
    image_names: Iterable[str],
    mask_index: dict[str, list[tuple[int, Path]]],
    *,
    image_mode: str = "symlink",
    limit: Optional[int] = None,
    overwrite: bool = False,
) -> Path:
    """Convert one LV-MHP-v1 image-name split and return its COCO JSON path."""
    names = list(image_names)
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        names = names[:limit]

    split_dir = output_dir / split
    images_dir = split_dir / "images"
    annotations_path = split_dir / "annotations.json"
    if annotations_path.exists() and not overwrite:
        raise FileExistsError(
            f"{annotations_path} already exists; pass --overwrite to replace it"
        )
    split_dir.mkdir(parents=True, exist_ok=True)
    if image_mode != "none":
        images_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    used_image_ids: set[int] = set()
    annotation_id = 1

    for fallback_image_id, file_name in enumerate(
        tqdm(names, desc=f"Converting LV-MHP-v1 {split}"), start=1
    ):
        source_image = dataset_root / "images" / file_name
        if not source_image.is_file():
            raise FileNotFoundError(f"Listed image does not exist: {source_image}")
        image_stem = Path(file_name).stem
        try:
            image_id = int(image_stem)
        except ValueError:
            image_id = fallback_image_id
        if image_id in used_image_ids:
            raise ValueError(f"Duplicate numeric image id {image_id} in split {split}")
        used_image_ids.add(image_id)

        masks = mask_index.get(image_stem)
        if not masks:
            raise FileNotFoundError(f"No person masks found for image {file_name}")
        with Image.open(source_image) as image:
            width, height = image.size
        images.append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": file_name,
        })
        if image_mode != "none":
            _materialize_image(
                source_image, images_dir / file_name, image_mode, overwrite
            )

        for person_id, mask_path in masks:
            with Image.open(mask_path) as mask_image:
                mask = np.asarray(mask_image)
            if mask.ndim != 2:
                raise ValueError(
                    f"Expected a grayscale mask, got shape {mask.shape} in {mask_path}"
                )
            if mask.shape != (height, width):
                raise ValueError(
                    f"Mask {mask_path} is {mask.shape[1]}x{mask.shape[0]}, but "
                    f"image {file_name} is {width}x{height}"
                )
            category_ids = np.unique(mask)
            invalid = [
                int(value)
                for value in category_ids
                if int(value) != 0 and int(value) not in LV_MHP_CATEGORIES
            ]
            if invalid:
                raise ValueError(f"Mask {mask_path} contains unknown labels {invalid}")
            for category_id in category_ids:
                category_id = int(category_id)
                if category_id == 0:
                    continue
                annotation = mask_to_annotation(
                    mask,
                    category_id,
                    annotation_id=annotation_id,
                    image_id=image_id,
                    person_id=person_id,
                    mask_file=mask_path.name,
                )
                if annotation is not None:
                    annotations.append(annotation)
                    annotation_id += 1

    coco = {
        "info": {
            "description": f"LV-MHP-v1 {split} converted from per-person parsing masks"
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category_id, "name": name, "supercategory": "human parsing"}
            for category_id, name in LV_MHP_CATEGORIES.items()
        ],
    }
    temporary_path = annotations_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(coco, file, separators=(",", ":"))
        file.write("\n")
    temporary_path.replace(annotations_path)
    print(
        f"Wrote {len(images)} images and {len(annotations)} annotations to {split_dir}"
    )
    return annotations_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="LV-MHP-v1 directory containing images, annotations, and list files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory for split/images and split/annotations.json outputs.",
    )
    parser.add_argument(
        "--val-source",
        choices=("train", "test"),
        default="train",
        help=(
            "Hold validation data out of train_list.txt, or use test_list.txt as "
            "validation (default: train)."
        ),
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of train_list.txt reserved for val when --val-source=train.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Train/val split seed.")
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Also convert test_list.txt to a separate test output split.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("symlink", "copy", "none"),
        default="symlink",
        help=(
            "How to populate each split's images directory. With 'none', pass the "
            "source dataset's images directory to train_text.py (default: symlink)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Convert at most this many images per split (useful for smoke checks).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing annotation JSON and materialized images.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    train_list = read_image_list(dataset_root / "train_list.txt")
    test_list = read_image_list(dataset_root / "test_list.txt")
    mask_index = index_masks(dataset_root / "annotations")

    if args.val_source == "train":
        train_names, val_names = split_train_val(
            train_list, args.val_fraction, args.seed
        )
    else:
        train_names, val_names = train_list, test_list

    splits = [("train", train_names), ("val", val_names)]
    if args.include_test:
        if args.val_source == "test":
            raise ValueError(
                "--include-test cannot be combined with --val-source=test because "
                "that would duplicate the same images in val and test"
            )
        splits.append(("test", test_list))

    for split, names in splits:
        annotations_path = convert_split(
            dataset_root,
            args.output_dir,
            split,
            names,
            mask_index,
            image_mode=args.image_mode,
            limit=args.limit,
            overwrite=args.overwrite,
        )
        image_root = (
            dataset_root / "images"
            if args.image_mode == "none"
            else annotations_path.parent / "images"
        )
        print(f"CocoDetection root:    {image_root}")
        print(f"CocoDetection annFile: {annotations_path}")


if __name__ == "__main__":
    main()
