"""Export a HichTala Hugging Face detection set (DIOR, DOTA) to COCO files on disk.

``prototype_train/train_text.py`` and ``tools/pseudo_label.py`` read COCO files
plus an image folder; ``tools/ovd_eval.py`` reads DIOR straight from the hub.
This writes the former with the same image ids and the same 0-based category ids
as ``DiorDetectionDataset.build_coco_gt``, so both paths score the same boxes.
Images are copied as their original encoded bytes, never re-encoded.

Example:

    uv run python tools/convert_dior_to_coco.py --output-dir /mnt/datasets/sat/dior_coco

which yields ``{train,validation,test}/annotations.json`` and ``.../images``.
``--dataset HichTala/dota`` exports DOTA v1.5 the same way (16 classes, already
cut into 512 px patches by the uploader; same xywh boxes as DIOR).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm


def convert_split(dataset: str, split: str, output_dir: Path) -> None:
    from datasets import Image as HfImage, load_dataset

    ds = load_dataset(dataset, split=split)
    class_names = ds.features["objects"]["category"].feature.names
    ds = ds.cast_column("image", HfImage(decode=False))

    image_dir = output_dir / split / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for sample in tqdm(ds, desc=f"{dataset} {split}"):
        image_id = int(sample["image_id"])
        data = sample["image"]["bytes"]
        suffix = ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
        file_name = f"{image_id}{suffix}"
        (image_dir / file_name).write_bytes(data)
        images.append({
            "id": image_id,
            "file_name": file_name,
            "width": int(sample["width"]),
            "height": int(sample["height"]),
        })
        objects = sample["objects"]
        areas = objects.get("area")
        for index, (bbox, category) in enumerate(zip(objects["bbox"], objects["category"])):
            x, y, w, h = [float(v) for v in bbox]
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": int(category),
                "bbox": [x, y, w, h],
                "area": float(areas[index]) if areas is not None else w * h,
                "iscrowd": 0,
            })

    coco = {
        "info": {"description": f"{split} ({dataset})"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": name} for i, name in enumerate(class_names)],
    }
    (output_dir / split / "annotations.json").write_text(json.dumps(coco), encoding="utf-8")
    print(f"  {split}: {len(images)} images, {len(annotations)} boxes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="HichTala/dior")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    args = parser.parse_args()
    for split in args.splits:
        convert_split(args.dataset, split, args.output_dir)


if __name__ == "__main__":
    main()
