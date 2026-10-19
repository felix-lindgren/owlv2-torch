"""Pseudo-label images with an OWLv2 teacher for self-training.

Writes a COCO file whose annotations carry the teacher's ``score``; train with
``train_text.py --pseudo-label-threshold T [--pseudo-ignore-threshold L]`` so the
thresholds stay a training-time choice and one labelling pass serves every sweep.
Store generously (``--min-score`` well below any threshold you expect to use).

The teacher can be far more expensive than the student, since it runs once:
``--model-size large``, a fine-tuned ``--checkpoint``, and ``--tile-size`` for
zoomed sliding-window inference merged with a full-image pass.

Two input modes:

``--coco ann.json --images DIR``
    Label exactly that file's images and reuse its categories, so the output
    shares category ids with a validation set built the same way. With
    ``--keep-gt-shots K --seed S`` the images ``train_text.py --shots-per-class K
    --seed S`` would select keep their human labels (no score) instead, which is
    the semi-supervised setup: a K-shot teacher, pseudo-labels for the rest.
``--images DIR --classes ...``
    Label every image under DIR with a free vocabulary (ids 1..N).

Run the teacher on the RTX 3060 with ``CUDA_VISIBLE_DEVICES=1`` while the 5070 Ti
trains; the script prints the device it got.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ovd_eval import (
    SATELLITE_QUERY_TEMPLATE,
    build_owlv2_model,
    generate_tiles,
    merge_tiled_detections,
    run_owlv2_inference,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


class TiledImageDataset(Dataset):
    """One item per (image, tile); the full image is one more tile when requested."""

    def __init__(self, images: list[dict], root: Path, tile_size: int, overlap: int, full_image: bool):
        self.root = root
        self.items = []
        for image in images:
            width, height = image["width"], image["height"]
            tiles = []
            if tile_size > 0:
                tiles = generate_tiles(width, height, tile_size, overlap)
            if full_image or not tiles:
                tiles.append((0, 0, width, height))
            # An image no larger than one tile yields the full frame twice.
            for tile in dict.fromkeys(tiles):
                self.items.append((image, tile))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        image, (x0, y0, x1, y1) = self.items[index]
        with Image.open(self.root / image["file_name"]) as pil:
            pil = pil.convert("RGB")
            if (x0, y0, x1, y1) != (0, 0, pil.width, pil.height):
                pil = pil.crop((x0, y0, x1, y1))
            crop = pil.copy()
        return {
            "image": crop,
            "image_id": image["id"],
            "target_size": (crop.height, crop.width),
            "offset": (float(x0), float(y0)),
        }


def scan_images(root: Path) -> list[dict]:
    images = []
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    for image_id, path in enumerate(paths, start=1):
        with Image.open(path) as pil:
            width, height = pil.size
        images.append({
            "id": image_id,
            "file_name": path.relative_to(root).as_posix(),
            "width": width,
            "height": height,
        })
    return images


def read_vocabulary(args) -> list[str]:
    names = list(args.classes or [])
    if args.classes_file:
        lines = Path(args.classes_file).read_text(encoding="utf-8").splitlines()
        names += [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    if not names:
        raise SystemExit("--classes or --classes-file is required without --coco")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise SystemExit(f"Duplicate class names: {duplicates}")
    return names


def shot_image_ids(annotations: Path, images: Path, shots: int, seed: int) -> set[int]:
    """The images ``train_text.py --shots-per-class shots --seed seed`` trains on."""
    from torchvision.datasets import CocoDetection

    from prototype_train.train import select_shot_indices

    dataset = CocoDetection(root=str(images), annFile=str(annotations))
    category_ids = sorted(dataset.coco.getCatIds())
    return {
        dataset.ids[index]
        for index in select_shot_indices(dataset, category_ids, shots, seed)
    }


def build_trt_teacher(args):
    """OWLv2 with a TensorRT vision tower built from the teacher's own weights.

    Engines bake in their weights and only run on the GPU architecture they were
    built on, so the default path is per checkpoint and per compute capability.
    The first run builds it (minutes); later runs reuse it while the checkpoint's
    vision weights still match the engine's fingerprint.
    """
    from OWLv2torch.torch_version.owlv2_tensorrt import OwlV2TRT
    from prototype_train.train_text import load_text_checkpoint

    major, minor = torch.cuda.get_device_capability(torch.device(args.device))
    name = f"owlv2_vis_{args.model_size}_sm{major}{minor}.engine"
    if args.trt_engine:
        engine_path = Path(args.trt_engine)
    elif args.checkpoint:
        engine_path = Path(args.checkpoint).with_suffix("") / "trt" / name
    else:
        engine_path = REPO_ROOT / "trt_engines" / name

    model = OwlV2TRT(
        engine_path=engine_path,
        model_type=args.model_size,
        build_missing=False,
        # detect() runs the torch heads; only the tower is worth an engine.
        trt_heads=False,
        min_batch=1,
        opt_batch=args.batch_size,
        max_batch=args.batch_size,
    ).to(args.device)
    if args.checkpoint:
        load_text_checkpoint(model, args.checkpoint)

    rebuild = not model.trt_vision_matches_weights()
    if not rebuild:
        max_batch = int(model.trt.engine.get_tensor_profile_shape("image", 0)[2][0])
        rebuild = max_batch < args.batch_size
    if rebuild:
        model.build_trt_vision()
    if model.trt is None:
        raise SystemExit(f"TensorRT engine unavailable at {engine_path} (is 'tensorrt' installed?)")
    print(f"TensorRT vision tower: {engine_path}")
    return model


def clip_detection(detection: dict, width: int, height: int, min_size: float) -> dict | None:
    x, y, w, h = detection["bbox"]
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    return {**detection, "bbox": [x1, y1, x2 - x1, y2 - y1]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="Image root; COCO file_name values are relative to it.")
    p.add_argument("--coco", help="Label this COCO file's images with its categories.")
    p.add_argument("--classes", nargs="+", help="Vocabulary when no --coco is given.")
    p.add_argument("--classes-file", help="Vocabulary file, one class per line.")
    p.add_argument("--output", required=True)
    p.add_argument("--model-size", default="large", choices=["base", "large"])
    p.add_argument("--checkpoint", help="Fine-tuned teacher weights from train_text.py.")
    p.add_argument("--query-template", default=SATELLITE_QUERY_TEMPLATE)
    p.add_argument(
        "--tensorrt",
        action="store_true",
        help="Run the vision tower as a TensorRT engine built from the teacher's weights.",
    )
    p.add_argument(
        "--trt-engine",
        help="Engine path. Default: next to --checkpoint (or in trt_engines/), named by GPU arch.",
    )
    p.add_argument("--tile-size", type=int, default=0, help="Sliding-window tile size in source pixels; 0 = off.")
    p.add_argument("--tile-overlap", type=int, default=160)
    p.add_argument(
        "--full-image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also run the whole image when tiling, so objects larger than a tile survive.",
    )
    p.add_argument("--nms-iou", type=float, default=0.5, help="Per-class NMS, within and across tiles.")
    p.add_argument("--min-score", type=float, default=0.05, help="Boxes below this are not stored.")
    p.add_argument("--top-k", type=int, default=300, help="Detections kept per forward pass.")
    p.add_argument("--max-per-image", type=int, default=300, help="Cap after merging tiles.")
    p.add_argument("--min-box-size", type=float, default=2.0, help="In source pixels, after clipping.")
    p.add_argument("--keep-gt-shots", type=int, help="With --coco: these shot images keep human labels.")
    p.add_argument("--seed", type=int, default=0, help="Seed for --keep-gt-shots, as in train_text.py.")
    p.add_argument("--limit", type=int, help="Label only the first N images (smoke runs).")
    p.add_argument("--batch-size", type=int, default=None, help="Default: 16 for base, 4 for large.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.keep_gt_shots is not None and not args.coco:
        p.error("--keep-gt-shots needs --coco")
    if args.batch_size is None:
        args.batch_size = 16 if args.model_size == "base" else 4
    root = Path(args.images)

    if args.coco:
        coco = json.loads(Path(args.coco).read_text(encoding="utf-8"))
        categories = sorted(coco["categories"], key=lambda c: c["id"])
        images = sorted(coco["images"], key=lambda im: im["id"])
    else:
        categories = [{"id": i, "name": name} for i, name in enumerate(read_vocabulary(args), start=1)]
        images = scan_images(root)
    if args.limit is not None:
        images = images[: args.limit]
    class_names = [c["name"] for c in categories]

    kept_annotations = []
    gt_image_ids: set[int] = set()
    if args.keep_gt_shots is not None:
        gt_image_ids = shot_image_ids(Path(args.coco), root, args.keep_gt_shots, args.seed)
        selected = {im["id"] for im in images}
        gt_image_ids &= selected
        kept_annotations = [
            {key: value for key, value in ann.items() if key != "score"}
            for ann in coco["annotations"]
            if ann["image_id"] in gt_image_ids
        ]
        print(f"Keeping human labels on {len(gt_image_ids)} shot images ({len(kept_annotations)} boxes)")
    to_label = [im for im in images if im["id"] not in gt_image_ids]

    dataset = TiledImageDataset(to_label, root, args.tile_size, args.tile_overlap, args.full_image)
    print(
        f"Teacher: OWLv2 {args.model_size}"
        + (f" + {args.checkpoint}" if args.checkpoint else "")
        + f", {len(class_names)} classes, prompt {args.query_template!r}"
    )
    print(f"Labelling {len(to_label)} images as {len(dataset)} forward passes")
    if str(args.device).startswith("cuda"):
        print(f"Device: {torch.cuda.get_device_name(torch.device(args.device))}")

    if args.tensorrt:
        model = build_trt_teacher(args)
    else:
        model = build_owlv2_model(args.model_size, args.device)
        if args.checkpoint:
            from prototype_train.train_text import load_text_checkpoint

            load_text_checkpoint(model, args.checkpoint)
    model.eval()

    detections = run_owlv2_inference(
        model,
        dataset,
        class_names,
        device=args.device,
        score_threshold=args.min_score,
        top_k=args.top_k,
        batch_size=args.batch_size,
        fast_preprocess=True,
        use_autocast=True,
        num_workers=args.num_workers,
        desc="Pseudo-labelling",
        query_template=args.query_template,
    )
    # Always, not only across tiles: neighbouring patches emit near-duplicate boxes,
    # and as pseudo-labels each duplicate would become its own matched target.
    detections = merge_tiled_detections(detections, args.nms_iou)

    by_image: dict[int, list[dict]] = {}
    for detection in detections:
        by_image.setdefault(detection["image_id"], []).append(detection)
    image_by_id = {im["id"]: im for im in to_label}
    annotations = list(kept_annotations)
    next_id = max((ann["id"] for ann in annotations), default=0) + 1
    for image_id, image_detections in by_image.items():
        image = image_by_id[image_id]
        image_detections.sort(key=lambda d: -d["score"])
        for detection in image_detections[: args.max_per_image]:
            clipped = clip_detection(detection, image["width"], image["height"], args.min_box_size)
            if clipped is None:
                continue
            x, y, w, h = clipped["bbox"]
            annotations.append({
                "id": next_id,
                "image_id": image_id,
                "category_id": categories[clipped["category_id"]]["id"],
                "bbox": [round(v, 2) for v in (x, y, w, h)],
                "area": round(w * h, 2),
                "iscrowd": 0,
                "score": round(clipped["score"], 5),
            })
            next_id += 1

    output = {
        "info": {
            "description": "OWLv2 pseudo-labels (tools/pseudo_label.py)",
            "date_created": datetime.now().isoformat(timespec="seconds"),
            "teacher": {
                key: getattr(args, key)
                for key in (
                    "model_size", "checkpoint", "tensorrt", "query_template", "tile_size", "tile_overlap",
                    "full_image", "nms_iou", "min_score", "top_k", "max_per_image",
                    "keep_gt_shots", "seed", "coco",
                )
            },
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output), encoding="utf-8")

    scored = [ann for ann in annotations if "score" in ann]
    print(f"\nWrote {len(annotations)} annotations ({len(scored)} pseudo) for {len(images)} images to {out_path}")
    n = max(len(to_label), 1)
    for threshold in (0.1, 0.2, 0.3, 0.5):
        count = sum(ann["score"] >= threshold for ann in scored)
        print(f"  score >= {threshold:.1f}: {count:>8d} boxes ({count / n:.2f}/image)")
    per_class = Counter(ann["category_id"] for ann in scored if ann["score"] >= 0.2)
    name_by_id = {c["id"]: c["name"] for c in categories}
    top = ", ".join(f"{name_by_id[c]} {k}" for c, k in per_class.most_common(8))
    print(f"  most common at >= 0.2: {top or 'none'}")


if __name__ == "__main__":
    main()
