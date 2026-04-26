from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from OWLv2torch import OwlV2, tokenize


INF = float("inf")
AREA_RANGES: list[tuple[str, tuple[float, float]]] = [
    ("all", (0, INF)),
    ("very_tiny", (0, 8 * 8)),
    ("tiny", (8 * 8, 16 * 16)),
    ("small", (0, 32 * 32)),
    ("medium", (32 * 32, 96 * 96)),
    ("large", (96 * 96, INF)),
]

DOTA_V1_5_CLASSES: list[str] = [
    "plane",
    "ship",
    "storage-tank",
    "baseball-diamond",
    "tennis-court",
    "basketball-court",
    "ground-track-field",
    "harbor",
    "bridge",
    "large-vehicle",
    "small-vehicle",
    "helicopter",
    "roundabout",
    "soccer-ball-field",
    "swimming-pool",
    "container-crane",
]


def class_name_to_query(name: str) -> str:
    cleaned = name.strip().lower().replace("-", " ").replace("_", " ")
    return f"a satellite photo of a {cleaned}"


def detect(
    model: OwlV2,
    pixel_values: torch.Tensor,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run OWLv2 open-vocabulary detection."""
    batch_size = pixel_values.shape[0]

    if token_ids.shape[0] != batch_size:
        token_ids = token_ids.repeat(batch_size, 1)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat(batch_size, 1)
    num_queries = token_ids.shape[0] // batch_size

    image_feats, _ = model._image_grid_features(pixel_values)
    text_features = model.get_text_features(token_ids, attention_mask)
    text_features = text_features.reshape(batch_size, num_queries, -1)
    query_token_ids = token_ids.reshape(batch_size, num_queries, -1)
    query_mask = query_token_ids[..., 0] > 0

    pred_logits, _ = model.class_head(image_feats, text_features, query_mask)
    objectness_logits = model.objectness_head(image_feats)[..., 0]
    pred_boxes = torch.sigmoid(model.box_head(image_feats) + model.box_bias)
    return pred_logits, objectness_logits, pred_boxes


class DiorDetectionDataset(Dataset):
    def __init__(self, split: str = "test", limit: Optional[int] = None):
        from datasets import load_dataset

        ds = load_dataset("HichTala/dior", split=split)
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))
        self.ds = ds
        self.class_names: list[str] = ds.features["objects"]["category"].feature.names

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        sample = self.ds[idx]
        image = sample["image"].convert("RGB")
        return {
            "image": image,
            "image_id": int(sample["image_id"]),
            "target_size": (image.height, image.width),
            "offset": (0.0, 0.0),
        }

    def build_coco_gt(self) -> dict:
        images = []
        annotations = []
        ann_id = 1
        for sample in self.ds:
            img_id = int(sample["image_id"])
            images.append({
                "id": img_id,
                "width": int(sample["width"]),
                "height": int(sample["height"]),
                "file_name": f"{img_id}.jpg",
            })
            bboxes = sample["objects"]["bbox"]
            cats = sample["objects"]["category"]
            areas = sample["objects"].get("area")
            for i, (bbox, cat) in enumerate(zip(bboxes, cats)):
                x, y, w, h = [float(v) for v in bbox]
                area = float(areas[i]) if areas is not None else w * h
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cat),
                    "bbox": [x, y, w, h],
                    "area": area,
                    "iscrowd": 0,
                })
                ann_id += 1

        return {
            "info": {"description": "DIOR test set (HichTala/dior)"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": [{"id": i, "name": n} for i, n in enumerate(self.class_names)],
        }


def parse_dota_label(label_path: Path) -> list[dict]:
    annotations: list[dict] = []
    with open(label_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("imagesource:") or line.startswith("gsd:"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                coords = [float(v) for v in parts[:8]]
            except ValueError:
                continue
            class_name = parts[8]
            try:
                difficult = int(parts[9])
            except ValueError:
                difficult = 0
            xs = coords[0::2]
            ys = coords[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            w = max(0.0, x_max - x_min)
            h = max(0.0, y_max - y_min)
            annotations.append({
                "class_name": class_name,
                "difficult": difficult,
                "bbox_xywh": (x_min, y_min, w, h),
            })
    return annotations


def discover_dota_samples(data_root: Path) -> list[dict]:
    img_dir = data_root / "images"
    lbl_dir = data_root / "labels"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        raise FileNotFoundError(
            f"Expected '{img_dir}' and '{lbl_dir}' to exist. Use --data-root to override."
        )

    samples: list[dict] = []
    img_paths = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    )
    for idx, img_path in enumerate(img_paths, start=1):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        samples.append({
            "image_id": idx,
            "stem": img_path.stem,
            "image_path": img_path,
            "label_path": lbl_path,
        })
    return samples


def generate_tiles(
    width: int,
    height: int,
    tile_size: int,
    overlap: int,
) -> list[tuple[int, int, int, int]]:
    if tile_size <= 0:
        return [(0, 0, width, height)]
    stride = max(1, tile_size - overlap)

    def _starts(extent: int) -> list[int]:
        if extent <= tile_size:
            return [0]
        starts = list(range(0, max(0, extent - tile_size) + 1, stride))
        last = extent - tile_size
        if starts[-1] != last:
            starts.append(last)
        return starts

    tiles = []
    for y0 in _starts(height):
        for x0 in _starts(width):
            x1 = min(width, x0 + tile_size) if tile_size < width else width
            y1 = min(height, y0 + tile_size) if tile_size < height else height
            tiles.append((x0, y0, x1, y1))
    return tiles


class DotaDetectionDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        limit: Optional[int] = None,
        tile_size: int = 0,
        tile_overlap: int = 200,
        include_difficult: bool = False,
    ):
        self.class_names = list(DOTA_V1_5_CLASSES)
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}
        self.include_difficult = include_difficult

        samples = discover_dota_samples(data_root)
        if limit is not None:
            samples = samples[:limit]
        self.samples = samples

        self.items: list[tuple[dict, tuple[int, int, int, int], tuple[int, int]]] = []
        for sample in self.samples:
            with Image.open(sample["image_path"]) as im:
                width, height = im.size
            for tile in generate_tiles(width, height, tile_size, tile_overlap):
                self.items.append((sample, tile, (width, height)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        sample, (x0, y0, x1, y1), (width, height) = self.items[idx]
        with Image.open(sample["image_path"]) as image:
            image = image.convert("RGB")
            if (x0, y0, x1, y1) == (0, 0, width, height):
                crop = image.copy()
            else:
                crop = image.crop((x0, y0, x1, y1)).copy()
        return {
            "image": crop,
            "image_id": int(sample["image_id"]),
            "target_size": (crop.height, crop.width),
            "offset": (float(x0), float(y0)),
        }

    def build_coco_gt(self) -> dict:
        images: list[dict] = []
        annotations: list[dict] = []
        ann_id = 1
        for sample in self.samples:
            with Image.open(sample["image_path"]) as im:
                width, height = im.size
            images.append({
                "id": sample["image_id"],
                "width": width,
                "height": height,
                "file_name": sample["image_path"].name,
            })
            for ann in parse_dota_label(sample["label_path"]):
                if not self.include_difficult and ann["difficult"] == 1:
                    continue
                cls_name = ann["class_name"]
                if cls_name not in self.class_to_id:
                    continue
                x, y, w, h = ann["bbox_xywh"]
                if w <= 0 or h <= 0:
                    continue
                annotations.append({
                    "id": ann_id,
                    "image_id": sample["image_id"],
                    "category_id": self.class_to_id[cls_name],
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                })
                ann_id += 1

        return {
            "info": {"description": "DOTAv1.5 (local)"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": [{"id": i, "name": n} for i, n in enumerate(self.class_names)],
        }


def collate_detection_items(samples: list[dict]) -> dict:
    return {
        "images": [sample["image"] for sample in samples],
        "image_ids": [int(sample["image_id"]) for sample in samples],
        "target_sizes": [sample["target_size"] for sample in samples],
        "offsets": [sample["offset"] for sample in samples],
    }


def run_owlv2_inference(
    model: OwlV2,
    dataset: Dataset,
    class_names: list[str],
    device: str,
    score_threshold: float,
    top_k: Optional[int],
    batch_size: int,
    fast_preprocess: bool,
    use_autocast: bool,
    num_workers: int,
    desc: str,
) -> list[dict]:
    queries = [class_name_to_query(c) for c in class_names]
    text_inputs = tokenize(queries, context_length=16, truncate=True).to(device)
    attention_mask = (text_inputs == 0).to(device)

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": max(0, num_workers),
        "collate_fn": collate_detection_items,
        "pin_memory": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    detections: list[dict] = []
    device_type = torch.device(device).type
    autocast_enabled = use_autocast and device_type == "cuda"
    loader = DataLoader(dataset, **loader_kwargs)

    for batch in tqdm(loader, total=len(loader), desc=f"{desc} (bs={batch_size})"):
        images = batch["images"]
        image_ids = batch["image_ids"]
        offsets = batch["offsets"]
        target_sizes = torch.tensor(batch["target_sizes"], dtype=torch.float32, device=device)
        pixel_values = model.preprocess_image(images, fast=fast_preprocess).to(device)

        with torch.inference_mode():
            with torch.autocast(
                device_type=device_type,
                dtype=torch.float16,
                enabled=autocast_enabled,
            ):
                pred_logits, objectness_logits, pred_boxes = detect(
                    model, pixel_values, text_inputs, attention_mask
                )
                boxes_xyxy_batch = model.postprocess_boxes(pred_boxes.float(), target_sizes)

        pred_logits = pred_logits.float()
        objectness_logits = objectness_logits.float()
        class_probs = torch.sigmoid(pred_logits)
        obj_probs = torch.sigmoid(objectness_logits)
        scores_per_class = class_probs * obj_probs.unsqueeze(-1)
        max_scores, max_labels = scores_per_class.max(dim=-1)

        for b, img_id in enumerate(image_ids):
            scores_b = max_scores[b]
            labels_b = max_labels[b]
            boxes_b = boxes_xyxy_batch[b]

            keep = scores_b >= score_threshold
            scores_b = scores_b[keep]
            labels_b = labels_b[keep]
            boxes_b = boxes_b[keep]

            if top_k is not None and scores_b.numel() > top_k:
                top_idx = torch.topk(scores_b, top_k).indices
                scores_b = scores_b[top_idx]
                labels_b = labels_b[top_idx]
                boxes_b = boxes_b[top_idx]

            boxes_np = boxes_b.detach().cpu().numpy()
            scores_np = scores_b.detach().cpu().numpy()
            labels_np = labels_b.detach().cpu().numpy()
            ox, oy = offsets[b]

            for box, score, label in zip(boxes_np, scores_np, labels_np):
                x1, y1, x2, y2 = [float(v) for v in box]
                detections.append({
                    "image_id": int(img_id),
                    "category_id": int(label),
                    "bbox": [x1 + ox, y1 + oy, x2 - x1, y2 - y1],
                    "score": float(score),
                })

    return detections


def coco_eval_with_custom_sizes(
    coco_gt_dict: dict,
    detections: list[dict],
    class_names: list[str],
    title: str,
) -> None:
    from faster_coco_eval import COCO, COCOeval_faster

    coco_gt = COCO(coco_gt_dict)
    if not detections:
        print("No detections produced — skipping COCO eval.")
        return
    coco_dt = coco_gt.loadRes(detections)

    cocoeval = COCOeval_faster(coco_gt, coco_dt, iouType="bbox")
    cocoeval.params.areaRng = [list(r) for _, r in AREA_RANGES]
    cocoeval.params.areaRngLbl = [lbl for lbl, _ in AREA_RANGES]
    cocoeval.params.maxDets = [1, 10, 100]
    cocoeval.evaluate()
    cocoeval.accumulate()

    precision = cocoeval.eval["precision"]
    recall = cocoeval.eval["recall"]
    iou_thrs = cocoeval.params.iouThrs
    max_dets = cocoeval.params.maxDets
    iou50_idx = int(np.where(np.isclose(iou_thrs, 0.50))[0][0])
    iou75_idx = int(np.where(np.isclose(iou_thrs, 0.75))[0][0])
    last_md_idx = len(max_dets) - 1

    def _mean(arr: np.ndarray) -> float:
        valid = arr[arr > -1]
        return float(valid.mean()) if valid.size else float("nan")

    print(f"\n=== {title}: COCO mAP ===")
    print(f"{'metric':<28s} {'IoU':<14s} {'value':>8s}")
    for area_idx, (lbl, _) in enumerate(AREA_RANGES):
        ap_all = _mean(precision[:, :, :, area_idx, last_md_idx])
        print(f"  AP   area={lbl:<10s}    @[0.50:0.95]   {ap_all:>8.4f}")
        if lbl == "all":
            ap_50 = _mean(precision[iou50_idx, :, :, area_idx, last_md_idx])
            ap_75 = _mean(precision[iou75_idx, :, :, area_idx, last_md_idx])
            print(f"  AP   area={lbl:<10s}    @0.50          {ap_50:>8.4f}")
            print(f"  AP   area={lbl:<10s}    @0.75          {ap_75:>8.4f}")

    print()
    for area_idx, (lbl, _) in enumerate(AREA_RANGES):
        ar = _mean(recall[:, :, area_idx, last_md_idx])
        print(f"  AR   area={lbl:<10s}  maxDets={max_dets[-1]:<4d}  {ar:>8.4f}")

    print("\n=== Per-class AP @[0.50:0.95] (area=all) ===")
    all_idx = next(i for i, (lbl, _) in enumerate(AREA_RANGES) if lbl == "all")
    for k, name in enumerate(class_names):
        per_cls = _mean(precision[:, :, k, all_idx, last_md_idx])
        print(f"  {name:30s} {per_cls:.4f}")


def save_detections(path: str, detections: list[dict]) -> None:
    import json

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(detections, f)
