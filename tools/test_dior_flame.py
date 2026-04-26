"""Few-shot evaluation of FLAME (post-processing on top of OWLv2) on DIOR.

Pipeline per target class:
    1. Sample ``--support-images`` images from the DIOR ``train`` split that
       contain at least one ground-truth box of the class.
    2. Run OWLv2 over the support set with a low objectness threshold to
       collect dense proposals (``FlamePipeline.generate_proposals``).
    3. Use FLAME active selection to pick ``--shots`` informative proposals.
    4. Auto-label each picked proposal as positive / negative by IoU >= ``--iou-threshold``
       against any GT box of the target class (no human in the loop).
    5. Train the FLAME refiner (SVM by default) on those labels.
    6. Run the refiner on the test split, collect (box, refiner_score)
       triples, optionally NMS, and feed them into a single COCO eval.

The script also runs the matching zero-shot baseline on the same test images
so the FLAME delta is visible in one report.

Usage:
    python tools/test_dior_flame.py --model-size large --shots 30 \
        --support-images 8 --limit 200 --classes ship harbor

Requires the ``train`` extras (``pip install -e ".[train]"``) for
``faster-coco-eval`` plus ``scikit-learn`` (already a FLAME dep).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from OWLv2torch import OwlV2, tokenize
from OWLv2torch.torch_version.flame import FlameConfig, FlamePipeline


# ---------------------------------------------------------------------------
# Helpers (mirrors tools/test_dior.py so the two reports are comparable).
# ---------------------------------------------------------------------------
def class_name_to_query(name: str) -> str:
    """Convert DIOR class names like ``"Expressway service area"`` to a prompt."""
    cleaned = name.strip().lower().replace("-", " ").replace("_", " ")
    return f"a satellite photo of a {cleaned}"


INF = float("inf")
AREA_RANGES: list[tuple[str, tuple[float, float]]] = [
    ("all",       (0,        INF)),
    ("very_tiny", (0,        8 * 8)),
    ("tiny",      (8 * 8,    16 * 16)),
    ("small",     (0,        32 * 32)),
    ("medium",    (32 * 32,  96 * 96)),
    ("large",     (96 * 96,  INF)),
]


def build_coco_gt(dataset, class_names: list[str]) -> dict:
    """Convert an HF DIOR split into a COCO-format dict (matches test_dior.py)."""
    images, annotations = [], []
    ann_id = 1
    for sample in dataset:
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
    categories = [{"id": i, "name": n} for i, n in enumerate(class_names)]
    return {
        "info": {"description": "DIOR test set (HichTala/dior) — FLAME few-shot"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def coco_eval_with_custom_sizes(
    coco_gt_dict: dict,
    detections: list[dict],
    class_names: list[str],
    title: str,
    class_idxs: Optional[list[int]] = None,
) -> None:
    """Run COCO eval with custom area ranges and print a per-size summary."""
    from faster_coco_eval import COCO, COCOeval_faster

    coco_gt = COCO(coco_gt_dict)
    if not detections:
        print(f"\n=== {title}: no detections — skipping COCO eval. ===")
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

    if class_idxs is None:
        class_idxs = list(range(len(class_names)))

    print(f"\n=== {title}: Per-class AP @[0.50:0.95] (area=all) ===")
    all_idx = next(i for i, (lbl, _) in enumerate(AREA_RANGES) if lbl == "all")
    for k in class_idxs:
        name = class_names[k]
        per_cls = _mean(precision[:, :, k, all_idx, last_md_idx])
        print(f"  {name:30s} {per_cls:.4f}")


# ---------------------------------------------------------------------------
# Auto-labeling
# ---------------------------------------------------------------------------
def iou_xyxy_vs_xywh(box_xyxy: np.ndarray, gt_xywh: tuple[float, float, float, float]) -> float:
    """IoU between a single xyxy proposal and a single COCO-style xywh GT box."""
    ax1, ay1, ax2, ay2 = float(box_xyxy[0]), float(box_xyxy[1]), float(box_xyxy[2]), float(box_xyxy[3])
    gx, gy, gw, gh = gt_xywh
    bx1, by1, bx2, by2 = float(gx), float(gy), float(gx + gw), float(gy + gh)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def auto_label_candidates(
    flame: FlamePipeline,
    candidate_indices: list[int],
    support_samples: list[dict],
    class_idx: int,
    iou_threshold: float,
) -> list[bool]:
    """For each FLAME candidate, label True if its box matches a GT of ``class_idx``."""
    labels: list[bool] = []
    # Cache per-image GT lists.
    gt_cache: dict[int, list[tuple[float, float, float, float]]] = {}
    for sample_idx, sample in enumerate(support_samples):
        gts: list[tuple[float, float, float, float]] = []
        for bbox, cat in zip(sample["objects"]["bbox"], sample["objects"]["category"]):
            if int(cat) == class_idx:
                gts.append(tuple(float(v) for v in bbox))
        gt_cache[sample_idx] = gts

    for cand_idx in candidate_indices:
        prop = flame.proposals[cand_idx]
        gts = gt_cache.get(prop.image_idx, [])
        if not gts:
            labels.append(False)
            continue
        max_iou = max(iou_xyxy_vs_xywh(prop.box, gt) for gt in gts)
        labels.append(max_iou >= iou_threshold)
    return labels


# ---------------------------------------------------------------------------
# Support sampling
# ---------------------------------------------------------------------------
def sample_support_indices(
    support_index: dict[int, list[int]],
    class_idx: int,
    n: int,
    seed: int,
) -> list[int]:
    """Pick up to ``n`` train indices whose GT contains ``class_idx`` at least once."""
    eligible = sorted(support_index.get(class_idx, []))
    rng = random.Random(seed)
    rng.shuffle(eligible)
    return eligible[:n]


def seed_everything(seed: int) -> None:
    """Seed local RNGs used by this evaluation script."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_dior_images(samples: list[dict]) -> tuple[list[Image.Image], list[int]]:
    """Decode a DIOR batch to RGB PIL images plus image ids."""
    images = [sample["image"].convert("RGB") for sample in samples]
    image_ids = [int(sample["image_id"]) for sample in samples]
    return images, image_ids


def materialise_test_images_with_dataloader(
    test_ds,
    batch_size: int,
    num_workers: int,
) -> tuple[list[Image.Image], list[int]]:
    """Load test images with DataLoader workers so decode can be prefetched."""
    loader_kwargs = {
        "batch_size": max(1, batch_size),
        "shuffle": False,
        "num_workers": max(0, num_workers),
        "collate_fn": collate_dior_images,
        "pin_memory": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    test_images: list[Image.Image] = []
    test_image_ids: list[int] = []
    loader = DataLoader(test_ds, **loader_kwargs)
    for images, image_ids in tqdm(loader, desc="decode"):
        test_images.extend(images)
        test_image_ids.extend(image_ids)
    return test_images, test_image_ids


def build_support_index(train_ds, num_classes: int) -> dict[int, list[int]]:
    """Build train indices per class once, without loading train images per class."""
    support_index: dict[int, list[int]] = {i: [] for i in range(num_classes)}
    for sample_idx, objects in enumerate(tqdm(train_ds["objects"], desc="index train labels")):
        for cat in set(int(c) for c in objects["category"]):
            support_index.setdefault(cat, []).append(sample_idx)
    return support_index


# ---------------------------------------------------------------------------
# Per-class NMS (keeps result file bounded; helps mAP visibly).
# ---------------------------------------------------------------------------
def per_class_nms(detections: list[dict], iou_threshold: float) -> list[dict]:
    """Greedy NMS per (image_id, category_id), score = COCO 'score' field."""
    if iou_threshold <= 0.0 or iou_threshold >= 1.0:
        return detections
    try:
        from torchvision.ops import nms as tv_nms
    except ImportError:
        print("[WARN] torchvision.ops.nms unavailable; skipping NMS.")
        return detections

    grouped: dict[tuple[int, int], list[int]] = {}
    for i, det in enumerate(detections):
        grouped.setdefault((det["image_id"], det["category_id"]), []).append(i)

    keep_mask = np.zeros(len(detections), dtype=bool)
    for (_img, _cat), idxs in grouped.items():
        boxes_xyxy = np.empty((len(idxs), 4), dtype=np.float32)
        scores = np.empty(len(idxs), dtype=np.float32)
        for j, det_i in enumerate(idxs):
            x, y, w, h = detections[det_i]["bbox"]
            boxes_xyxy[j] = [x, y, x + w, y + h]
            scores[j] = detections[det_i]["score"]
        kept = tv_nms(
            torch.from_numpy(boxes_xyxy),
            torch.from_numpy(scores),
            iou_threshold,
        ).numpy()
        for k in kept:
            keep_mask[idxs[int(k)]] = True

    return [d for d, keep in zip(detections, keep_mask) if keep]


# ---------------------------------------------------------------------------
# Zero-shot baseline (single-class subset of tools/test_dior.py).
# ---------------------------------------------------------------------------
@torch.inference_mode()
def zero_shot_predict(
    model: OwlV2,
    images: list[Image.Image],
    image_ids: list[int],
    class_idx: int,
    text_query: str,
    device: str,
    score_threshold: float,
    top_k: Optional[int],
    batch_size: int = 1,
    fast_preprocess: bool = False,
    use_autocast: bool = True,
) -> list[dict]:
    """Plain OWLv2 detection for a single class, COCO format."""
    token_ids = tokenize([text_query], context_length=16, truncate=True).to(device)
    attention_mask = (token_ids == 0).to(device)

    detections: list[dict] = []
    bs = max(1, batch_size)
    for start in range(0, len(images), bs):
        chunk_imgs = images[start:start + bs]
        chunk_ids = image_ids[start:start + bs]
        target_size = torch.tensor(
            [img.size[::-1] for img in chunk_imgs],
            dtype=torch.float32,
            device=device,
        )
        pixel_values = model.preprocess_image(chunk_imgs, fast=fast_preprocess).to(device)
        device_type = torch.device(device).type
        autocast_enabled = use_autocast and device_type == "cuda"
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=autocast_enabled):
            pred_logits, objectness_logits, pred_boxes, _, _ = model.forward_object_detection(
                pixel_values, token_ids, attention_mask
            )
            boxes_xyxy = model.postprocess_boxes(pred_boxes.float(), target_size)  # [B, P*P, 4]
        pred_logits = pred_logits.float()
        objectness_logits = objectness_logits.float()
        scores = torch.sigmoid(pred_logits[:, :, 0]) * torch.sigmoid(objectness_logits)

        for b, img_id in enumerate(chunk_ids):
            keep = scores[b] >= score_threshold
            scores_b = scores[b, keep]
            boxes_b = boxes_xyxy[b, keep]
            if top_k is not None and scores_b.numel() > top_k:
                top_idx = torch.topk(scores_b, top_k).indices
                scores_b = scores_b[top_idx]
                boxes_b = boxes_b[top_idx]
            for box, score in zip(boxes_b.cpu().numpy(), scores_b.cpu().numpy()):
                x1, y1, x2, y2 = [float(v) for v in box]
                detections.append({
                    "image_id": int(img_id),
                    "category_id": int(class_idx),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                })
    return detections


# ---------------------------------------------------------------------------
# FLAME few-shot per class
# ---------------------------------------------------------------------------
def run_flame_for_class(
    model: OwlV2,
    class_idx: int,
    class_name: str,
    train_ds,
    support_index: dict[int, list[int]],
    test_images: list[Image.Image],
    test_image_ids: list[int],
    args,
) -> tuple[list[dict], list[dict], dict]:
    """Run the full FLAME pipeline for one class.

    Returns:
        flame_dets: COCO detections for this class from the refiner.
        zero_dets:  COCO detections for this class from the zero-shot baseline.
        info:       small status dict for logging (n_support, n_pos, n_neg, ...).
    """
    info: dict = {
        "class_name": class_name,
        "class_idx": class_idx,
        "n_support": 0,
        "n_proposals": 0,
        "n_candidates": 0,
        "n_pos": 0,
        "n_neg": 0,
        "trained": False,
        "skipped_reason": None,
    }
    text_query = class_name_to_query(class_name)
    stage_times: dict[str, float] = {}
    stage_pbar = tqdm(
        total=7,
        desc=f"class[{class_name}]",
        leave=False,
        dynamic_ncols=True,
    )

    def finish_stage(name: str, start_time: float) -> None:
        stage_times[name] = time.perf_counter() - start_time
        stage_pbar.set_postfix_str(f"{name} {stage_times[name]:.1f}s")
        stage_pbar.update(1)

    try:
        # 1. Sample support images.
        t0 = time.perf_counter()
        stage_pbar.set_postfix_str("sampling support")
        support_idx = sample_support_indices(
            support_index, class_idx, args.support_images, seed=args.seed + class_idx
        )
        if not support_idx:
            info["skipped_reason"] = "no train images contain this class"
            return [], [], info
        support_samples = [train_ds[i] for i in support_idx]
        support_images = [s["image"].convert("RGB") for s in support_samples]
        info["n_support"] = len(support_samples)
        finish_stage("support", t0)

        # 2-3. Generate proposals + select annotation candidates.
        cfg = FlameConfig(
            objectness_threshold=args.objectness_threshold,
            text_score_threshold=args.text_score_threshold,
            max_proposals_per_image=args.max_proposals_per_image,
            num_clusters=args.shots,
            classifier_type=args.classifier,
            svm_kernel=args.svm_kernel,
            svm_C=args.svm_c,
            classifier_threshold=args.classifier_threshold,
            fast_image_preprocess=args.fast_preprocess,
            use_autocast=args.autocast,
            # Silence per-call FLAME prints during the test loop (one print pair
            # per detect() call is too chatty when the test set has 100s of images).
            verbose=False,
        )
        flame = FlamePipeline(model, cfg)
        t0 = time.perf_counter()
        stage_pbar.set_postfix_str("generating support proposals")
        flame.generate_proposals(support_images, text_query)
        info["n_proposals"] = len(flame.proposals)
        finish_stage("proposals", t0)
        if not flame.proposals:
            info["skipped_reason"] = "no proposals from support set"
            return [], [], info

        t0 = time.perf_counter()
        stage_pbar.set_postfix_str("selecting candidates")
        candidate_indices = flame.select_annotation_candidates()
        info["n_candidates"] = len(candidate_indices)
        finish_stage("select", t0)

        # 4. Auto-label.
        t0 = time.perf_counter()
        stage_pbar.set_postfix_str("auto-labeling")
        labels = auto_label_candidates(
            flame, candidate_indices, support_samples, class_idx, args.iou_threshold
        )
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        info["n_pos"], info["n_neg"] = n_pos, n_neg
        finish_stage("label", t0)

        if n_pos == 0 or n_neg == 0:
            info["skipped_reason"] = (
                f"degenerate auto-labels (pos={n_pos}, neg={n_neg}); "
                f"try --support-images/--shots/--objectness-threshold/--iou-threshold"
            )
            return [], [], info

        t0 = time.perf_counter()
        stage_pbar.set_postfix_str("training refiner")
        flame.label_candidates(candidate_indices, labels)
        flame.train_refiner()
        info["trained"] = True
        finish_stage("train", t0)

        # 5. Run on test set in image-batches. Chunking bounds memory and gives us
        # a progress bar. Verbose printing is silenced via FlameConfig above.
        t0 = time.perf_counter()
        stage_pbar.set_postfix_str("running FLAME")
        flame_dets: list[dict] = []
        bs = max(1, args.detect_batch_size)
        n = len(test_images)
        pbar = tqdm(total=n, desc=f"FLAME[{class_name}]", leave=False, dynamic_ncols=True)
        for start in range(0, n, bs):
            end = min(start + bs, n)
            chunk_imgs = test_images[start:end]
            chunk_ids = test_image_ids[start:end]
            results = flame.detect(chunk_imgs, text_query)
            for r, img_id in zip(results, chunk_ids):
                if len(r["boxes"]) == 0:
                    continue
                for box, refiner_score, base_score in zip(
                    r["boxes"], r["refiner_scores"], r["scores"]
                ):
                    x1, y1, x2, y2 = [float(v) for v in box]
                    score = float(refiner_score)
                    if args.score_combine == "product":
                        score = score * float(base_score)
                    flame_dets.append({
                        "image_id": int(img_id),
                        "category_id": int(class_idx),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": score,
                    })
            pbar.update(end - start)
        pbar.close()
        finish_stage("flame", t0)

        # 6. Zero-shot baseline (same query, same images) for a side-by-side delta.
        t0 = time.perf_counter()
        stage_pbar.set_postfix_str("running zero-shot")
        zero_dets = zero_shot_predict(
            model,
            test_images,
            test_image_ids,
            class_idx=class_idx,
            text_query=text_query,
            device=str(next(model.parameters()).device),
            score_threshold=args.score_threshold,
            top_k=args.top_k,
            batch_size=args.detect_batch_size,
            fast_preprocess=args.fast_preprocess,
            use_autocast=args.autocast,
        )
        finish_stage("zero", t0)
        info["stage_times"] = stage_times
        return flame_dets, zero_dets, info
    finally:
        stage_pbar.close()


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", default="large", choices=["base", "large"])
    parser.add_argument("--split", default="test", help="DIOR split for evaluation")
    parser.add_argument("--train-split", default="train", help="DIOR split for support sampling")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap number of test images (for smoke tests).",
    )
    parser.add_argument(
        "--classes", nargs="*", default=None,
        help="Restrict evaluation to these DIOR class names (default: all).",
    )

    # FLAME knobs.
    parser.add_argument("--shots", type=int, default=30,
                        help="Number of FLAME annotation candidates per class.")
    parser.add_argument("--support-images", type=int, default=8,
                        help="Number of training images to sample per class.")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for auto-labeling candidates.")
    parser.add_argument("--objectness-threshold", type=float, default=0.1)
    parser.add_argument("--text-score-threshold", type=float, default=0.0,
                        help="Minimum sigmoid(text logit) for OWLv2 proposals; 0 keeps all.")
    parser.add_argument("--max-proposals-per-image", type=int, default=500)
    parser.add_argument("--classifier", default="svm", choices=["svm", "mlp"])
    parser.add_argument("--classifier-threshold", type=float, default=0.5,
                        help="Minimum FLAME refiner score to keep at inference; 0.5 is the SVM boundary.")
    parser.add_argument("--svm-kernel", default="rbf", choices=["linear", "poly", "rbf", "sigmoid"],
                        help="SVM kernel for the FLAME refiner.")
    parser.add_argument("--svm-c", type=float, default=1.0,
                        help="SVM C regularization parameter for the FLAME refiner.")
    parser.add_argument("--score-combine", default="refiner",
                        choices=["refiner", "product"],
                        help="Use refiner prob alone, or refiner * (sigmoid_class * sigmoid_obj).")

    # Eval knobs.
    parser.add_argument("--score-threshold", type=float, default=0.0,
                        help="Used only by the zero-shot baseline.")
    parser.add_argument("--top-k", type=int, default=300,
                        help="Per-image top-k cap for the zero-shot baseline.")
    parser.add_argument("--detect-batch-size", type=int, default=None,
                        help="Images per inference batch. Default: 32 for base, 8 for large.")
    parser.add_argument("--image-loader-batch-size", type=int, default=32,
                        help="Images per DataLoader batch while materialising the test split.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers for test image decoding.")
    parser.add_argument("--fast-preprocess", action="store_true",
                        help="Use torchvision resize instead of scipy exact resize.")
    parser.add_argument("--no-autocast", dest="autocast", action="store_false",
                        help="Disable CUDA autocast during OWLv2 inference.")
    parser.set_defaults(autocast=True)
    parser.add_argument("--nms-iou", type=float, default=0.5,
                        help="Per-class NMS IoU; set <=0 to disable.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-detections", default=None,
                        help="Optional path; writes a JSON dict {flame, zero_shot, info}.")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.detect_batch_size is None:
        args.detect_batch_size = 32 if args.model_size == "base" else 8
    seed_everything(args.seed)

    print(f"Loading OwlV2 ({args.model_size}) on {args.device}...")
    model = OwlV2(args.model_size).eval().to(args.device)
    if str(args.device).startswith("cuda") and not args.fast_preprocess:
        print(
            "[WARN] Using exact scipy image preprocessing. This is CPU-bound and can "
            "make GPU utilization look bursty/low. Add --fast-preprocess for higher throughput."
        )
    print(
        f"Inference settings: detect_batch_size={args.detect_batch_size}, "
        f"autocast={args.autocast}, fast_preprocess={args.fast_preprocess}, "
        f"num_workers={args.num_workers}"
    )

    print(f"Loading HichTala/dior splits: train='{args.train_split}', test='{args.split}'...")
    from datasets import load_dataset
    train_ds = load_dataset("HichTala/dior", split=args.train_split)
    test_ds = load_dataset("HichTala/dior", split=args.split)
    if args.limit is not None:
        test_ds = test_ds.select(range(min(args.limit, len(test_ds))))
    print(f"  -> train={len(train_ds)}, test={len(test_ds)}")

    class_names: list[str] = test_ds.features["objects"]["category"].feature.names
    coco_gt_dict = build_coco_gt(test_ds, class_names)

    if args.classes:
        unknown = [c for c in args.classes if c not in class_names]
        if unknown:
            raise SystemExit(f"Unknown class names: {unknown}\nKnown: {class_names}")
        target_class_idxs = [class_names.index(c) for c in args.classes]
    else:
        target_class_idxs = list(range(len(class_names)))
    print(f"  -> evaluating {len(target_class_idxs)}/{len(class_names)} classes: "
          f"{[class_names[i] for i in target_class_idxs]}")

    support_index = build_support_index(train_ds, len(class_names))

    # Materialise test images once — we reuse them for every class to avoid
    # paying HF dataset decode N_classes times.
    print("Materialising test images...")
    test_images, test_image_ids = materialise_test_images_with_dataloader(
        test_ds,
        batch_size=args.image_loader_batch_size,
        num_workers=args.num_workers,
    )

    flame_dets_all: list[dict] = []
    zero_dets_all: list[dict] = []
    infos: list[dict] = []
    for class_idx in target_class_idxs:
        class_name = class_names[class_idx]
        print(f"\n=== Class {class_idx}: {class_name} ===")
        flame_dets, zero_dets, info = run_flame_for_class(
            model, class_idx, class_name, train_ds, support_index,
            test_images, test_image_ids, args,
        )
        infos.append(info)
        if info["skipped_reason"]:
            print(f"  SKIP: {info['skipped_reason']}")
        else:
            print(
                f"  proposals={info['n_proposals']}  "
                f"candidates={info['n_candidates']}  "
                f"pos={info['n_pos']}  neg={info['n_neg']}  "
                f"flame_dets={len(flame_dets)}  zero_dets={len(zero_dets)}"
            )
            stage_times = info.get("stage_times")
            if stage_times:
                timing = "  timings: " + "  ".join(
                    f"{name}={seconds:.1f}s" for name, seconds in stage_times.items()
                )
                print(timing)
        flame_dets_all.extend(flame_dets)
        zero_dets_all.extend(zero_dets)

    # NMS.
    if args.nms_iou > 0:
        n_before = len(flame_dets_all)
        flame_dets_all = per_class_nms(flame_dets_all, args.nms_iou)
        zero_dets_all = per_class_nms(zero_dets_all, args.nms_iou)
        print(f"\nNMS@{args.nms_iou}: flame {n_before} -> {len(flame_dets_all)}")

    if args.save_detections:
        with open(args.save_detections, "w") as f:
            json.dump({"flame": flame_dets_all, "zero_shot": zero_dets_all, "info": infos}, f)
        print(f"Wrote detections + info to {args.save_detections}")

    # Two side-by-side reports on the *same* GT, restricted (in interpretation)
    # to the evaluated classes — COCO eval will report nan for the others.
    coco_eval_with_custom_sizes(
        coco_gt_dict, zero_dets_all, class_names,
        title=f"DIOR zero-shot baseline ({args.model_size})",
        class_idxs=target_class_idxs,
    )
    coco_eval_with_custom_sizes(
        coco_gt_dict, flame_dets_all, class_names,
        title=(
            f"DIOR FLAME few-shot (shots={args.shots}, "
            f"support_images={args.support_images}, classifier={args.classifier})"
        ),
        class_idxs=target_class_idxs,
    )


if __name__ == "__main__":
    main()
