"""Point (box-centre) similarity and COCO-style average precision over it.

A point detector predicts only object centres. Ground truth keeps its boxes,
but only to scale the localisation tolerance: the similarity is COCO's
single-keypoint OKS with the box area as the object scale,

    OKS = exp(-d^2 / (2 * area * (2 * sigma)^2)),

so a centre miss of ``2.355 * sigma * sqrt(area)`` scores 0.5. With the default
sigma that is about half the object's side, for a 5 px car and a 40 px tank
alike. The same function drives the training loss, the matching cost and the
evaluation, so training optimises what is scored.

pycocotools' keypoint mode is not used: an unmatched single-keypoint detection
has zero area there and is only ever charged to a size bucket starting at 0.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

# OKS = 0.5 at a centre miss of ~half the box side.
DEFAULT_POINT_SIGMA = 0.21
OKS_THRESHOLDS = np.linspace(0.5, 0.95, 10)
RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)
# sqrt(area) buckets in the evaluated image's pixels. These are AI-TOD's edges;
# they are in model-input pixels when called from the in-training evaluator.
POINT_SIZE_BUCKETS = (
    ("very_tiny", 0.0, 8.0),
    ("tiny", 8.0, 16.0),
    ("small", 16.0, 32.0),
    ("medium", 32.0, float("inf")),
)


def pairwise_oks(
    points: torch.Tensor, target_boxes: torch.Tensor, sigma: float
) -> torch.Tensor:
    """``[N, M]`` OKS between ``points`` ``[N, 2]`` and ``cxcywh`` boxes ``[M, 4]``.

    Units only need to agree between the two inputs (normalised or pixels).
    """
    dtype = torch.promote_types(points.dtype, torch.float32)
    distance_sq = (points.to(dtype)[:, None, :] - target_boxes[None, :, :2].to(dtype)).pow(2).sum(-1)
    area = (target_boxes[:, 2] * target_boxes[:, 3]).to(dtype).clamp_min(1e-12)
    return torch.exp(-distance_sq / (2.0 * area * (2.0 * sigma) ** 2)[None, :])


def _match_image(oks: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """COCO greedy matching: ``[T, num_dets]`` true positives, ``[T, num_gts]`` recovered.

    ``oks`` is ``[num_dets, num_gts]`` with detections sorted by descending score.
    Each detection takes the unmatched ground truth with the highest OKS that
    clears the threshold.
    """
    num_dets, num_gts = oks.shape
    tp = np.zeros((len(thresholds), num_dets), dtype=bool)
    taken = np.zeros((len(thresholds), num_gts), dtype=bool)
    if num_gts == 0:
        return tp, taken
    candidates = np.flatnonzero(oks.max(axis=1) >= thresholds.min())
    for t_index, threshold in enumerate(thresholds):
        for det in candidates:
            row = np.where(taken[t_index], -1.0, oks[det])
            gt = int(row.argmax())
            if row[gt] >= threshold:
                taken[t_index, gt] = True
                tp[t_index, det] = True
    return tp, taken


def _average_precision(scores: np.ndarray, tp: np.ndarray, num_gts: int) -> np.ndarray:
    """101-point interpolated AP per threshold row of ``tp`` (COCOeval's recipe)."""
    order = np.argsort(-scores, kind="mergesort")
    tp = tp[:, order]
    tp_sum = np.cumsum(tp, axis=1, dtype=np.float64)
    fp_sum = np.cumsum(~tp, axis=1, dtype=np.float64)
    ap = np.zeros(tp.shape[0])
    for t_index in range(tp.shape[0]):
        recall = tp_sum[t_index] / num_gts
        precision = tp_sum[t_index] / np.maximum(tp_sum[t_index] + fp_sum[t_index], 1e-12)
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        indices = np.searchsorted(recall, RECALL_THRESHOLDS, side="left")
        sampled = np.zeros(len(RECALL_THRESHOLDS))
        valid = indices < len(precision)
        sampled[valid] = precision[indices[valid]]
        ap[t_index] = sampled.mean()
    return ap


class PointAPAccumulator:
    """Collects per-image point detections and ground truth, then scores them.

    ``update`` takes pixel-space ``points [P, 2]`` (centres), ``scores [P]``,
    ``labels [P]`` and ground truth ``gt_boxes [M, 4]`` cxcywh with
    ``gt_labels [M]``. ``max_detections`` caps each image's detections by score
    across classes, as the box evaluator's ``top_k`` does.
    """

    def __init__(self, sigma: float = DEFAULT_POINT_SIGMA, max_detections: int = 1500):
        self.sigma = sigma
        self.max_detections = max_detections
        self.scores: dict[int, list[np.ndarray]] = defaultdict(list)
        self.tp: dict[int, list[np.ndarray]] = defaultdict(list)
        self.num_gts: dict[int, int] = defaultdict(int)
        # Per size bucket: GT count, recovered at OKS 0.5, recovered averaged over thresholds.
        self.bucket_total = np.zeros(len(POINT_SIZE_BUCKETS))
        self.bucket_hit_50 = np.zeros(len(POINT_SIZE_BUCKETS))
        self.bucket_hit = np.zeros(len(POINT_SIZE_BUCKETS))

    def update(self, points, scores, labels, gt_boxes, gt_labels) -> None:
        points, scores, labels = points.double().cpu(), scores.float().cpu(), labels.cpu()
        gt_boxes, gt_labels = gt_boxes.double().cpu(), gt_labels.cpu()
        if scores.numel() > self.max_detections:
            keep = torch.topk(scores, self.max_detections).indices
            points, scores, labels = points[keep], scores[keep], labels[keep]
        gt_side = (gt_boxes[:, 2] * gt_boxes[:, 3]).clamp_min(0).sqrt().numpy()
        gt_bucket = np.full(len(gt_side), -1)
        for bucket, (_, low, high) in enumerate(POINT_SIZE_BUCKETS):
            gt_bucket[(gt_side >= low) & (gt_side < high)] = bucket

        for label in set(labels.tolist()) | set(gt_labels.tolist()):
            det_mask, gt_mask = labels == label, gt_labels == label
            det_scores = scores[det_mask]
            order = torch.argsort(det_scores, descending=True, stable=True)
            det_points = points[det_mask][order]
            det_scores = det_scores[order].numpy()
            class_gts = gt_boxes[gt_mask]
            oks = (
                pairwise_oks(det_points, class_gts, self.sigma).numpy()
                if len(det_points) and len(class_gts)
                else np.zeros((len(det_points), len(class_gts)))
            )
            self.scores[label].append(det_scores)
            tp, recovered = _match_image(oks, OKS_THRESHOLDS)
            self.tp[label].append(tp)
            self.num_gts[label] += len(class_gts)

            buckets = gt_bucket[gt_mask.numpy()]
            for bucket in range(len(POINT_SIZE_BUCKETS)):
                in_bucket = buckets == bucket
                self.bucket_total[bucket] += in_bucket.sum()
                self.bucket_hit_50[bucket] += recovered[0, in_bucket].sum()
                self.bucket_hit[bucket] += recovered[:, in_bucket].sum() / len(OKS_THRESHOLDS)

    def compute(self, per_class: bool = False) -> dict:
        """``map`` (OKS .50:.95), ``map_50``, ``map_75``, recall, per-size recall.

        ``map`` is the mean over classes with ground truth, as in COCO. With
        ``per_class`` the result also holds ``classes``, ``map_per_class`` and
        ``mar_per_class`` tensors (OKS .50:.95, recall at ``max_detections``).
        """
        per_class_ap, per_class_recall, class_labels = [], [], []
        for label, num_gts in sorted(self.num_gts.items()):
            if num_gts == 0:
                continue  # COCO leaves classes without ground truth out of the mean
            scores = np.concatenate(self.scores[label]) if self.scores[label] else np.zeros(0)
            tp = (
                np.concatenate(self.tp[label], axis=1)
                if self.tp[label]
                else np.zeros((len(OKS_THRESHOLDS), 0), dtype=bool)
            )
            per_class_ap.append(_average_precision(scores, tp, num_gts))
            per_class_recall.append(tp.sum(axis=1) / num_gts)
            class_labels.append(label)
        if not per_class_ap:
            return {"map": -1.0, "map_50": -1.0, "map_75": -1.0, "mar": -1.0, "mar_50": -1.0}
        ap = np.mean(per_class_ap, axis=0)
        recall = np.mean(per_class_recall, axis=0)
        metrics = {
            "map": float(ap.mean()),
            "map_50": float(ap[0]),
            "map_75": float(ap[5]),
            "mar": float(recall.mean()),
            "mar_50": float(recall[0]),
        }
        for bucket, (name, _, _) in enumerate(POINT_SIZE_BUCKETS):
            total = self.bucket_total[bucket]
            metrics[f"mar_{name}"] = float(self.bucket_hit[bucket] / total) if total else -1.0
            metrics[f"mar_50_{name}"] = float(self.bucket_hit_50[bucket] / total) if total else -1.0
            metrics[f"num_gt_{name}"] = float(total)
        if per_class:
            metrics["classes"] = torch.tensor(class_labels)
            metrics["map_per_class"] = torch.tensor([float(ap.mean()) for ap in per_class_ap])
            metrics["mar_per_class"] = torch.tensor([float(r.mean()) for r in per_class_recall])
        return metrics
