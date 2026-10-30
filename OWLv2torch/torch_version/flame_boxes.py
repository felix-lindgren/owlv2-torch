"""A shared, ridge-regularized residual box regressor on frozen visual features."""

from __future__ import annotations

import math

import torch
from torch.nn.functional import normalize
from torchvision.ops import box_iou


class BoxRefiner:
    def __init__(self, *, ridge=0.01, min_iou=0.3, max_proposals_per_object=4,
                 max_center_shift=0.25, max_log_scale=0.4, strength=0.5):
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be finite and positive")
        if not math.isfinite(min_iou) or not 0 < min_iou <= 1:
            raise ValueError("min_iou must be in (0, 1]")
        if not isinstance(max_proposals_per_object, int) or max_proposals_per_object < 1:
            raise ValueError("max_proposals_per_object must be a positive integer")
        if not all(math.isfinite(v) and 0 < v <= 1 for v in (max_center_shift, max_log_scale)):
            raise ValueError("Box correction bounds must be in (0, 1]")
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("strength must be in [0, 1]")
        self.config = dict(ridge=ridge, min_iou=min_iou,
                           max_proposals_per_object=max_proposals_per_object,
                           max_center_shift=max_center_shift, max_log_scale=max_log_scale,
                           strength=strength)
        self.weight = None
        self.stats = None

    @staticmethod
    def _boxes(boxes, device):
        boxes = torch.as_tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4)
        if not bool(torch.isfinite(boxes).all()) or bool((boxes[:, 2:] <= boxes[:, :2]).any()):
            raise ValueError("Boxes must be finite xyxy coordinates with positive width and height")
        return boxes

    @staticmethod
    def _proposals(boxes, device):
        """Proposal boxes may be empty: OWLv2 clamps the boxes of padding patches of a non-square
        image to its edge. An empty box overlaps nothing, so it is never matched or corrected."""
        boxes = torch.as_tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4)
        if not bool(torch.isfinite(boxes).all()) or bool((boxes[:, 2:] < boxes[:, :2]).any()):
            raise ValueError("Proposal boxes must be finite xyxy coordinates")
        return boxes

    @staticmethod
    def _design(features):
        x = normalize(features.float(), dim=-1)
        return torch.cat((x, x.new_ones((len(x), 1))), dim=-1)

    @torch.inference_mode()
    def prepare_support(self, support, *, device="cpu"):
        """Retain only matched training rows on CPU for later incremental fits."""
        device = torch.device(device)
        records = []
        for visual, proposals, truth in support:
            visual = visual.detach().float().to(device)
            if visual.ndim != 2 or visual.shape[1] == 0 or not bool(torch.isfinite(visual).all()):
                raise ValueError("Visual features must be a finite [P,D] tensor with D > 0")
            proposals, truth = self._proposals(proposals, device), self._boxes(truth, device)
            if len(visual) != len(proposals):
                raise ValueError("Features and proposal boxes must have the same row count")
            chosen = torch.empty(0, dtype=torch.long, device=device)
            if len(proposals) and len(truth):
                overlap, matched = box_iou(proposals, truth).max(1)
                eligible = (overlap >= self.config["min_iou"]).nonzero()[:, 0]
                selected = []
                for object_id in matched[eligible].unique():
                    own = eligible[matched[eligible] == object_id]
                    selected.append(own[overlap[own].argsort(descending=True, stable=True)
                                        [:self.config["max_proposals_per_object"]]])
                if selected:
                    chosen = torch.cat(selected)
            records.append((visual[chosen].cpu(), proposals[chosen].cpu(), truth.cpu().clone()))
        return records

    @torch.inference_mode()
    def fit(self, support, *, device="cpu"):
        """Fit from (visual features [P,D], proposal boxes [P,4], GT boxes [G,4]).

        Match each proposal to its nearest ground-truth box above ``min_iou``.
        Keep the tightest few proposals per object and give each represented
        object equal total weight. Minimize ``sum_i w_i ||x_i W - target_i||^2
        + ridge * ||W||^2`` with weights summing to one; the intercept also has
        a zero-residual prior.
        This closed-form fit requires no backbone gradients or optimizer steps.
        """
        device = torch.device(device)
        rows, targets, weights = [], [], []
        images = objects = represented = eligible = 0
        dimension = None
        for visual, proposals, truth in support:
            images += 1
            visual = visual.detach().float().to(device)
            if visual.ndim != 2 or visual.shape[1] == 0 or not bool(torch.isfinite(visual).all()):
                raise ValueError("Visual features must be a finite [P,D] tensor with D > 0")
            if dimension is not None and dimension != visual.shape[1]:
                raise ValueError("All support features must have the same dimension")
            dimension = visual.shape[1]
            proposals = self._proposals(proposals, device)
            truth = self._boxes(truth, device)
            if len(visual) != len(proposals):
                raise ValueError("Features and proposal boxes must have the same row count")
            objects += len(truth)
            if not len(truth) or not len(proposals):
                continue
            overlap, matched = box_iou(proposals, truth).max(1)
            candidates = (overlap >= self.config["min_iou"]).nonzero()[:, 0]
            eligible += len(candidates)
            for object_id in matched[candidates].unique():
                own = candidates[matched[candidates] == object_id]
                chosen = own[overlap[own].argsort(descending=True, stable=True)
                             [:self.config["max_proposals_per_object"]]]
                boxes = proposals[chosen]
                size = boxes[:, 2:] - boxes[:, :2]
                center = (boxes[:, 2:] + boxes[:, :2]) / 2
                target = truth[object_id]
                target_size = target[2:] - target[:2]
                target_center = (target[2:] + target[:2]) / 2
                offsets = torch.cat(((target_center - center) / size, (target_size / size).log()), dim=1)
                rows.append(self._design(visual[chosen]))
                targets.append(offsets)
                weights.append(size.new_full((len(chosen),), 1 / len(chosen)))
                represented += 1
        if not rows:
            raise ValueError("No matched boxed support proposals; provide boxes or lower min_iou")
        x, y, sample_weight = torch.cat(rows).double(), torch.cat(targets).double(), torch.cat(weights).double()
        sample_weight /= sample_weight.sum()
        gram = x.T @ (sample_weight[:, None] * x)
        gram.diagonal().add_(self.config["ridge"])
        rhs = x.T @ (sample_weight[:, None] * y)
        weight = torch.cholesky_solve(rhs, torch.linalg.cholesky(gram))
        prediction = x @ weight
        self.weight = weight.float().cpu()
        self.stats = dict(images=images, objects=objects, represented_objects=represented,
                          eligible_proposals=eligible, selected_proposals=len(x), dimension=dimension,
                          parameters=self.weight.numel(),
                          zero_offset_mse=float((sample_weight[:, None] * y.square()).sum() / 4),
                          fitted_offset_mse=float((sample_weight[:, None] * (prediction - y).square()).sum() / 4))
        return self

    @torch.inference_mode()
    def __call__(self, visual, boxes):
        """Correct xyxy boxes before NMS; scores and class choices are unchanged."""
        if self.weight is None or self.config["strength"] == 0:
            return boxes
        if visual.ndim != 2 or visual.shape[1] + 1 != self.weight.shape[0] or len(visual) != len(boxes):
            raise ValueError("Inference features/boxes do not match the fitted box refiner")
        offsets = self._design(visual) @ self.weight.to(visual.device)
        delta_center = offsets[:, :2].clamp(-self.config["max_center_shift"], self.config["max_center_shift"])
        delta_scale = offsets[:, 2:].clamp(-self.config["max_log_scale"], self.config["max_log_scale"])
        size = boxes[:, 2:] - boxes[:, :2]
        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        center = center + self.config["strength"] * delta_center * size
        size = size * (self.config["strength"] * delta_scale).exp()
        return torch.cat((center - size / 2, center + size / 2), dim=1)

    def state_dict(self):
        return dict(config=self.config.copy(), weight=self.weight, stats=self.stats)

    @classmethod
    def from_state_dict(cls, state):
        head = cls(**state["config"])
        head.weight, head.stats = state["weight"], state["stats"]
        return head

    @property
    def storage_bytes(self):
        return self.weight.numel() * self.weight.element_size() if self.weight is not None else 0
