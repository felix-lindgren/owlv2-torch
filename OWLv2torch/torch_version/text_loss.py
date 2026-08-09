"""Losses for supervised text-conditioned OWLv2 detection training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou, generalized_box_iou

from OWLv2torch.torch_version.loss import (
    _hard_negative_indices,
    _objectness_loss,
    box_cxcywh_to_xyxy,
    sigmoid_focal_loss,
)

CLASS_LOSS_CHOICES = ("focal", "vfl", "mal")

# DEIM's defaults: MAL leaves the background term unscaled, VFL scales it by alpha.
DEFAULT_VFL_ALPHA = 0.2


@torch.no_grad()
def hungarian_match_text(
    class_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    *,
    class_cost: float = 1.0,
    bbox_cost: float = 5.0,
    giou_cost: float = 2.0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Match patch predictions to targets using text class and box costs."""
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:  # pragma: no cover - scipy is a project dependency
        linear_sum_assignment = None

    _, _, num_classes = class_logits.shape
    matches = []
    for logits, boxes, target in zip(class_logits, pred_boxes, targets):
        target_boxes = target["boxes"].to(device=boxes.device, dtype=boxes.dtype)
        target_labels = target["labels"].to(device=logits.device, dtype=torch.long)
        if target_boxes.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=boxes.device)
            matches.append((empty, empty))
            continue
        if (target_labels < 0).any() or (target_labels >= num_classes).any():
            raise ValueError(
                f"Text-query labels must be in [0, {num_classes}); "
                f"got {target_labels.detach().cpu().tolist()}"
            )

        cost = boxes.new_zeros((boxes.shape[0], target_boxes.shape[0]), dtype=torch.float32)
        if class_cost:
            # Each target label indexes the corresponding text query.
            cost = cost - class_cost * logits.float().sigmoid()[:, target_labels]
        if bbox_cost:
            cost = cost + bbox_cost * torch.cdist(
                boxes.float(), target_boxes.float(), p=1
            )
        if giou_cost:
            giou = generalized_box_iou(
                box_cxcywh_to_xyxy(boxes.float()),
                box_cxcywh_to_xyxy(target_boxes.float()),
            )
            cost = cost + giou_cost * (1.0 - giou)

        cost_cpu = cost.cpu()
        if linear_sum_assignment is not None:
            pred_indices, target_indices = linear_sum_assignment(cost_cpu)
            pred_indices = torch.as_tensor(pred_indices, dtype=torch.long, device=boxes.device)
            target_indices = torch.as_tensor(target_indices, dtype=torch.long, device=boxes.device)
        else:
            pred_indices, target_indices = _greedy_assignment(cost_cpu, boxes.device)
        matches.append((pred_indices, target_indices))
    return matches


def _greedy_assignment(cost: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pred_indices = []
    target_indices = []
    work = cost.clone()
    for _ in range(min(work.shape)):
        flat_index = int(work.argmin().item())
        pred_index, target_index = divmod(flat_index, work.shape[1])
        pred_indices.append(pred_index)
        target_indices.append(target_index)
        work[pred_index, :] = torch.inf
        work[:, target_index] = torch.inf
    return (
        torch.tensor(pred_indices, dtype=torch.long, device=device),
        torch.tensor(target_indices, dtype=torch.long, device=device),
    )


def _focal_class_loss(
    logits: torch.Tensor,
    class_targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Focal loss summed over classes, averaged over predictions.

    Note the deliberate ``sum`` over the class dimension. A plain
    ``reduction="mean"`` also divides by ``num_classes``, which makes the
    classification term ``num_classes`` times smaller than the
    ``sum / num_boxes`` convention that ``lambda_cls``/``lambda_l1``/
    ``lambda_giou`` are inherited from (DETR, RT-DETR, D-FINE). With 46
    Fashionpedia classes that shrank the effective ``lambda_cls`` to ~1/46.
    """
    per_element = sigmoid_focal_loss(
        logits, class_targets, alpha=0.25, gamma=2.0, reduction="none"
    )
    if class_weights is not None:
        per_element = per_element * class_weights.to(
            device=per_element.device, dtype=per_element.dtype
        )
    return per_element.sum(dim=-1).mean()


def _quality_class_loss(
    class_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    matches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    variant: str,
    gamma: float,
    alpha: float | None,
    class_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """IoU-aware dense classification loss: MAL (DEIM) or VFL (VarifocalNet).

    Both replace the matched/mined focal split with a single BCE over every
    prediction, where matched positions regress towards the IoU of their own box
    instead of a hard 1.0, and unmatched positions are down-weighted by
    ``p**gamma`` the way focal loss does. Mining is therefore unnecessary: all
    ``num_patches * num_classes`` entries contribute.

    The two differ only in how a low-quality match is treated. VFL weights
    positives by ``q`` itself, so a badly localized match contributes almost no
    gradient; MAL gives every positive weight 1 and moves the quality into the
    target (``q**gamma``). Returns the positive/background split of the same
    tensor, normalized by the batch's matched-box count.
    """
    if variant not in ("mal", "vfl"):
        raise ValueError(f"variant must be 'mal' or 'vfl', got {variant!r}")
    if alpha is None and variant == "vfl":
        alpha = DEFAULT_VFL_ALPHA

    device = class_logits.device
    one_hot = torch.zeros_like(class_logits)
    quality = torch.zeros_like(class_logits)
    num_boxes = 0

    for batch_index, (pred_indices, target_indices) in enumerate(matches):
        if pred_indices.numel() == 0:
            continue
        labels = targets[batch_index]["labels"].to(device=device, dtype=torch.long)[
            target_indices
        ]
        matched_boxes = pred_boxes[batch_index][pred_indices]
        target_boxes = targets[batch_index]["boxes"].to(
            device=device, dtype=matched_boxes.dtype
        )[target_indices]
        with torch.no_grad():
            ious = box_iou(
                box_cxcywh_to_xyxy(matched_boxes.float()),
                box_cxcywh_to_xyxy(target_boxes.float()),
            ).diagonal()
        one_hot[batch_index, pred_indices, labels] = 1.0
        quality[batch_index, pred_indices, labels] = ious.to(quality.dtype)
        num_boxes += pred_indices.numel()

    num_boxes = max(num_boxes, 1)
    background = class_logits.detach().sigmoid().pow(gamma) * (1.0 - one_hot)
    if alpha is not None:
        background = alpha * background

    if variant == "mal":
        target_score = quality.pow(gamma)
        weight = background + one_hot
    else:
        target_score = quality
        weight = background + quality

    loss = F.binary_cross_entropy_with_logits(
        class_logits, target_score, weight=weight, reduction="none"
    )
    if class_weights is not None:
        loss = loss * class_weights.to(device=loss.device, dtype=loss.dtype).view(1, 1, -1)
    return (loss * one_hot).sum() / num_boxes, (
        loss * (1.0 - one_hot)
    ).sum() / num_boxes


def compute_text_query_losses(
    outputs,
    targets: list[dict[str, torch.Tensor]],
    *,
    lambda_cls: float = 1.0,
    lambda_l1: float = 5.0,
    lambda_giou: float = 2.0,
    lambda_objectness: float = 0.5,
    match_class_cost: float = 1.0,
    match_bbox_cost: float = 5.0,
    match_giou_cost: float = 2.0,
    include_geometry_loss: bool = True,
    include_objectness_loss: bool = True,
    negative_ratio: int = 5,
    max_negatives_per_image: int = 512,
    negative_loss_weight: float = 1.0,
    class_loss: str = "focal",
    class_loss_gamma: float = 1.5,
    class_loss_alpha: float | None = None,
    class_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute detection losses where each class index denotes a text query.

    ``outputs`` is the normal return value of
    :meth:`OwlV2.forward_object_detection`: class logits, objectness logits,
    predicted boxes, class embeddings and extras.

    ``class_loss`` selects the classification term:

    ``focal``
        The original matched-focal plus hard-negative-mined-focal split, honouring
        ``negative_ratio``/``max_negatives_per_image``/``negative_loss_weight``.
    ``mal``/``vfl``
        DEIM's Matchability-Aware Loss / VarifocalNet's Varifocal Loss. Both are
        dense over every patch, so negative mining does not apply to them (it
        still governs the objectness term). See :func:`_quality_class_loss`.

    The box and objectness terms are identical across all three so the variants
    are directly comparable.
    """
    if class_loss not in CLASS_LOSS_CHOICES:
        raise ValueError(
            f"class_loss must be one of {CLASS_LOSS_CHOICES}, got {class_loss!r}"
        )
    class_logits, objectness_logits, pred_boxes = outputs[:3]
    if class_weights is not None:
        if class_weights.ndim != 1 or class_weights.numel() != class_logits.shape[-1]:
            raise ValueError(
                "class_weights must have one entry per text query; got "
                f"{tuple(class_weights.shape)} for {class_logits.shape[-1]} queries"
            )
        if (class_weights < 0).any():
            raise ValueError("class_weights must be non-negative")
    if len(targets) != class_logits.shape[0]:
        raise ValueError(
            f"Received {len(targets)} targets for a batch of {class_logits.shape[0]}"
        )
    matches = hungarian_match_text(
        class_logits,
        pred_boxes,
        targets,
        class_cost=match_class_cost,
        bbox_cost=match_bbox_cost,
        giou_cost=match_giou_cost,
    )

    positive_class_losses = []
    negative_class_losses = []
    l1_losses = []
    giou_losses = []
    objectness_losses = []

    for batch_index, (pred_indices, target_indices) in enumerate(matches):
        logits = class_logits[batch_index]
        objectness = objectness_logits[batch_index]
        num_positives = pred_indices.numel()
        negative_indices = _hard_negative_indices(
            logits,
            objectness,
            pred_indices,
            num_positives=num_positives,
            negative_ratio=negative_ratio,
            max_negatives=max_negatives_per_image,
        )

        if num_positives:
            if class_loss == "focal":
                target_labels = targets[batch_index]["labels"].to(
                    device=logits.device, dtype=torch.long
                )[target_indices]
                positive_targets = torch.zeros_like(logits[pred_indices])
                positive_targets.scatter_(1, target_labels[:, None], 1.0)
                positive_class_losses.append(
                    _focal_class_loss(logits[pred_indices], positive_targets, class_weights)
                )

            matched_boxes = pred_boxes[batch_index][pred_indices]
            target_boxes = targets[batch_index]["boxes"].to(
                device=matched_boxes.device, dtype=matched_boxes.dtype
            )[target_indices]
            l1_losses.append(F.l1_loss(matched_boxes, target_boxes, reduction="mean"))
            giou = generalized_box_iou(
                box_cxcywh_to_xyxy(matched_boxes.float()),
                box_cxcywh_to_xyxy(target_boxes.float()),
            )
            giou_losses.append(1.0 - giou.diag().mean())

        if class_loss == "focal" and negative_indices.numel():
            negative_class_losses.append(
                _focal_class_loss(
                    logits[negative_indices],
                    torch.zeros_like(logits[negative_indices]),
                    class_weights,
                )
            )

        if include_objectness_loss:
            objectness_loss = _objectness_loss(objectness, pred_indices, negative_indices)
            if objectness_loss is not None:
                objectness_losses.append(objectness_loss)

    zero = pred_boxes.new_zeros(())
    if class_loss == "focal":
        loss_cls_pos = _mean_or_zero(positive_class_losses, zero)
        loss_cls_neg = _mean_or_zero(negative_class_losses, zero)
        loss_cls = loss_cls_pos + negative_loss_weight * loss_cls_neg
    else:
        # MAL/VFL are dense: every patch contributes, so there is nothing to mine
        # and nothing to rebalance. The split is reported for comparability only.
        loss_cls_pos, loss_cls_neg = _quality_class_loss(
            class_logits,
            pred_boxes,
            targets,
            matches,
            variant=class_loss,
            gamma=class_loss_gamma,
            alpha=class_loss_alpha,
            class_weights=class_weights,
        )
        loss_cls = loss_cls_pos + loss_cls_neg
    loss_l1 = _mean_or_zero(l1_losses, zero)
    loss_giou = _mean_or_zero(giou_losses, zero)
    loss_objectness = _mean_or_zero(objectness_losses, zero)

    loss = lambda_cls * loss_cls
    if include_geometry_loss:
        loss = loss + lambda_l1 * loss_l1 + lambda_giou * loss_giou
    if include_objectness_loss:
        loss = loss + lambda_objectness * loss_objectness

    return {
        "loss": loss,
        "L_cls": loss_cls,
        "L_cls_pos": loss_cls_pos,
        "L_cls_neg": loss_cls_neg,
        "L_l1": loss_l1,
        "L_giou": loss_giou,
        "L_obj": loss_objectness,
    }


def _mean_or_zero(values: list[torch.Tensor], zero: torch.Tensor) -> torch.Tensor:
    return torch.stack(values).mean() if values else zero
