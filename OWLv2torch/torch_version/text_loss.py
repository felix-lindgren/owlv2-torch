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
from OWLv2torch.torch_version.point_metrics import pairwise_oks

CLASS_LOSS_CHOICES = ("focal", "vfl", "mal")

# DEIM's defaults: MAL leaves the background term unscaled, VFL scales it by alpha.
DEFAULT_VFL_ALPHA = 0.2


def split_ignore_targets(
    targets: list[dict[str, torch.Tensor]],
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
    """Separate pseudo-label ignore boxes from the real targets.

    Ignore boxes ride through the dataset and batch augmentations as ordinary
    targets whose label is encoded as ``-1 - class``, so every geometric
    transform moves them exactly like the real boxes. This undoes the encoding
    and returns ``(targets, ignore_targets)``, both with non-negative labels.
    """
    real, ignore = [], []
    for target in targets:
        labels = target["labels"]
        is_ignore = labels < 0
        real.append({"boxes": target["boxes"][~is_ignore], "labels": labels[~is_ignore]})
        ignore.append({"boxes": target["boxes"][is_ignore], "labels": -1 - labels[is_ignore]})
    return real, ignore


@torch.no_grad()
def _ignored_patches(
    pred_boxes: torch.Tensor,
    ignore_targets: list[dict[str, torch.Tensor]],
    iou_threshold: float,
    point_sigma: float | None = None,
) -> torch.Tensor:
    """``[B, num_patches]`` mask of patches whose box overlaps an ignore box.

    With ``point_sigma`` the overlap is the OKS of the patch's predicted centre.

    These are the patches that would otherwise be pushed towards background for
    predicting an object the teacher was unsure about. Class-agnostic on purpose:
    a teacher score in the uncertain band says little about the class either.
    """
    ignored = torch.zeros(pred_boxes.shape[:2], dtype=torch.bool, device=pred_boxes.device)
    for batch_index, target in enumerate(ignore_targets):
        boxes = target["boxes"]
        if boxes.numel() == 0:
            continue
        boxes = boxes.to(device=pred_boxes.device, dtype=torch.float32)
        if point_sigma is not None:
            iou = pairwise_oks(pred_boxes[batch_index, :, :2], boxes, point_sigma)
        else:
            iou = box_iou(box_cxcywh_to_xyxy(pred_boxes[batch_index].float()), box_cxcywh_to_xyxy(boxes))
        ignored[batch_index] = iou.amax(dim=1) >= iou_threshold
    return ignored


@torch.no_grad()
def hungarian_match_text(
    class_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    *,
    class_cost: float = 1.0,
    bbox_cost: float = 5.0,
    giou_cost: float = 2.0,
    point_sigma: float | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Match patch predictions to targets using text class and box costs.

    With ``point_sigma`` only centres are compared: ``bbox_cost`` weighs the L1
    centre distance and ``giou_cost`` weighs ``1 - OKS`` in place of ``1 - GIoU``.

    On CUDA all images are solved in one launch of the Triton solver in
    :mod:`OWLv2torch.torch_version.linear_assignment`, which reproduces SciPy's
    optimum without a host round trip per image; elsewhere SciPy solves each image.
    """
    _, num_patches, num_classes = class_logits.shape
    labels = [target["labels"] for target in targets if target["labels"].numel()]
    if labels:
        all_labels = torch.cat([label.to(class_logits.device) for label in labels])
        if ((all_labels < 0) | (all_labels >= num_classes)).any():
            raise ValueError(
                f"Text-query labels must be in [0, {num_classes}); "
                f"got {all_labels.detach().cpu().tolist()}"
            )

    costs: list[torch.Tensor | None] = []
    for logits, boxes, target in zip(class_logits, pred_boxes, targets):
        target_boxes = target["boxes"].to(device=boxes.device, dtype=boxes.dtype)
        target_labels = target["labels"].to(device=logits.device, dtype=torch.long)
        if target_boxes.numel() == 0:
            costs.append(None)
            continue

        cost = boxes.new_zeros((boxes.shape[0], target_boxes.shape[0]), dtype=torch.float32)
        if class_cost:
            # Each target label indexes the corresponding text query.
            cost = cost - class_cost * logits.float().sigmoid()[:, target_labels]
        if point_sigma is not None:
            if bbox_cost:
                cost = cost + bbox_cost * torch.cdist(
                    boxes[:, :2].float(), target_boxes[:, :2].float(), p=1
                )
            if giou_cost:
                cost = cost + giou_cost * (1.0 - pairwise_oks(boxes[:, :2], target_boxes, point_sigma))
        else:
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
        costs.append(cost)

    max_targets = max((cost.shape[1] for cost in costs if cost is not None), default=0)
    if class_logits.is_cuda and 0 < max_targets <= num_patches:
        return _match_on_gpu(costs, num_patches, max_targets)
    return [_match_on_cpu(cost, pred_boxes.device) for cost in costs]


def _match_on_cpu(cost: torch.Tensor | None, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve one ``[num_patches, num_targets]`` cost with SciPy (greedy without it)."""
    if cost is None:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:  # pragma: no cover - scipy is a project dependency
        return _greedy_assignment(cost.cpu(), device)
    pred_indices, target_indices = linear_sum_assignment(cost.cpu())
    return (
        torch.as_tensor(pred_indices, dtype=torch.long, device=device),
        torch.as_tensor(target_indices, dtype=torch.long, device=device),
    )


def _match_on_gpu(
    costs: list[torch.Tensor | None], num_patches: int, max_targets: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Solve every image's cost in one Triton launch, in SciPy's output layout.

    Target rows are padded to a power of two with zero cost, which cannot change
    the real rows' optimum (see :func:`batched_match_cost`); the row count is a
    compile-time constant of the kernel, so this bounds the number of recompiles.
    The only host sync is the check for rows the solver left unassigned, and those
    images are re-solved with SciPy.
    """
    from OWLv2torch.torch_version.linear_assignment import batch_linear_assignment

    reference = next(cost for cost in costs if cost is not None)
    rows = min(1 << (max_targets - 1).bit_length(), num_patches)
    batch = reference.new_zeros((len(costs), rows, num_patches))
    for index, cost in enumerate(costs):
        if cost is not None:
            batch[index, : cost.shape[1]] = cost.T
    assigned = batch_linear_assignment(batch)
    sizes = torch.tensor(
        [0 if cost is None else cost.shape[1] for cost in costs], device=assigned.device
    )
    real_rows = torch.arange(rows, device=assigned.device)[None] < sizes[:, None]
    failed = ((assigned < 0) & real_rows).any(1).tolist()

    matches = []
    for index, cost in enumerate(costs):
        if cost is None or failed[index]:
            matches.append(_match_on_cpu(cost, reference.device))
            continue
        # SciPy returns the pairs sorted by patch index.
        patches = assigned[index, : cost.shape[1]]
        target_indices = patches.argsort()
        matches.append((patches[target_indices], target_indices))
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
    ignored: torch.Tensor | None = None,
    no_negative_classes: torch.Tensor | None = None,
    point_sigma: float | None = None,
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

    ``ignored`` (``[B, num_patches]``) drops the background term of unmatched
    patches that overlap a pseudo-label ignore box, and ``no_negative_classes``
    (``[B, num_classes]``) drops it for whole classes of an image.

    With ``point_sigma`` the quality is the matched centre's OKS instead of IoU.
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
            if point_sigma is not None:
                ious = pairwise_oks(matched_boxes[:, :2], target_boxes, point_sigma).diagonal()
            else:
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
    if ignored is not None:
        # Positives win: a matched patch keeps its other classes as background.
        ignored = ignored & (one_hot.amax(-1) == 0)
        background = background * (~ignored).unsqueeze(-1).to(background.dtype)
    if no_negative_classes is not None:
        background = background * (~no_negative_classes).unsqueeze(1).to(background.dtype)

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
    ignore_targets: list[dict[str, torch.Tensor]] | None = None,
    ignore_iou_threshold: float = 0.5,
    no_negative_classes: torch.Tensor | None = None,
    point_sigma: float | None = None,
    point_size_weight: float = 0.0,
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

    ``ignore_targets`` (from :func:`split_ignore_targets`) are pseudo-labels the
    teacher was unsure about. They are never matched; an unmatched patch whose box
    reaches ``ignore_iou_threshold`` IoU with one gets no negative classification
    or objectness loss, instead of being trained as background.

    ``no_negative_classes`` (``[B, num_classes]`` bool) removes the background
    classification term of those classes in those images, for pseudo-labelled
    images whose teacher misses most instances of a class: the unlabelled ones
    would otherwise be trained as background. Matched positives still count, and
    objectness is untouched since it is class-agnostic.

    ``point_sigma`` trains a point (box-centre) detector. Ground-truth boxes only
    set the tolerance: matching, the MAL/VFL quality target and the ignore mask
    use OKS (:func:`pairwise_oks`) instead of IoU, ``lambda_l1`` weighs the L1
    on the centre alone and ``lambda_giou`` weighs ``1 - OKS``, reported as
    ``L_oks``. ``point_size_weight`` adds an L1 on the matched width and height
    (``L_wh``): without it they get no gradient and drift as the shared box-head
    layers train, and the unmatched patches' centres drift with them.
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
        point_sigma=point_sigma,
    )
    ignored = (
        _ignored_patches(pred_boxes, ignore_targets, ignore_iou_threshold, point_sigma)
        if ignore_targets is not None
        else None
    )

    positive_class_losses = []
    negative_class_losses = []
    l1_losses = []
    giou_losses = []
    size_losses = []
    objectness_losses = []

    for batch_index, (pred_indices, target_indices) in enumerate(matches):
        logits = class_logits[batch_index]
        objectness = objectness_logits[batch_index]
        num_positives = pred_indices.numel()
        excluded = pred_indices
        if ignored is not None:
            # Mining skips ignored patches, so they leave both negative terms.
            excluded = torch.cat([pred_indices, ignored[batch_index].nonzero().squeeze(1)])
        negative_indices = _hard_negative_indices(
            logits,
            objectness,
            excluded,
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
            if point_sigma is not None:
                l1_losses.append(
                    F.l1_loss(matched_boxes[:, :2], target_boxes[:, :2], reduction="mean")
                )
                oks = pairwise_oks(matched_boxes[:, :2], target_boxes, point_sigma)
                giou_losses.append(1.0 - oks.diag().mean())
                size_losses.append(
                    F.l1_loss(matched_boxes[:, 2:], target_boxes[:, 2:], reduction="mean")
                )
            else:
                l1_losses.append(F.l1_loss(matched_boxes, target_boxes, reduction="mean"))
                giou = generalized_box_iou(
                    box_cxcywh_to_xyxy(matched_boxes.float()),
                    box_cxcywh_to_xyxy(target_boxes.float()),
                )
                giou_losses.append(1.0 - giou.diag().mean())

        if class_loss == "focal" and negative_indices.numel():
            negative_weights = class_weights
            if no_negative_classes is not None:
                keep = (~no_negative_classes[batch_index]).to(logits.dtype)
                negative_weights = keep if class_weights is None else class_weights.to(
                    keep.device, keep.dtype
                ) * keep
            negative_class_losses.append(
                _focal_class_loss(
                    logits[negative_indices],
                    torch.zeros_like(logits[negative_indices]),
                    negative_weights,
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
            ignored=ignored,
            no_negative_classes=no_negative_classes,
            point_sigma=point_sigma,
        )
        loss_cls = loss_cls_pos + loss_cls_neg
    loss_l1 = _mean_or_zero(l1_losses, zero)
    loss_giou = _mean_or_zero(giou_losses, zero)
    loss_objectness = _mean_or_zero(objectness_losses, zero)

    loss_size = _mean_or_zero(size_losses, zero)

    loss = lambda_cls * loss_cls
    if include_geometry_loss:
        loss = loss + lambda_l1 * loss_l1 + lambda_giou * loss_giou + point_size_weight * loss_size
    if include_objectness_loss:
        loss = loss + lambda_objectness * loss_objectness

    result = {
        "loss": loss,
        "L_cls": loss_cls,
        "L_cls_pos": loss_cls_pos,
        "L_cls_neg": loss_cls_neg,
        "L_l1": loss_l1,
        "L_oks" if point_sigma is not None else "L_giou": loss_giou,
        "L_obj": loss_objectness,
    }
    if point_sigma is not None:
        result["L_wh"] = loss_size
    if ignored is not None:
        result["ignored_patches"] = ignored.float().sum(1).mean()
    return result


def _mean_or_zero(values: list[torch.Tensor], zero: torch.Tensor) -> torch.Tensor:
    return torch.stack(values).mean() if values else zero


DEFAULT_MAX_TARGETS = 128


def pad_text_targets(
    targets: list[dict[str, torch.Tensor]],
    max_targets: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad per-image targets to fixed ``[B, max_targets]`` slots for the batched loss.

    Returns normalised cxcywh boxes, labels, and a validity mask. Host targets are
    padded on the CPU and copied once; device targets (e.g. after mosaic) are padded
    in place on the device. Neither path syncs. Padded boxes get a small non-zero
    size so their GIoU stays finite: a NaN in a masked slot would still poison the
    masked sums.
    """
    counts = [target["boxes"].shape[0] for target in targets]
    if max(counts, default=0) > max_targets:
        raise ValueError(
            f"An image has {max(counts)} boxes, more than max_targets={max_targets}; "
            "raise --max-targets"
        )
    on_host = all(target["boxes"].device.type == "cpu" for target in targets)
    build_device = torch.device("cpu") if on_host else device
    batch_size = len(targets)
    boxes = torch.zeros((batch_size, max_targets, 4), dtype=torch.float32, device=build_device)
    boxes[..., 2:] = 1e-3
    labels = torch.zeros((batch_size, max_targets), dtype=torch.long, device=build_device)
    valid = torch.zeros((batch_size, max_targets), dtype=torch.bool, device=build_device)
    for index, (target, count) in enumerate(zip(targets, counts)):
        boxes[index, :count] = target["boxes"].to(device=build_device, dtype=torch.float32)
        labels[index, :count] = target["labels"].to(device=build_device, dtype=torch.long)
        valid[index, :count] = True
    if on_host:
        boxes, labels, valid = (
            tensor.pin_memory().to(device, non_blocking=True)
            for tensor in (boxes, labels, valid)
        )
    return boxes, labels, valid


def _pairwise_iou_giou(
    boxes_a: torch.Tensor, boxes_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcasting IoU and GIoU of xyxy boxes (torchvision's ops are 2-D only)."""
    top_left = torch.maximum(boxes_a[..., :2], boxes_b[..., :2])
    bottom_right = torch.minimum(boxes_a[..., 2:], boxes_b[..., 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(-1)
    area_a = (boxes_a[..., 2:] - boxes_a[..., :2]).prod(-1)
    area_b = (boxes_b[..., 2:] - boxes_b[..., :2]).prod(-1)
    union = area_a + area_b - intersection
    iou = intersection / union
    hull = (
        torch.maximum(boxes_a[..., 2:], boxes_b[..., 2:])
        - torch.minimum(boxes_a[..., :2], boxes_b[..., :2])
    ).prod(-1)
    return iou, iou - (hull - union) / hull


@torch.no_grad()
def batched_match_cost(
    class_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    class_cost: float = 1.0,
    bbox_cost: float = 5.0,
    giou_cost: float = 2.0,
) -> torch.Tensor:
    """:func:`hungarian_match_text`'s cost, laid out ``[B, max_targets, num_patches]``.

    Padded target rows get a constant zero cost. A constant row cannot change the
    optimal assignment of the real rows while there are at least as many patches as
    target slots, so padding only adds trivially solved rows.
    """
    batch_size, num_patches, _ = class_logits.shape
    probabilities = class_logits.float().sigmoid()
    label_index = target_labels[:, None, :].expand(batch_size, num_patches, -1)
    class_probability = probabilities.gather(2, label_index).transpose(1, 2)
    boxes = pred_boxes.float()
    _, giou = _pairwise_iou_giou(
        box_cxcywh_to_xyxy(target_boxes)[:, :, None],
        box_cxcywh_to_xyxy(boxes)[:, None],
    )
    cost = (
        -class_cost * class_probability
        + bbox_cost * torch.cdist(target_boxes, boxes, p=1)
        + giou_cost * (1.0 - giou)
    )
    return cost.masked_fill(~target_valid[..., None], 0.0)


def batched_text_query_losses(
    outputs,
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    lambda_cls: float = 1.0,
    lambda_l1: float = 5.0,
    lambda_giou: float = 2.0,
    lambda_objectness: float = 0.5,
    match_class_cost: float = 1.0,
    match_bbox_cost: float = 5.0,
    match_giou_cost: float = 2.0,
    negative_ratio: int = 5,
    max_negatives_per_image: int = 512,
    class_loss: str = "mal",
    class_loss_gamma: float = 1.5,
    class_loss_alpha: float | None = None,
    class_weights: torch.Tensor | None = None,
    ignore_boxes: torch.Tensor | None = None,
    ignore_valid: torch.Tensor | None = None,
    ignore_iou_threshold: float = 0.5,
    no_negative_classes: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """:func:`compute_text_query_losses` for ``mal``/``vfl`` without host syncs.

    Targets come from :func:`pad_text_targets`, so every shape is static and the
    whole loss, Hungarian matching included, stays on the GPU: the CPU can queue the
    backward while the forward is still running, and the loss can be compiled into
    one graph with the model. Matching uses the Triton solver in
    :mod:`OWLv2torch.torch_version.linear_assignment`.

    The terms match the per-image loss to fp32 rounding (~1e-7 on 8 LV-MHP
    batches, gradients included). The extra ``matching_failures`` entry counts target
    boxes the solver left unassigned; those boxes are dropped from the step rather
    than matched to an arbitrary patch, and it should stay 0.

    ``ignore_boxes``/``ignore_valid`` are pseudo-label ignore boxes padded by
    :func:`pad_text_targets`, with the same semantics as ``ignore_targets`` in the
    per-image loss; ``no_negative_classes`` is shared with it unchanged.
    """
    from OWLv2torch.torch_version.linear_assignment import batch_linear_assignment

    if class_loss not in ("mal", "vfl"):
        raise ValueError(f"The batched loss supports class_loss mal/vfl, got {class_loss!r}")
    if class_loss_alpha is None and class_loss == "vfl":
        class_loss_alpha = DEFAULT_VFL_ALPHA
    class_logits, objectness_logits, pred_boxes = outputs[:3]
    batch_size, num_patches, num_classes = class_logits.shape

    assigned_patch = batch_linear_assignment(
        batched_match_cost(
            class_logits,
            pred_boxes,
            target_boxes,
            target_labels,
            target_valid,
            class_cost=match_class_cost,
            bbox_cost=match_bbox_cost,
            giou_cost=match_giou_cost,
        )
    )
    matched = target_valid & (assigned_patch >= 0)
    assigned_patch = assigned_patch.clamp(min=0)
    matched_weight = matched.float()
    num_matched = matched_weight.sum(1)
    images_with_boxes = (num_matched > 0).float()
    num_images_with_boxes = images_with_boxes.sum().clamp(min=1)
    per_image_count = num_matched.clamp(min=1)

    # Geometry: mean over each image's matches, then over images that have any,
    # exactly as the per-image loss averages its list of per-image terms.
    matched_boxes = pred_boxes.gather(1, assigned_patch[..., None].expand(-1, -1, 4))
    l1 = (matched_boxes - target_boxes.to(matched_boxes.dtype)).abs().sum(-1)
    loss_l1 = (
        (l1 * matched_weight).sum(1) / (4 * per_image_count) * images_with_boxes
    ).sum() / num_images_with_boxes
    iou, giou = _pairwise_iou_giou(
        box_cxcywh_to_xyxy(matched_boxes.float()), box_cxcywh_to_xyxy(target_boxes)
    )
    loss_giou = (
        (1.0 - (giou * matched_weight).sum(1) / per_image_count) * images_with_boxes
    ).sum() / num_images_with_boxes

    # Dense MAL/VFL as in _quality_class_loss. Assigned patches are distinct within
    # an image, so scatter_add never adds two real boxes to one entry; unmatched and
    # padded slots add 0.
    flat_index = assigned_patch * num_classes + target_labels
    one_hot = class_logits.new_zeros((batch_size, num_patches * num_classes))
    quality = torch.zeros_like(one_hot)
    one_hot.scatter_add_(1, flat_index, matched_weight.to(one_hot.dtype))
    quality.scatter_add_(1, flat_index, (iou.detach() * matched_weight).to(quality.dtype))
    one_hot = one_hot.view_as(class_logits)
    quality = quality.view_as(class_logits)
    num_boxes = num_matched.sum().clamp(min=1)
    background = class_logits.detach().sigmoid().pow(class_loss_gamma) * (1.0 - one_hot)
    if class_loss_alpha is not None:
        background = class_loss_alpha * background
    positive_patch = one_hot.amax(-1) > 0
    ignored = None
    if ignore_boxes is not None:
        with torch.no_grad():
            ignore_iou, _ = _pairwise_iou_giou(
                box_cxcywh_to_xyxy(pred_boxes.float())[:, :, None],
                box_cxcywh_to_xyxy(ignore_boxes)[:, None],
            )
            ignored = ((ignore_iou >= ignore_iou_threshold) & ignore_valid[:, None]).any(-1)
            ignored = ignored & ~positive_patch
        background = background * (~ignored).unsqueeze(-1).to(background.dtype)
    if no_negative_classes is not None:
        background = background * (~no_negative_classes).unsqueeze(1).to(background.dtype)
    if class_loss == "mal":
        target_score = quality.pow(class_loss_gamma)
        weight = background + one_hot
    else:
        target_score = quality
        weight = background + quality
    class_bce = F.binary_cross_entropy_with_logits(
        class_logits, target_score, weight=weight, reduction="none"
    )
    if class_weights is not None:
        class_bce = class_bce * class_weights.to(
            device=class_bce.device, dtype=class_bce.dtype
        ).view(1, 1, -1)
    loss_cls_pos = (class_bce * one_hot).sum() / num_boxes
    loss_cls_neg = (class_bce * (1.0 - one_hot)).sum() / num_boxes
    loss_cls = loss_cls_pos + loss_cls_neg

    # Objectness: matched patches plus the top-`limit` unmatched patches by
    # class x objectness score, as _hard_negative_indices picks them. A fixed-k topk
    # with a per-image rank mask replaces the data-dependent nonzero/topk.
    with torch.no_grad():
        excluded_patch = positive_patch if ignored is None else positive_patch | ignored
        scores = (
            class_logits.float().sigmoid().amax(-1) * objectness_logits.float().sigmoid()
        ).masked_fill(excluded_patch, -1.0)
        negative_patch = torch.topk(scores, k=max_negatives_per_image, dim=1).indices
        limit = torch.where(
            num_matched > 0,
            (num_matched * negative_ratio).clamp(max=max_negatives_per_image),
            torch.full_like(num_matched, float(max_negatives_per_image)),
        )
        negative_weight = (
            torch.arange(max_negatives_per_image, device=limit.device)[None] < limit[:, None]
        ).float()
        if ignored is not None:
            # Only reachable when fewer patches than `limit` are left unexcluded.
            negative_weight = negative_weight * (~ignored.gather(1, negative_patch)).float()
    objectness = objectness_logits.float()
    positive_bce = F.binary_cross_entropy_with_logits(
        objectness.gather(1, assigned_patch),
        torch.ones_like(matched_weight),
        reduction="none",
    )
    negative_bce = F.binary_cross_entropy_with_logits(
        objectness.gather(1, negative_patch),
        torch.zeros_like(negative_weight),
        reduction="none",
    )
    loss_objectness = (
        ((positive_bce * matched_weight).sum(1) + (negative_bce * negative_weight).sum(1))
        / (num_matched + limit)
    ).mean()

    loss = (
        lambda_cls * loss_cls
        + lambda_l1 * loss_l1
        + lambda_giou * loss_giou
        + lambda_objectness * loss_objectness
    )
    result = {
        "loss": loss,
        "L_cls": loss_cls,
        "L_cls_pos": loss_cls_pos,
        "L_cls_neg": loss_cls_neg,
        "L_l1": loss_l1,
        "L_giou": loss_giou,
        "L_obj": loss_objectness,
        "matching_failures": (target_valid & ~matched).sum().float(),
    }
    if ignored is not None:
        result["ignored_patches"] = ignored.float().sum(1).mean()
    return result
