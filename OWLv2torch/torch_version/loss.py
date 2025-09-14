import torch
import math
from typing import List, Dict
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou
from OWLv2torch.torch_version.prototypes import VisualPrototypeBank

def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2.0, reduction="mean"):
    """
    inputs: logits, targets: {0,1}
    """
    prob = torch.sigmoid(inputs)
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

def normalized_wasserstein_distance(pred_boxes, target_boxes, C, reduction='mean', use_uniform_var=False, eps=1e-9):
    """
    pred_boxes, target_boxes: [N,4] in cxcywh (same units as C; typically pixels)
    C: dataset-level constant (e.g., mean object side length in pixels)
    use_uniform_var: if True, use 1/12 instead of 1/4 for the size term
    """
    # Δ terms (no need to convert to xyxy)
    dcx = pred_boxes[:, 0] - target_boxes[:, 0]
    dcy = pred_boxes[:, 1] - target_boxes[:, 1]
    dw  = pred_boxes[:, 2] - target_boxes[:, 2]
    dh  = pred_boxes[:, 3] - target_boxes[:, 3]

    k = (1.0/12.0) if use_uniform_var else (1.0/4.0)
    W2_sq = dcx*dcx + dcy*dcy + k*(dw*dw + dh*dh)

    # exponential normalization to [0,1]
    C2 = (C*C) + eps
    sim = torch.exp(- W2_sq / (2.0*C2))

    loss = 1.0 - sim
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

def box_cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def _nwd_W2_sq(a, b, k=0.25):
    # a,b: [Q,4] and [M,4] in cxcywh (normalized 0..1)
    # returns [Q,M] of W2^2 = dcenter^2 + k*(dw^2+dh^2)
    dcx = a[:, 0][:, None] - b[:, 0][None, :]
    dcy = a[:, 1][:, None] - b[:, 1][None, :]
    dw  = a[:, 2][:, None] - b[:, 2][None, :]
    dh  = a[:, 3][:, None] - b[:, 3][None, :]
    return dcx*dcx + dcy*dcy + k*(dw*dw + dh*dh)

def hungarian_match(class_logits, pred_boxes, targets,
                    lambda_l1=0.0, lambda_giou=0.0,
                    lambda_nwd=1.0, k_size_term=0.25):
    """
    One-to-one Hungarian, geometry-driven.
    class_logits unused by default (since you're only training the class embedding).
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:
        linear_sum_assignment = None

    out = []
    B, Q, _ = class_logits.shape
    for b in range(B):
        boxes = pred_boxes[b]                      # [Q,4] cxcywh in [0,1]
        tgt = targets[b]
        if len(tgt["boxes"]) == 0:
            out.append((torch.empty(0, dtype=torch.long),
                        torch.empty(0, dtype=torch.long)))
            continue

        tboxes = tgt["boxes"].to(boxes.device)     # [M,4]
        M = tboxes.shape[0]

        with torch.no_grad():
            # Geometry costs
            cost = 0.0
            if lambda_nwd > 0:
                nwd = _nwd_W2_sq(boxes, tboxes, k=k_size_term)  # [Q,M]
                cost = cost + lambda_nwd * nwd

            if lambda_l1 > 0:
                cost = cost + lambda_l1 * torch.cdist(boxes, tboxes, p=1)  # [Q,M]

            if lambda_giou > 0:
                giou = generalized_box_iou(
                    box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(tboxes)
                )  # [Q,M]
                cost = cost + lambda_giou * (1.0 - giou)

            cost = cost.cpu()

            if linear_sum_assignment is not None:
                row, col = linear_sum_assignment(cost)
                idx_pred = torch.as_tensor(row, dtype=torch.long)
                idx_gt   = torch.as_tensor(col, dtype=torch.long)
            else:
                # simple greedy fallback
                QN, MN = cost.shape
                idx_pred, idx_gt = [], []
                used_p, used_g = set(), set()
                for _ in range(min(QN, MN)):
                    qg = cost.argmin().item()
                    q, g = divmod(qg, MN)
                    while q in used_p or g in used_g:
                        cost[q, g] = float('inf')
                        qg = cost.argmin().item()
                        q, g = divmod(qg, MN)
                    idx_pred.append(q); idx_gt.append(g)
                    cost[q, :] = float('inf')
                    cost[:, g] = float('inf')
                    used_p.add(q); used_g.add(g)
                idx_pred = torch.tensor(idx_pred, dtype=torch.long)
                idx_gt   = torch.tensor(idx_gt, dtype=torch.long)

        out.append((idx_pred.to(boxes.device), idx_gt.to(boxes.device)))
    return out

def build_proto_targets(proto_bank: VisualPrototypeBank, labels: torch.Tensor) -> torch.Tensor:
    """
    Given ground-truth class labels [M], return a multi-hot vector over prototypes [P=C*K]
    where *all K prototypes of that class are positives*. Used for classification loss.
    """
    C, K = proto_bank.num_classes, proto_bank.K
    P = C * K
    Y = torch.zeros((labels.shape[0], P), device=labels.device)
    for i, c in enumerate(labels.tolist()):
        start = c * K
        Y[i, start:start+K] = 1.0
    return Y  # [M, P]



def compute_losses(outputs, targets, bank: VisualPrototypeBank,
                   lambda_cls=1.0, lambda_l1=5.0, lambda_giou=2.0,
                   use_objectness=True, use_nwd=True,):
    proto_logits, class_logits, objectness_logits, pred_boxes, _ = outputs
    B, Q, C = class_logits.shape
    losses = {}

    # Hungarian
    indices = hungarian_match(class_logits, pred_boxes, targets)

    # Classification (prototype-level, focal BCE)
    cls_losses = []
    l1_losses = []
    giou_losses = []
    obj_losses = []

    for b in range(B):
        idx_q, idx_g = indices[b]
        if len(idx_q) == 0:
            if use_objectness:
                obj_target = torch.zeros_like(objectness_logits[b])
                obj_losses.append(F.binary_cross_entropy_with_logits(objectness_logits[b], obj_target))
            continue

        # Gather matched predictions
        pb = pred_boxes[b][idx_q]          # [M,4] cxcywh
        cb = class_logits[b][idx_q]        # [M,C]
        pl: Unknown = proto_logits[b][idx_q]        # [M, C*K]
        ob = objectness_logits[b]

        gt_boxes = targets[b]["boxes"].to(pb.device)      # [M,4]
        gt_labels = targets[b]["labels"].long().to(cb.device)  # [M]

        # Prototype multi-label targets for classification
        Y_proto = build_proto_targets(bank, gt_labels)    # [M, C*K]

        # Focal BCE over prototypes (positives are *all K* for the GT class)
        cls_losses.append(sigmoid_focal_loss(pl, Y_proto, alpha=0.25, gamma=2.0, reduction="mean"))

        # Boxes: L1 + GIoU
        l1_losses.append(F.l1_loss(pb, gt_boxes, reduction="mean"))
        if use_nwd:
            # Use NWD for small objects
            nwd_loss = normalized_wasserstein_distance(pb, gt_boxes, C=0.01171875, reduction="mean")
            giou_losses.append(nwd_loss)
        else:
            # Original GIoU
            giou = generalized_box_iou(box_cxcywh_to_xyxy(pb), box_cxcywh_to_xyxy(gt_boxes))
            giou_losses.append(1.0 - giou.diag().mean())
        # Objectness (optional): positives at matched indices, negatives otherwise
        if use_objectness:
            obj_t = torch.zeros_like(ob)
            obj_t[idx_q] = 1.0
            obj_losses.append(F.binary_cross_entropy_with_logits(ob, obj_t))

    L_cls = torch.stack(cls_losses).mean() if cls_losses else torch.tensor(0.0, device=pred_boxes.device)
    L_l1  = torch.stack(l1_losses).mean() if l1_losses else torch.tensor(0.0, device=pred_boxes.device)
    L_g   = torch.stack(giou_losses).mean() if giou_losses else torch.tensor(0.0, device=pred_boxes.device)
    L_obj = torch.stack(obj_losses).mean() if obj_losses else torch.tensor(0.0, device=pred_boxes.device)

    loss = lambda_cls*L_cls + lambda_l1*L_l1 + lambda_giou*L_g + (0.5*L_obj if use_objectness else 0.0)

    losses.update(dict(loss=loss, L_cls=L_cls, L_l1=L_l1, L_giou=L_g, L_obj=L_obj))
    return losses
