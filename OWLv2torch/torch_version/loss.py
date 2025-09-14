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

def box_cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def hungarian_match(class_logits, pred_boxes, targets, lambda_cls=1.0, lambda_l1=5.0, lambda_giou=2.0):
    """
    class_logits: [B, Q, C]
    pred_boxes:   [B, Q, 4] (cxcywh in [0,1])
    targets: list of dicts per image: {"boxes": [M,4] (cxcywh in [0,1]), "labels": [M]}
    returns: list of (idx_pred, idx_gt) for each batch item
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:
        linear_sum_assignment = None

    out = []
    B, Q, C = class_logits.shape
    for b in range(B):
        boxes = pred_boxes[b]                             # [Q,4]
        cls = class_logits[b]                             # [Q,C]
        tgt = targets[b]
        if len(tgt["boxes"]) == 0:
            out.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
            continue
        tboxes = tgt["boxes"].to(boxes.device)            # [M,4] cxcywh
        tlabels = tgt["labels"].to(cls.device).long()     # [M]
        M = tboxes.shape[0]

        with torch.no_grad():
            # classification cost: -log_softmax over classes
            cls_cost = -F.log_softmax(cls, dim=-1)[:, tlabels]   # [Q,M]

            # bbox costs
            b1 = boxes
            b2 = tboxes
            l1_cost = torch.cdist(b1, b2, p=1)                    # [Q,M]

            giou = generalized_box_iou(box_cxcywh_to_xyxy(b1), box_cxcywh_to_xyxy(b2))  # [Q,M]
            giou_cost = 1 - giou

            cost = lambda_cls*cls_cost + lambda_l1*l1_cost + lambda_giou*giou_cost
            cost = cost.cpu()

            if linear_sum_assignment is not None:
                row_ind, col_ind = linear_sum_assignment(cost)
                idx_pred = torch.as_tensor(row_ind, dtype=torch.long)
                idx_gt = torch.as_tensor(col_ind, dtype=torch.long)
            else:
                # Greedy fallback
                idx_pred, idx_gt = [], []
                used_p = set()
                used_g = set()
                for _ in range(min(Q, M)):
                    q, g = divmod(cost.argmin().item(), M)
                    while q in used_p or g in used_g:
                        cost[q, g] = math.inf
                        q, g = divmod(cost.argmin().item(), M)
                    idx_pred.append(q); idx_gt.append(g)
                    cost[q, :] = math.inf
                    cost[:, g] = math.inf
                    used_p.add(q); used_g.add(g)
                idx_pred = torch.tensor(idx_pred, dtype=torch.long)
                idx_gt = torch.tensor(idx_gt, dtype=torch.long)

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
                   use_objectness=True):
    proto_logits, class_logits, objectness_logits, pred_boxes, _ = outputs
    B, Q, C = class_logits.shape
    losses = {}

    # Hungarian
    indices = hungarian_match(class_logits, pred_boxes, targets,
                              lambda_cls=lambda_cls, lambda_l1=lambda_l1, lambda_giou=lambda_giou)

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
        giou = generalized_box_iou(box_cxcywh_to_xyxy(pb), box_cxcywh_to_xyxy(gt_boxes))
        giou_losses.append(1.0 - giou.diag().mean())  # matched pairs on diagonal

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
