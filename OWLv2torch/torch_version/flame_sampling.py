"""Instance-capped positives and fixed-budget hard-negative mining for FLAME."""

from __future__ import annotations

import math
import time

import torch
from torch.nn.functional import normalize
from torchvision.ops import box_iou, nms


def _validate_counts(hard_negatives, random_negatives, max_positives_per_object):
    if any(not isinstance(n, int) or n < 0 for n in (hard_negatives, random_negatives)):
        raise ValueError("Negative sample counts must be nonnegative integers")
    if max_positives_per_object is not None and (
        not isinstance(max_positives_per_object, int) or max_positives_per_object < 1
    ):
        raise ValueError("max_positives_per_object must be None or a positive integer")


def _cap_positives(features, overlap, matched, cap):
    positives = (overlap >= 0.7).nonzero()[:, 0]
    if cap is None or not len(positives):
        return positives
    selected = []
    for object_id in matched[positives].unique():
        candidates = positives[matched[positives] == object_id]
        if len(candidates) <= cap:
            selected.append(candidates)
            continue
        # Start at the tightest proposal, then cover different feature directions.
        candidates = candidates[overlap[candidates].argsort(descending=True, stable=True)]
        x = normalize(features[candidates].float(), dim=-1)
        chosen = [0]
        distance = x.new_full((len(x),), float("inf"))
        for _ in range(cap - 1):
            distance = torch.minimum(distance, (1 - x @ x[chosen[-1]]).clamp_min(0))
            distance[chosen] = -1
            chosen.append(int(distance.argmax()))
        selected.append(candidates[torch.tensor(chosen, device=candidates.device)])
    return torch.cat(selected).sort().values


@torch.inference_mode()
def sample_proposals(proposals, boxes, labels, classes, rng, *, hard_negatives=50,
                     random_negatives=50, max_positives_per_object=None):
    """Return proposal indices per query; uncapped sampling matches original FLAME."""
    _validate_counts(hard_negatives, random_negatives, max_positives_per_object)
    features, logits, objectness, props = proposals
    boxes = torch.as_tensor(boxes, dtype=torch.float32, device=props.device).reshape(-1, 4)
    labels = (torch.zeros(len(boxes), dtype=torch.long, device=props.device) if labels is None
              else torch.as_tensor(labels, device=props.device))
    if len(labels) != len(boxes):
        raise ValueError(f"got {len(boxes)} boxes but {len(labels)} labels")
    if bool(((labels < 0) | (labels >= classes)).any()):
        raise ValueError("Support labels must index the queries")
    scores = torch.sigmoid(logits) * torch.sigmoid(objectness)[:, None]
    states = []
    for c in range(classes):
        own = boxes[labels == c]
        if len(own):
            overlap, matched = box_iou(props, own).max(1)
        else:
            overlap = props.new_zeros(len(props))
            matched = torch.zeros(len(props), dtype=torch.long, device=props.device)
        negative = (overlap < 0.3).nonzero()[:, 0]
        hard = negative[scores[negative, c].argsort(descending=True)[:hard_negatives]]
        rest = negative[~torch.isin(negative, hard)]
        easy = rest[torch.randperm(len(rest), generator=rng)[:random_negatives].to(rest.device)]
        positive = _cap_positives(features, overlap, matched, max_positives_per_object)
        states.append(dict(positive=positive, negative_pool=negative, hard=hard, easy=easy,
                           positive_before=int((overlap >= 0.7).sum()), objects=len(own),
                           represented_objects=len(matched[positive].unique()) if len(positive) else 0))
    return states


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _counts(flame):
    counts = []
    for chunks in flame.examples:
        y = torch.cat([y for _, y in chunks]) if chunks else torch.empty(0)
        counts.append(dict(positive=int((y == 1).sum()), negative=int((y == 0).sum())))
    return counts


def _populate(flame, support, selections):
    flame.examples = [[] for _ in flame.queries]
    for (proposals, _, _), states in zip(support, selections):
        features = proposals[0]
        for c, state in enumerate(states):
            flame._append(c, features[state["positive"].to(features.device)], 1)
            negatives = torch.cat((state["hard"], state["easy"]))
            flame._append(c, features[negatives.to(features.device)], 0)


def fit_support(flame, support, *, max_positives_per_object=None, mining_rounds=0,
                mining_nms=None, hard_negatives=50, random_negatives=50, seed=0):
    """Fit from cached (proposals, xyxy ground-truth boxes, labels) records.

    A mining round replaces hard negatives while retaining the original random
    anchors and selected positives. This keeps the training-bank budget fixed.
    Ground truth must cover every instance of every query in the support images.
    The full support pool is caller-owned and is not saved in a checkpoint.
    """
    _validate_counts(hard_negatives, random_negatives, max_positives_per_object)
    if not isinstance(mining_rounds, int) or mining_rounds < 0:
        raise ValueError("mining_rounds must be a nonnegative integer")
    if mining_nms is not None and (not math.isfinite(mining_nms) or not 0 < mining_nms <= 1):
        raise ValueError("mining_nms must be None or between 0 and 1")
    support = list(support)
    if not support:
        raise ValueError("At least one fully annotated support record is required")
    config = dict(max_positives_per_object=max_positives_per_object, mining_rounds=mining_rounds,
                  mining_nms=mining_nms, hard_negatives=hard_negatives,
                  random_negatives=random_negatives, seed=seed)
    rng = torch.Generator().manual_seed(seed)
    _synchronize(flame.device)
    start = time.perf_counter()
    selections = []
    with torch.inference_mode():
        for proposals, boxes, labels in support:
            states = sample_proposals(tuple(t.to(flame.device) for t in proposals), boxes, labels,
                                      len(flame.queries), rng, hard_negatives=hard_negatives,
                                      random_negatives=random_negatives,
                                      max_positives_per_object=max_positives_per_object)
            selections.append([{k: v.cpu() if torch.is_tensor(v) else v for k, v in s.items()} for s in states])
    _populate(flame, support, selections)
    _synchronize(flame.device)
    stats = dict(config=config, sampling_seconds=time.perf_counter() - start,
                 initial_counts=_counts(flame), rounds=[])
    stats["objects"] = [sum(states[c]["objects"] for states in selections) for c in range(len(flame.queries))]
    stats["represented_objects"] = [sum(states[c]["represented_objects"] for states in selections)
                                    for c in range(len(flame.queries))]
    stats["positive_before"] = [sum(states[c]["positive_before"] for states in selections)
                               for c in range(len(flame.queries))]

    def fit_once():
        _synchronize(flame.device)
        started = time.perf_counter()
        flame.fit()
        _synchronize(flame.device)
        return dict(fit_seconds=time.perf_counter() - started,
                    solver=[head.get("fit") if head else None for head in flame.svms])

    stats["initial_fit"] = fit_once()
    for _ in range(mining_rounds):
        _synchronize(flame.device)
        start = time.perf_counter()
        changed, total, fallback = 0, 0, 0
        with torch.inference_mode():
            for (proposals, _, _), states in zip(support, selections):
                features, logits, objectness, boxes = (t.to(flame.device) for t in proposals)
                scores = flame.scores(features, logits, objectness)
                for c, state in enumerate(states):
                    eligible = state["negative_pool"].to(flame.device)
                    easy = state["easy"].to(flame.device)
                    eligible = eligible[~torch.isin(eligible, easy)]
                    budget = len(state["hard"])
                    ranked = eligible[scores[eligible, c].argsort(descending=True)]
                    if mining_nms is None:
                        selected = ranked[:budget]
                    else:
                        selected = ranked[nms(boxes[ranked], scores[ranked, c], mining_nms)[:budget]]
                        if len(selected) < budget:
                            fill = ranked[~torch.isin(ranked, selected)][:budget - len(selected)]
                            fallback += len(fill)
                            selected = torch.cat((selected, fill))
                    selected = selected.cpu()
                    if len(selected) != budget or bool(torch.isin(selected, state["easy"]).any()):
                        raise RuntimeError("Mining changed the negative budget or duplicated random anchors")
                    changed += int((~torch.isin(selected, state["hard"])).sum())
                    total += budget
                    state["hard"] = selected
        _populate(flame, support, selections)
        if _counts(flame) != stats["initial_counts"]:
            raise RuntimeError("A mining round changed the positive or negative sample counts")
        _synchronize(flame.device)
        stage = dict(mining_seconds=time.perf_counter() - start, changed_hard_negatives=changed,
                     hard_negative_count=total, nms_fill_count=fallback)
        stage.update(fit_once())
        stats["rounds"].append(stage)
    stats["final_counts"] = _counts(flame)
    flame.sampling_config, flame.sampling_stats = config, stats
    return flame
