"""APE prior refinement for FLAME's SVM features (arXiv:2304.01195).

This ports the channel criterion from the authors' ``cal_criterion``:
https://github.com/yangyangyang127/APE/blob/main/utils.py
It does not replace FLAME with APE's cache classifier or train the backbone.
"""

from __future__ import annotations

import math

import torch
from torch.nn.functional import normalize


def select_channels(text, examples, channels, *, similarity_weight=0.9, source="text_support"):
    """Select shared channels using class separation, without evaluation labels.

    ``text`` is [classes, dimensions]. ``examples`` contains FLAME's per-class
    (features, binary targets) chunks. As in the authors' implementation, the
    variance term uses text features. ``text_support`` computes inter-class
    similarity from text plus positive support features; ``text`` uses only text.
    Class means make the pairwise sum exact even with unequal support counts.
    Background negatives remain available to the SVM, but do not define classes
    for APE's multiclass channel criterion.
    """
    if text.ndim != 2 or text.shape[0] < 2:
        raise ValueError("APE channel selection requires at least two queries")
    if not isinstance(channels, int) or not 1 <= channels <= text.shape[1]:
        raise ValueError(f"ape_channels must be between 1 and {text.shape[1]}")
    if not math.isfinite(similarity_weight) or not 0 <= similarity_weight <= 1:
        raise ValueError("ape_similarity_weight must be between 0 and 1")
    if source not in {"text", "text_support"}:
        raise ValueError("ape_source must be 'text' or 'text_support'")
    # Preserve the original order for a full-dimensional identity control.
    if channels == text.shape[1]:
        return torch.arange(channels)
    text = normalize(text.detach().float().cpu(), dim=-1)
    means = text.clone()
    if source == "text_support":
        for c, chunks in enumerate(examples):
            total, count = text[c].clone(), 1
            for features, targets in chunks:
                positive = features[targets == 1].float().cpu()
                total += positive.sum(0)
                count += len(positive)
            means[c] = total / count
    classes = len(text)
    similarity = (means.sum(0).square() - means.square().sum(0)) / (classes * (classes - 1))
    variance = text.var(0)
    criterion = (1 - similarity_weight) * variance - similarity_weight * similarity
    return criterion.argsort(descending=True, stable=True)[:channels].sort().values
