"""Frozen-feature GDA and Tip-Adapter refiners for FLAME proposals.

GDA: https://arxiv.org/abs/2402.04087
Tip-Adapter: https://arxiv.org/abs/2207.09519

Detection adaptations use one foreground/background task per query, equal
priors, and normalized margins/cache counts so different support counts do
not determine a class's fusion weight. No backbone parameters are updated.
"""

from __future__ import annotations

import math

import torch
from torch.nn.functional import normalize


class ProposalRefiner:
    def __init__(self, method, *, gda_shrinkage="auto", tip_beta=5.0,
                 tip_cache_size=128, tip_negative=True, seed=0):
        if method not in {"gda", "tip"}:
            raise ValueError("Refiner method must be 'gda' or 'tip'")
        if gda_shrinkage != "auto" and (
            not isinstance(gda_shrinkage, (int, float)) or not math.isfinite(gda_shrinkage)
            or not 0 < gda_shrinkage <= 1
        ):
            raise ValueError("gda_shrinkage must be 'auto' or a number in (0, 1]")
        if not math.isfinite(tip_beta) or tip_beta <= 0:
            raise ValueError("tip_beta must be finite and positive")
        if not isinstance(tip_cache_size, int) or tip_cache_size < 1:
            raise ValueError("tip_cache_size must be a positive integer")
        self.method = method
        self.gda_shrinkage = gda_shrinkage
        self.tip_beta = tip_beta
        self.tip_cache_size = tip_cache_size
        self.tip_negative = tip_negative
        self.seed = seed
        self.state = {}
        self.counts = []
        self.missing = []

    @torch.inference_mode()
    def fit(self, examples, dimension, *, device="cpu"):
        device = torch.device(device)
        classes = len(examples)
        self.counts, self.missing = [], []
        weights = torch.zeros(dimension, classes)
        biases = torch.zeros(classes)
        cache_keys, cache_values = [], []
        for c, chunks in enumerate(examples):
            if chunks:
                x, y = (torch.cat(t) for t in zip(*chunks))
                x = normalize(x.float().to(device), dim=-1)
                y = y.to(device)
            else:
                x = torch.empty(0, dimension, device=device)
                y = torch.empty(0, device=device)
            positive, negative = x[y == 1], x[y == 0]
            self.counts.append({"positive": len(positive), "negative": len(negative)})
            need_negative = self.method == "gda" or self.tip_negative
            if not len(positive) or (need_negative and not len(negative)):
                self.missing.append(c)
                continue
            if self.method == "gda":
                # Pooled within-class covariance with equal classification priors.
                positive, negative = positive.double(), negative.double()
                mean_pos, mean_neg = positive.mean(0), negative.mean(0)
                pos_centered, neg_centered = positive - mean_pos, negative - mean_neg
                n = len(positive) + len(negative)
                covariance = (pos_centered.T @ pos_centered + neg_centered.T @ neg_centered) / max(n - 2, 1)
                shrinkage = (dimension / (n - 1 + dimension)
                             if self.gda_shrinkage == "auto" else self.gda_shrinkage)
                # The automatic ridge has the paper's empirical Bayes direction.
                variance = covariance.diagonal().mean().clamp_min(1e-12)
                covariance *= 1 - shrinkage
                covariance.diagonal().add_(shrinkage * variance + 1e-12)
                delta = mean_pos - mean_neg
                weight = torch.cholesky_solve(delta[:, None], torch.linalg.cholesky(covariance))[:, 0]
                # Anchor the two class means at +1/-1 for stable residual fusion.
                scale = (0.5 * delta @ weight).clamp_min(1e-12)
                weight /= scale
                weights[:, c] = weight.float().cpu()
                biases[c] = float(-0.5 * (mean_pos + mean_neg) @ weight)
            else:
                rng = torch.Generator().manual_seed(self.seed + c)
                for group, sign in [(positive, 1.0)] + ([(negative, -1.0)] if self.tip_negative else []):
                    if len(group) > self.tip_cache_size:
                        indices = torch.randperm(len(group), generator=rng)[:self.tip_cache_size]
                        group = group[indices.to(device)]
                    values = torch.zeros(len(group), classes)
                    values[:, c] = sign / len(group)
                    cache_keys.append(group.cpu())
                    cache_values.append(values)
        if self.method == "gda":
            self.state = {"weight": weights, "bias": biases}
        else:
            self.state = {
                "keys": torch.cat(cache_keys) if cache_keys else torch.empty(0, dimension),
                "values": torch.cat(cache_values) if cache_values else torch.empty(0, classes),
            }
        return self

    @torch.inference_mode()
    def margins(self, embeds):
        """Return [proposals, queries] corrections; missing queries get zero."""
        x = normalize(embeds.float(), dim=-1)
        if self.method == "gda":
            return x @ self.state["weight"].to(x.device) + self.state["bias"].to(x.device)
        keys, values = self.state["keys"].to(x.device), self.state["values"].to(x.device)
        if not len(keys):
            return x.new_zeros((len(x), values.shape[1]))
        # Bound affinity memory independently of query/support counts.
        chunk_size = max(1, min(4096, 8_000_000 // len(keys)))
        output = x.new_empty((len(x), values.shape[1]))
        for start in range(0, len(x), chunk_size):
            affinity = x[start:start + chunk_size] @ keys.T
            affinity = torch.exp(-self.tip_beta * (1 - affinity.clamp(-1, 1)))
            output[start:start + chunk_size] = affinity @ values
        return output

    @property
    def storage_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.state.values())
