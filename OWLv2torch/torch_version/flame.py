"""FLAME few-shot refinement for OWLv2 ("On-the-Fly OVD Adaptation with FLAME", arXiv 2510.17670).

A cascade: zero-shot OWLv2 scores every patch, and a small RBF-SVM per query,
trained on a handful of labelled proposals, corrects that score. The SVM margin
is added to the zero-shot logit rather than replacing it, so the text prior still
ranks whatever the few examples say nothing about.

    flame = Flame(model, ["a photo of a chimney", "a photo of a dam"])

    # Either give it boxes: every instance of every query in a few images.
    flame.add_example(image, boxes=[[x1, y1, x2, y2], ...], labels=[0, ...])

    # Or let FLAME pick what to ask about (the paper's active step) and answer yes/no.
    candidates = flame.suggest(unlabelled_images, k=30)
    flame.add_labels(candidates, [True, False, ...])

    flame.fit()
    boxes, scores, labels = flame.detect(image)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torchvision.ops import batched_nms, box_iou

from OWLv2torch.utils.tokenizer import tokenize


@dataclass
class Candidate:
    """A proposal FLAME wants labelled: is ``box`` in ``images[image]`` a ``queries[label]``?"""

    image: int
    box: list[float]
    label: int
    score: float
    embedding: torch.Tensor = field(repr=False)


class Flame:
    def __init__(
        self,
        model,
        queries: str | Sequence[str],
        *,
        alpha: float = 8.0,
        svm_c: float = 1.0,
        batch_size: int = 8,
        svm_backend: str = "sklearn",
        n_components: int = 512,
        svm_max_iter: int = 300,
        svm_seed: int = 0,
        svm_tol: float = 1e-3,
        svm_cache_size_mb: float = 256,
        svm_working_set_size: int = 512,
    ):
        """``alpha`` weighs the SVM margin against the zero-shot logit; ``svm_c`` is C.

        ``svm_backend`` is ``sklearn`` (CPU RBF SVC), ``torch`` (full-kernel
        C-SVC on the model's device), ``nystrom`` or ``rff``. The Torch C-SVC
        uses ``svm_tol`` for its global KKT gap, ``svm_cache_size_mb`` MiB for
        kernel rows, and up to ``svm_working_set_size`` variables per solve.
        ``svm_max_iter`` bounds working-set solves for the exact Torch backend.
        The approximations fit ``n_components`` features in Torch on the model's
        device, using balanced, Huber-smoothed hinge loss (width 0.01).
        """
        if svm_backend not in {"sklearn", "torch", "nystrom", "rff"}:
            raise ValueError(f"Unknown SVM backend: {svm_backend}")
        if n_components < 1 or svm_max_iter < 1 or not math.isfinite(svm_c) or svm_c <= 0:
            raise ValueError("n_components, svm_max_iter and svm_c must be positive")
        if not all(math.isfinite(v) and v > 0 for v in (svm_tol, svm_cache_size_mb)):
            raise ValueError("svm_tol and svm_cache_size_mb must be finite and positive")
        if svm_working_set_size < 2:
            raise ValueError("svm_working_set_size must be at least 2")
        self.model = model.eval()
        self.queries = [queries] if isinstance(queries, str) else list(queries)
        self.alpha = alpha
        self.svm_c = svm_c
        self.batch_size = batch_size
        self.svm_backend = svm_backend
        self.n_components = n_components
        self.svm_max_iter = svm_max_iter
        self.svm_seed = svm_seed
        self.svm_tol = svm_tol
        self.svm_cache_size_mb = svm_cache_size_mb
        self.svm_working_set_size = svm_working_set_size
        self.device = next(model.parameters()).device
        tokens = tokenize(self.queries, context_length=16, truncate=True).to(self.device)
        with torch.inference_mode():
            self.text = model.get_text_features(tokens, tokens == 0).float()
        # Per query: (embeddings, targets) of every labelled proposal.
        self.examples: list[list[tuple[torch.Tensor, ...]]] = [[] for _ in self.queries]
        self.svms: list[dict | None] = [None] * len(self.queries)
        self._rng = torch.Generator().manual_seed(0)

    @torch.inference_mode()
    def proposals(self, images: list):
        """Yield ``(embeds [P, D], logits [P, C], objectness [P], boxes [P, 4])`` per image.

        One entry per patch: its normalized class embedding, zero-shot logit for each
        query, objectness logit and ``xyxy`` box in pixels.
        """
        model = self.model
        for start in range(0, len(images), self.batch_size):
            batch = [_load(im) for im in images[start:start + self.batch_size]]
            sizes = [(im.height, im.width) for im in batch]
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                pixels = model.preprocess_image(batch, device=self.device)
                feats, _ = model._image_grid_features(pixels)
                text = self.text.unsqueeze(0).expand(len(batch), -1, -1)
                logits, embeds = model.class_head(feats, text, None)
                objectness = model.objectness_head(feats)[..., 0]
                boxes = torch.sigmoid(model.box_head(feats) + model.box_bias)
            boxes = model.postprocess_boxes(boxes.float(), sizes)
            for b in range(len(batch)):
                yield embeds[b].float(), logits[b].float(), objectness[b].float(), boxes[b]

    def add_example(self, image, boxes, labels=None, *, hard_negatives: int = 50, random_negatives: int = 50):
        """Learn from one image whose ``boxes`` cover every instance of every query in it.

        ``labels`` gives each box's query index (default: all 0). An image with no
        boxes is a pure negative. Proposals that fit a box tightly (IoU >= 0.7)
        become positives for its query; for every query, the highest-scoring
        proposals that miss all of its boxes (IoU < 0.3), plus a random sample of the
        rest, become negatives. Proposals in between are left out.
        """
        self._add_proposals(next(self.proposals([image])), boxes, labels,
                            hard_negatives=hard_negatives, random_negatives=random_negatives)

    def _add_proposals(self, proposals, boxes, labels=None, *, hard_negatives=50, random_negatives=50):
        """Add an example from cached proposals, using the same sampling as add_example."""
        embeds, logits, objectness, props = proposals
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4).to(props.device)
        labels = torch.zeros(len(boxes), dtype=torch.long) if labels is None else torch.as_tensor(labels)
        if len(labels) != len(boxes):
            raise ValueError(f"got {len(boxes)} boxes but {len(labels)} labels")
        scores = torch.sigmoid(logits) * torch.sigmoid(objectness)[:, None]
        for c in range(len(self.queries)):
            own = boxes[labels.to(props.device) == c]
            overlap = box_iou(props, own).amax(1) if len(own) else torch.zeros(len(props), device=props.device)
            negatives = (overlap < 0.3).nonzero()[:, 0]
            hard = negatives[scores[negatives, c].argsort(descending=True)[:hard_negatives]]
            rest = negatives[~torch.isin(negatives, hard)]
            easy = rest[torch.randperm(len(rest), generator=self._rng)[:random_negatives].to(rest.device)]
            positives = (overlap >= 0.7).nonzero()[:, 0]
            self._append(c, embeds[positives], 1)
            self._append(c, embeds[torch.cat([hard, easy])], 0)

    def suggest(self, images, k: int = 30, *, per_image: int = 100) -> list[Candidate]:
        """Pick ``k`` proposals per query worth asking about, from unlabelled ``images``.

        This is the paper's one-step active selection (Algorithm 1): keep the
        uncertain proposals near the zero-shot decision boundary, then take the one
        closest to each of ``k`` k-means centres so the questions are diverse. The
        boundary band is the ``4 * k`` best zero-shot scores in the pool rather than
        the paper's PCA + KDE density band, which on DIOR picked ~2 positives out of
        30 and did no better than k-means alone. The pool should contain the object:
        if none of the ``k`` answers is yes, that query stays zero-shot.
        """
        pool = [[] for _ in self.queries]  # per query: (image, zero-shot score, embed, box)
        for i, (embeds, logits, objectness, boxes) in enumerate(self.proposals(list(images))):
            scores = torch.sigmoid(logits) * torch.sigmoid(objectness)[:, None]
            for c in range(len(self.queries)):
                top = scores[:, c].topk(min(per_image, len(scores))).indices
                pool[c] += zip([i] * len(top), scores[top, c].tolist(), embeds[top].cpu(), boxes[top].tolist())
        candidates = []
        for c, items in enumerate(pool):
            scores = np.array([s for _, s, _, _ in items])
            embeds = torch.stack([e for _, _, e, _ in items]).numpy()
            for j in _marginal_samples(embeds, scores, k):
                i, s, e, b = items[j]
                candidates.append(Candidate(i, b, c, s, e))
        return candidates

    def add_labels(self, candidates: Sequence[Candidate], answers: Sequence[bool]):
        """Record yes/no answers for candidates returned by :meth:`suggest`."""
        if len(candidates) != len(answers):
            raise ValueError(f"got {len(candidates)} candidates but {len(answers)} answers")
        for cand, yes in zip(candidates, answers):
            self._append(cand.label, cand.embedding[None], int(bool(yes)))

    def _append(self, c, embeds, target):
        if len(embeds):
            self.examples[c].append((embeds.detach().half().cpu(),
                                     torch.full((len(embeds),), target, dtype=torch.int8)))

    def fit(self):
        """Train one SVM per query that has both positives and negatives; others stay zero-shot."""
        for c, examples in enumerate(self.examples):
            self.svms[c] = None
            if not examples:
                continue
            x, y = (torch.cat(t) for t in zip(*examples))
            x = x.float()
            if y.min() == y.max():
                continue
            mean, std = x.mean(0), x.std(0) + 1e-6
            z = (x - mean) / std
            if self.svm_backend != "sklearn":
                variance = float(z.double().var(correction=0))
                gamma = 1.0 / (z.shape[1] * variance) if variance else 1.0
                if self.svm_backend == "torch":
                    from .flame_svm import fit_rbf

                    fitted = fit_rbf(z, y, gamma=gamma, c=self.svm_c, device=self.device,
                                     max_iter=self.svm_max_iter, tol=self.svm_tol,
                                     cache_size_mb=self.svm_cache_size_mb,
                                     working_set_size=self.svm_working_set_size)
                else:
                    from .flame_features import fit_approximate

                    fitted = fit_approximate(
                        z, y, backend=self.svm_backend, gamma=gamma, c=self.svm_c,
                        n_components=self.n_components, device=self.device,
                        seed=self.svm_seed + c, max_iter=self.svm_max_iter,
                    )
                self.svms[c] = {"mean": mean, "std": std, **fitted}
                continue
            from sklearn.svm import SVC

            svc = SVC(C=self.svm_c, kernel="rbf", gamma="scale", class_weight="balanced")
            svc.fit(z.numpy(), y.numpy())
            self.svms[c] = {
                "mean": mean, "std": std, "gamma": float(svc._gamma),
                "vectors": torch.as_tensor(svc.support_vectors_, dtype=torch.float32),
                "coef": torch.as_tensor(svc.dual_coef_[0], dtype=torch.float32),
                "bias": float(svc.intercept_[0]),
            }
        missing = [q for q, s in zip(self.queries, self.svms) if s is None]
        if missing:
            print(f"[FLAME] no positive+negative examples for {missing}; those stay zero-shot")
        return self

    def margins(self, embeds):
        """SVM decision margins ``[P, C]``; zero for queries without a classifier."""
        out = embeds.new_zeros((len(embeds), len(self.queries)))
        for c, svm in enumerate(self.svms):
            if svm is None:
                continue
            x = (embeds - svm["mean"].to(embeds.device)) / svm["std"].to(embeds.device)
            if svm.get("backend") == "rff":
                features = torch.cos(x @ svm["frequencies"].to(embeds.device)
                                     + svm["phases"].to(embeds.device))
                margin = features @ svm["coef"].to(embeds.device) + svm["bias"]
            else:
                sv = svm["vectors"].to(embeds.device)
                dist = (x * x).sum(1, keepdim=True) + (sv * sv).sum(1) - 2 * x @ sv.T
                margin = torch.exp(-svm["gamma"] * dist.clamp_min(0)) @ svm["coef"].to(embeds.device) + svm["bias"]
            out[:, c] = margin
        return out

    def scores(self, embeds, logits, objectness):
        """``sigmoid(zero-shot logit + alpha * margin) * sigmoid(objectness)``."""
        return torch.sigmoid(logits + self.alpha * self.margins(embeds)) * torch.sigmoid(objectness)[:, None]

    def detect(self, images, threshold: float = 0.01, top_k: int = 100, nms: float | None = 0.5):
        """Return ``(boxes xyxy, scores, labels)`` for one image, or a list of them for a list.

        Refined scores are low in absolute terms: F1 peaked at thresholds of
        0.005-0.02 on both DIOR and Fashionpedia (zero-shot: 0.02-0.05).
        """
        single = not isinstance(images, (list, tuple))
        results = []
        for embeds, logits, objectness, boxes in self.proposals([images] if single else list(images)):
            scores = self.scores(embeds, logits, objectness)
            results.append(tuple(t.cpu() for t in select_detections(boxes, scores, threshold, top_k, nms)))
        return results[0] if single else results

    def save(self, path):
        """Save the queries and labelled examples; :meth:`load` refits the SVMs from them."""
        torch.save({"queries": self.queries, "alpha": self.alpha, "svm_c": self.svm_c,
                    "svm_backend": self.svm_backend, "n_components": self.n_components,
                    "svm_max_iter": self.svm_max_iter, "svm_seed": self.svm_seed,
                    "svm_tol": self.svm_tol, "svm_cache_size_mb": self.svm_cache_size_mb,
                    "svm_working_set_size": self.svm_working_set_size,
                    "examples": self.examples}, path)

    @classmethod
    def load(cls, path, model, **kwargs):
        state = torch.load(path, weights_only=True)
        options = {key: state[key] for key in ("svm_backend", "n_components", "svm_max_iter", "svm_seed",
                                              "svm_tol", "svm_cache_size_mb", "svm_working_set_size")
                   if key in state}
        options.update(kwargs)
        flame = cls(model, state["queries"], alpha=state["alpha"], svm_c=state["svm_c"], **options)
        flame.examples = state["examples"]
        return flame.fit()


def select_detections(boxes, scores, threshold: float = 0.01, top_k: int = 100, nms: float | None = 0.5):
    """Give each patch its best query, then threshold, per-class NMS and keep the ``top_k`` best."""
    scores, labels = scores.max(1)
    keep = (scores >= threshold).nonzero()[:, 0]
    if nms:
        keep = keep[batched_nms(boxes[keep], scores[keep], labels[keep], nms)]
    keep = keep[scores[keep].argsort(descending=True)[:top_k]]
    return boxes[keep], scores[keep], labels[keep]


def _marginal_samples(embeds: np.ndarray, scores: np.ndarray, k: int, band: int = 4) -> np.ndarray:
    """Indices of ``k`` diverse proposals among the ``band * k`` best zero-shot scores."""
    from sklearn.cluster import KMeans

    if len(embeds) <= k:
        return np.arange(len(embeds))
    marginal = np.argsort(-scores)[:band * k]
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(embeds[marginal])
    picks = []
    for j in range(k):
        members = marginal[km.labels_ == j]
        if len(members):
            picks.append(members[np.linalg.norm(embeds[members] - km.cluster_centers_[j], axis=1).argmin()])
    return np.array(picks)


def _load(image):
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    return image.convert("RGB")
