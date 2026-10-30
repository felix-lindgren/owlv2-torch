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
from torchvision.ops import batched_nms

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
        svm_backend: str = "torch",
        n_components: int = 512,
        svm_max_iter: int = 300,
        svm_seed: int = 0,
        svm_tol: float = 1e-3,
        svm_cache_size_mb: float = 256,
        svm_working_set_size: int = 512,
        ape_channels: int | None = None,
        ape_similarity_weight: float = 0.9,
        ape_source: str = "text_support",
        ape_normalize: bool = False,
        feature_source: str = "class",
        visual_weight: float = 0.5,
        visual_normalize: bool = True,
        box_refinement: bool = True,
    ):
        """``alpha`` weighs the SVM margin against the zero-shot logit; ``svm_c`` is C.

        ``svm_backend`` is ``sklearn`` (CPU RBF SVC), ``torch`` (full-kernel
        C-SVC on the model's device), ``nystrom`` or ``rff``. The Torch C-SVC
        uses ``svm_tol`` for its global KKT gap, ``svm_cache_size_mb`` MiB for
        kernel rows, and up to ``svm_working_set_size`` variables per solve.
        ``svm_max_iter`` bounds working-set solves for the exact Torch backend.
        The approximations fit ``n_components`` features in Torch on the model's
        device, using balanced, Huber-smoothed hinge loss (width 0.01).

        ``ape_channels`` enables experimental APE channel selection before the
        SVM's standardization. It requires at least two queries and preserves
        the original zero-shot logits. ``ape_source`` is ``text_support`` (text
        plus positive support features) or ``text``. The criterion weighs
        inter-class similarity by ``ape_similarity_weight`` and text variance
        by its complement. ``ape_normalize`` L2-normalizes the selected features
        before standardization, as in APE's cosine cache. ``None`` keeps the
        original FLAME features.

        ``feature_source`` is ``class`` (the default text-aligned embeddings),
        ``visual`` (features before the class projection), or ``combined``.
        ``visual_normalize`` L2-normalizes visual features. In combined mode,
        ``visual_weight`` sets their contribution to the standardized RBF
        distance; each branch is balanced for its dimension count. Zero-shot
        logits, objectness and boxes always use the original model heads.

        ``box_refinement`` fits the development-selected shared box-correction
        head by default when boxed visual support is available. ``add_example``
        retains at most four visual rows per matched object, and ``fit`` trains
        the head alongside the classifiers. Set it to ``False`` for original
        boxes. Yes/no-only support cannot train a box head.
        """
        if svm_backend not in {"sklearn", "torch", "nystrom", "rff"}:
            raise ValueError(f"Unknown SVM backend: {svm_backend}")
        if n_components < 1 or svm_max_iter < 1 or not math.isfinite(svm_c) or svm_c <= 0:
            raise ValueError("n_components, svm_max_iter and svm_c must be positive")
        if not all(math.isfinite(v) and v > 0 for v in (svm_tol, svm_cache_size_mb)):
            raise ValueError("svm_tol and svm_cache_size_mb must be finite and positive")
        if svm_working_set_size < 2:
            raise ValueError("svm_working_set_size must be at least 2")
        if feature_source not in {"class", "visual", "combined"}:
            raise ValueError("feature_source must be 'class', 'visual' or 'combined'")
        if not math.isfinite(visual_weight) or not 0 < visual_weight < 1:
            raise ValueError("visual_weight must be between 0 and 1, exclusive")
        if ape_channels is not None and feature_source != "class":
            raise ValueError("APE channel selection requires feature_source='class'")
        if not isinstance(box_refinement, bool):
            raise ValueError("box_refinement must be a boolean")
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
        self.ape_channels = ape_channels
        self.ape_similarity_weight = ape_similarity_weight
        self.ape_source = ape_source
        self.ape_normalize = ape_normalize
        self.ape_indices = None
        self.feature_source = feature_source
        self.visual_weight = visual_weight
        self.visual_normalize = visual_normalize
        self.device = next(model.parameters()).device
        tokens = tokenize(self.queries, context_length=16, truncate=True).to(self.device)
        with torch.inference_mode():
            self.text = model.get_text_features(tokens, tokens == 0).float()
        # Per query: (embeddings, targets) of every labelled proposal.
        self.examples: list[list[tuple[torch.Tensor, ...]]] = [[] for _ in self.queries]
        self.svms: list[dict | None] = [None] * len(self.queries)
        self.refiner = None
        self.refiner_config = None
        self.box_refiner = None
        from .flame_boxes import BoxRefiner

        self.box_refinement = box_refinement
        self.box_refiner_config = BoxRefiner().config
        self.box_examples = []
        self._box_examples_dirty = False
        self.sampling_config = None
        self.sampling_stats = None
        self._rng = torch.Generator().manual_seed(0)

    @torch.inference_mode()
    def proposal_features(self, images: list):
        """Yield ``(class_embeds, visual_feats, logits, objectness, boxes)`` per image.

        Both feature branches come from the same encoder pass. Boxes are xyxy
        pixels, and the three detection outputs always use the original heads.
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
                yield embeds[b].float(), feats[b].float(), logits[b].float(), objectness[b].float(), boxes[b]

    def select_features(self, class_embeds, visual_feats=None):
        """Choose SVM inputs from the two feature branches, including cached ones."""
        if self.feature_source == "class":
            return class_embeds.float()
        if visual_feats is None:
            raise ValueError("Visual features are required for this feature_source")
        visual = visual_feats.float()
        if self.visual_normalize:
            visual = torch.nn.functional.normalize(visual, dim=-1)
        if self.feature_source == "visual":
            return visual
        return torch.cat((class_embeds.float(), visual), dim=-1)

    @property
    def feature_dim(self):
        if self.feature_source == "class":
            return self.text.shape[-1]
        return self.model.vision_dim + (self.text.shape[-1] if self.feature_source == "combined" else 0)

    @torch.inference_mode()
    def proposals(self, images: list):
        """Yield selected features [P,D], original logits [P,C], objectness [P], boxes [P,4]."""
        for embeds, visual, logits, objectness, boxes in self.proposal_features(images):
            yield self.select_features(embeds, visual), logits, objectness, boxes

    def add_example(self, image, boxes, labels=None, *, hard_negatives: int = 50, random_negatives: int = 50,
                    max_positives_per_object: int | None = None):
        """Learn from one image whose ``boxes`` cover every instance of every query in it.

        ``labels`` gives each box's query index (default: all 0). An image with no
        boxes is a pure negative. Proposals that fit a box tightly (IoU >= 0.7)
        become positives for its query; for every query, the highest-scoring
        proposals that miss all of its boxes (IoU < 0.3), plus a random sample of the
        rest, become negatives. Proposals in between are left out.
        ``max_positives_per_object`` optionally caps each matched object's rows,
        retaining its tightest proposal and then diverse feature directions.
        Box refinement retains its own matched visual rows by default.
        """
        embeds, visual, logits, objectness, proposals = next(self.proposal_features([image]))
        self._add_proposals((self.select_features(embeds, visual), logits, objectness, proposals), boxes, labels,
                            hard_negatives=hard_negatives, random_negatives=random_negatives,
                            max_positives_per_object=max_positives_per_object, visual_feats=visual)

    def _add_proposals(self, proposals, boxes, labels=None, *, hard_negatives=50, random_negatives=50,
                       max_positives_per_object=None, visual_feats=None):
        """Add an example from cached proposals, using the same sampling as add_example."""
        from .flame_sampling import sample_proposals

        states = sample_proposals(proposals, boxes, labels, len(self.queries), self._rng,
                                  hard_negatives=hard_negatives, random_negatives=random_negatives,
                                  max_positives_per_object=max_positives_per_object)
        for c, state in enumerate(states):
            self._append(c, proposals[0][state["positive"]], 1)
            self._append(c, proposals[0][torch.cat((state["hard"], state["easy"]))], 0)
        if self.box_refinement and visual_feats is not None:
            from .flame_boxes import BoxRefiner

            head = BoxRefiner(**self.box_refiner_config)
            self.box_examples.extend(head.prepare_support([(visual_feats, proposals[3], boxes)], device=self.device))
            self._box_examples_dirty = True

    def fit_support(self, support, *, max_positives_per_object=None, mining_rounds=0,
                    mining_nms=None, hard_negatives=50, random_negatives=50, seed=0):
        """Fit from cached ``(proposals, GT boxes xyxy, labels)`` records, optionally mining once.

        This rebuilds the training bank from the supplied full support pool.
        Support images must be fully annotated. Mining keeps the random negative
        anchors and replaces hard negatives at a fixed budget. The caller owns
        the full proposal cache; checkpoints retain selected examples and the
        sampling configuration. Timings and solver diagnostics are available in
        ``sampling_stats``.

        Full five-tensor :meth:`proposal_features` records also train the default
        box head. Four-tensor :meth:`proposals` records retain classifier-only
        behavior because they may omit the required visual features.
        """
        from .flame_sampling import fit_support

        classification_support, boxed_support = [], []
        for proposals, boxes, labels in support:
            if len(proposals) == 5:
                embeds, visual, logits, objectness, original_boxes = proposals
                proposals = (self.select_features(embeds, visual), logits, objectness, original_boxes)
                boxed_support.append((visual, original_boxes, boxes))
            elif len(proposals) != 4:
                raise ValueError("Expected four selected tensors or five full proposal-feature tensors")
            classification_support.append((proposals, boxes, labels))
        if not classification_support:
            raise ValueError("At least one fully annotated support record is required")
        if self.box_refinement:
            from .flame_boxes import BoxRefiner

            self.box_examples = BoxRefiner(**self.box_refiner_config).prepare_support(boxed_support, device=self.device)
            self._box_examples_dirty = True
        return fit_support(self, classification_support, max_positives_per_object=max_positives_per_object,
                           mining_rounds=mining_rounds, mining_nms=mining_nms,
                           hard_negatives=hard_negatives, random_negatives=random_negatives, seed=seed)

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
        """Train classifiers and refresh the box head from newly added boxed support.

        Queries without positive and negative labels retain zero-shot scores.
        """
        self.ape_indices = None
        if self.ape_channels is not None:
            if self.feature_source != "class":
                raise ValueError("APE channel selection requires feature_source='class'")
            from .flame_ape import select_channels

            self.ape_indices = select_channels(
                self.text, self.examples, self.ape_channels,
                similarity_weight=self.ape_similarity_weight, source=self.ape_source,
            )
        for c, examples in enumerate(self.examples):
            self.svms[c] = None
            if not examples:
                continue
            x, y = (torch.cat(t) for t in zip(*examples))
            x = x.float()
            if x.shape[-1] != self.feature_dim:
                raise ValueError("Stored examples do not match feature_source; extract support features again")
            if y.min() == y.max():
                continue
            if self.ape_indices is not None:
                x = x[:, self.ape_indices]
                if self.ape_normalize:
                    x = torch.nn.functional.normalize(x, dim=-1)
            mean, std = x.mean(0), x.std(0) + 1e-6
            if self.feature_source == "combined":
                # Apply branch weights after standardization, which would
                # otherwise cancel any constant scaling of the input features.
                split, total = self.text.shape[-1], x.shape[-1]
                std[:split] *= math.sqrt(split / ((1 - self.visual_weight) * total))
                std[split:] *= math.sqrt((total - split) / (self.visual_weight * total))
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
        if self.refiner_config is not None:
            self.fit_refiner(**self.refiner_config)
        self._fit_box_examples()
        return self

    def _fit_box_examples(self):
        if not self.box_refinement or not self._box_examples_dirty:
            return
        from .flame_boxes import BoxRefiner

        self.box_refiner = (BoxRefiner(**self.box_refiner_config).fit(self.box_examples, device=self.device)
                            if any(len(record[0]) for record in self.box_examples) else None)
        self._box_examples_dirty = False

    def fit_refiner(self, method, *, weight=1.0, svm_weight=1.0, gda_shrinkage="auto",
                    tip_beta=5.0, tip_cache_size=128, tip_negative=True, seed=0):
        """Fit an optional GDA or Tip-Adapter proposal head from stored labels.

        After ``fit()``, use ``svm_weight=1`` to add this head to FLAME. For a
        standalone refiner, call this directly with ``svm_weight=0`` to avoid
        fitting SVMs. Detection logits are ``original + alpha * svm_weight *
        svm_margin + weight * refiner_margin``. Both heads use the same stored
        examples; the auxiliary head normalizes the selected proposal features.
        ``fit()`` also refreshes an attached refiner after new labels are added.
        """
        from .flame_refiners import ProposalRefiner

        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Refiner weight must be finite and nonnegative")
        if not math.isfinite(svm_weight) or not 0 <= svm_weight <= 1:
            raise ValueError("svm_weight must be between 0 and 1")
        refiner = ProposalRefiner(method, gda_shrinkage=gda_shrinkage, tip_beta=tip_beta,
                                  tip_cache_size=tip_cache_size, tip_negative=tip_negative, seed=seed)
        refiner.fit(self.examples, self.feature_dim, device=self.device)
        self.refiner = refiner
        self.refiner_config = dict(method=method, weight=weight, svm_weight=svm_weight,
                                   gda_shrinkage=gda_shrinkage, tip_beta=tip_beta,
                                   tip_cache_size=tip_cache_size, tip_negative=tip_negative, seed=seed)
        return self

    def fit_box_refiner(self, support, *, ridge=0.01, min_iou=0.3, max_proposals_per_object=4,
                        max_center_shift=0.25, max_log_scale=0.4, strength=0.5):
        """Explicitly fit or replace the shared residual head from cached boxed support.

        ``support`` contains ``(visual_feats, original_proposal_boxes, GT_boxes)``
        per image, all boxes xyxy in pixels. Use the pre-projection features
        from :meth:`proposal_features`, independently of the SVM feature source.
        Yes/no labels alone cannot fit this head. It changes boxes before NMS
        in :meth:`detect` and is saved with the checkpoint. New boxed examples
        refresh it on :meth:`fit` when automatic box refinement is enabled.
        """
        from .flame_boxes import BoxRefiner

        head = BoxRefiner(ridge=ridge, min_iou=min_iou,
                          max_proposals_per_object=max_proposals_per_object,
                          max_center_shift=max_center_shift, max_log_scale=max_log_scale,
                          strength=strength)
        support = list(support)
        head.fit(support, device=self.device)
        if head.weight.shape[0] != self.model.vision_dim + 1:
            raise ValueError("Box refiner requires the model's pre-projection visual features")
        self.box_refiner = head
        self.box_refiner_config = head.config.copy()
        self.box_examples = head.prepare_support(support, device=self.device)
        self._box_examples_dirty = False
        return self

    def refine_boxes(self, visual_feats, boxes):
        """Apply the optional box head to cached proposals, or return original boxes."""
        return boxes if self.box_refiner is None else self.box_refiner(visual_feats, boxes)

    def margins(self, embeds):
        """SVM decision margins ``[P, C]``; zero for queries without a classifier."""
        if self.ape_indices is not None:
            embeds = embeds[:, self.ape_indices.to(embeds.device)]
            if self.ape_normalize:
                embeds = torch.nn.functional.normalize(embeds, dim=-1)
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
        """Combine original logits with the SVM and optional refiner corrections."""
        if self.refiner is not None:
            config = self.refiner_config
            correction = (self.alpha * config["svm_weight"] * self.margins(embeds)
                          if config["svm_weight"] else 0)
            if config["weight"]:
                correction = correction + config["weight"] * self.refiner.margins(embeds)
            return torch.sigmoid(logits + correction) * torch.sigmoid(objectness)[:, None]
        return torch.sigmoid(logits + self.alpha * self.margins(embeds)) * torch.sigmoid(objectness)[:, None]

    def detect(self, images, threshold: float = 0.01, top_k: int = 100, nms: float | None = 0.5):
        """Return ``(boxes xyxy, scores, labels)`` for one image, or a list of them for a list.

        Refined scores are low in absolute terms: F1 peaked at thresholds of
        0.005-0.02 on both DIOR and Fashionpedia (zero-shot: 0.02-0.05).
        """
        single = not isinstance(images, (list, tuple))
        results = []
        for embeds, visual, logits, objectness, boxes in self.proposal_features([images] if single else list(images)):
            scores = self.scores(self.select_features(embeds, visual), logits, objectness)
            boxes = self.refine_boxes(visual, boxes)
            results.append(tuple(t.cpu() for t in select_detections(boxes, scores, threshold, top_k, nms)))
        return results[0] if single else results

    def save(self, path):
        """Save training banks, settings and the box head; :meth:`load` refits classifiers."""
        torch.save({"queries": self.queries, "alpha": self.alpha, "svm_c": self.svm_c,
                    "svm_backend": self.svm_backend, "n_components": self.n_components,
                    "svm_max_iter": self.svm_max_iter, "svm_seed": self.svm_seed,
                    "svm_tol": self.svm_tol, "svm_cache_size_mb": self.svm_cache_size_mb,
                    "svm_working_set_size": self.svm_working_set_size,
                    "ape_channels": self.ape_channels,
                    "ape_similarity_weight": self.ape_similarity_weight,
                    "ape_source": self.ape_source,
                    "ape_normalize": self.ape_normalize,
                    "refiner_config": self.refiner_config,
                    "box_refiner": self.box_refiner.state_dict() if self.box_refiner is not None else None,
                    "box_refinement": self.box_refinement,
                    "box_refiner_config": self.box_refiner_config,
                    "box_examples": self.box_examples, "box_examples_dirty": self._box_examples_dirty,
                    "feature_source": self.feature_source, "visual_weight": self.visual_weight,
                    "visual_normalize": self.visual_normalize,
                    "sampling_config": self.sampling_config,
                    "examples": self.examples}, path)

    @classmethod
    def load(cls, path, model, **kwargs):
        state = torch.load(path, weights_only=True)
        options = {key: state[key] for key in ("svm_backend", "n_components", "svm_max_iter", "svm_seed",
                                              "svm_tol", "svm_cache_size_mb", "svm_working_set_size",
                                              "ape_channels", "ape_similarity_weight", "ape_source",
                                              "ape_normalize", "feature_source", "visual_weight",
                                              "visual_normalize", "box_refinement")
                   if key in state}
        # Files predating backend selection were fitted with sklearn.
        options.setdefault("svm_backend", "sklearn")
        options.setdefault("box_refinement", False)
        options.update(kwargs)
        saved_source = state.get("feature_source", "class")
        if options.get("feature_source", "class") != saved_source:
            raise ValueError("Changing checkpoint feature_source requires re-extracting support features")
        if saved_source != "class" and options.get("visual_normalize", True) != state.get("visual_normalize", True):
            raise ValueError("Changing checkpoint visual_normalize requires re-extracting support features")
        flame = cls(model, state["queries"], alpha=state["alpha"], svm_c=state["svm_c"], **options)
        flame.examples = state["examples"]
        flame.sampling_config = state.get("sampling_config")
        flame.box_examples = state.get("box_examples", [])
        flame._box_examples_dirty = state.get("box_examples_dirty", False)
        if state.get("box_refiner") is not None:
            from .flame_boxes import BoxRefiner

            flame.box_refiner = BoxRefiner.from_state_dict(state["box_refiner"])
            if flame.box_refiner.weight.shape[0] != model.vision_dim + 1:
                raise ValueError("Saved box refiner does not match the model's visual feature dimension")
            flame.box_refiner_config = flame.box_refiner.config.copy()
        flame.box_refiner_config = state.get("box_refiner_config", flame.box_refiner_config)
        if kwargs.get("box_refinement") is False:
            flame.box_refiner = None
        config = state.get("refiner_config")
        if config is not None:
            if config["svm_weight"]:
                flame.fit()
            else:
                flame._fit_box_examples()
            return flame.fit_refiner(**config)
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
