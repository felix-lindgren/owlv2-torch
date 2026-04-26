"""
FLAME: Few-shot Localization via Active Marginal-Samples Exploration

A cascaded post-processing pipeline that refines OWLv2 zero-shot detections
using a lightweight classifier trained on actively selected, user-annotated samples.

Reference: "On-the-Fly OVD Adaptation with FLAME" (arXiv 2510.17670)
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from OWLv2torch.torch_version.owlv2 import OwlV2


@dataclass
class FlameConfig:
    """Configuration for the FLAME pipeline."""
    # Proposal generation
    objectness_threshold: float = 0.1       # low threshold for high recall
    text_score_threshold: float = 0.0       # threshold on sigmoid(text logit); 0 keeps all text-matched proposals
    max_proposals_per_image: int = 500      # cap proposals per image

    # Active sample selection
    pca_components: int = 32                # PCA dimensionality for density estimation
    kde_bandwidth: float = 0.5              # KDE bandwidth
    # Marginal band defined as { s : rl * f̂(s*) <= f̂(s) <= ru * f̂(s*) },
    # i.e. the "donut" around the density mode (paper Algorithm 1, line 7).
    marginal_density_ratio_low: float = 0.25   # rl in (0, 1)
    marginal_density_ratio_high: float = 0.75  # ru in (rl, 1)
    num_clusters: int = 30                  # number of clusters (= num shots to annotate)

    # Classifier
    classifier_type: str = "svm"            # "svm" or "mlp"
    # The paper uses an RBF SVM by default for the lightweight refiner.
    svm_kernel: str = "rbf"
    svm_C: float = 1.0
    mlp_hidden: tuple = (128, 64)
    mlp_max_iter: int = 500

    # Inference
    # For SVM, the refiner score is sigmoid(decision_function); threshold 0.5
    # therefore corresponds to the SVM's natural decision boundary
    # (decision_function >= 0). For MLP it is the predict_proba threshold.
    classifier_threshold: float = 0.5

    # Logging
    verbose: bool = True                    # set False to silence per-call FLAME prints

    # Preprocessing
    fast_image_preprocess: bool = False     # use torchvision resize instead of scipy exact resize

    # Inference precision
    use_autocast: bool = True               # use CUDA autocast for OWLv2 inference


@dataclass
class FlameProposal:
    """A single detection proposal with its features."""
    box: np.ndarray              # [4] xyxy in pixel coords
    score: float                 # objectness * text similarity
    text_similarity: float       # cosine(image embedding, text embedding), used by FLAME selection
    objectness: float
    embedding: np.ndarray        # [D] projected visual embedding
    image_idx: int               # which image this came from
    patch_idx: int               # which patch produced this


class FlamePipeline:
    """
    FLAME cascaded pipeline for adapting OWLv2 to specific domains.

    Usage:
        model = OwlV2("base")
        flame = FlamePipeline(model)

        # 1. Generate proposals from a set of images
        flame.generate_proposals(images, text_query="chimney")

        # 2. Select informative samples for annotation
        candidates = flame.select_annotation_candidates()

        # 3. User labels the candidates (list of True/False)
        labels = [True, False, True, ...]  # user provides these
        flame.label_candidates(candidates, labels)

        # 4. Train the refiner
        flame.train_refiner()

        # 5. Run refined detection on new images
        results = flame.detect(new_images, text_query="chimney")
    """

    def __init__(self, model: OwlV2, config: Optional[FlameConfig] = None):
        self.model = model
        self.model.eval()
        self.config = config or FlameConfig()

        self.proposals: list[FlameProposal] = []
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.refiner = None
        self._refiner_scaler: Optional[StandardScaler] = None
        self._candidate_indices: list[int] = []
        self._train_embeddings: Optional[np.ndarray] = None
        self._train_labels: Optional[np.ndarray] = None
        self._detection_proposals: list[FlameProposal] = []
        # Cached *classifier-space* embeddings (raw visual embeddings) for the
        # detections returned by the most recent detect() call. Used by
        # add_feedback_labels() so feedback features match training features.
        self._detection_classifier_embeddings: Optional[np.ndarray] = None

    @torch.no_grad()
    def generate_proposals(
        self,
        images: list,
        text_query: str,
        image_sizes: Optional[list[tuple[int, int]]] = None,
    ) -> list[FlameProposal]:
        """
        Stage 1: Run OWLv2 zero-shot detection with a low threshold to get
        high-recall proposals. Extract embeddings for each proposal.

        Args:
            images: list of PIL Images or file paths
            text_query: the text class to detect (e.g. "chimney")
            image_sizes: optional list of (H, W) original sizes for box rescaling
        """
        from OWLv2torch.utils.tokenizer import tokenize

        self.proposals = []
        if not images:
            return self.proposals

        device = next(self.model.parameters()).device

        token_ids = tokenize([text_query], context_length=16, truncate=True).to(device)
        attention_mask = (token_ids == 0).to(device)

        pixel_values = self.model.preprocess_image(
            images,
            fast=self.config.fast_image_preprocess,
        ).to(device)

        use_autocast = self.config.use_autocast and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast):
            pred_logits, objectness_logits, pred_boxes, class_embeds, extras = \
                self.model.forward_object_detection(pixel_values, token_ids, attention_mask)
            text_embeds = extras[1]

            if image_sizes is None:
                from PIL import Image as PILImage

                image_sizes = []
                for image in images:
                    pil_img = PILImage.open(image) if isinstance(image, str) else image
                    image_sizes.append((pil_img.height, pil_img.width))

            img_size = torch.tensor(image_sizes, dtype=torch.float32, device=device)
            boxes_xyxy = self.model.postprocess_boxes(pred_boxes.float(), img_size)  # [B, P*P, 4]

        objectness_logits = objectness_logits.float()
        pred_logits = pred_logits.float()
        class_embeds = class_embeds.float()
        text_embeds = text_embeds.float()
        obj_scores = torch.sigmoid(objectness_logits)       # [B, P*P]
        text_scores = pred_logits[:, :, 0]                  # [B, P*P]
        text_probs = torch.sigmoid(text_scores)
        combined_scores = text_probs * obj_scores
        text_cosines = torch.einsum("bpd,bd->bp", class_embeds, text_embeds[:, 0, :])

        for img_idx in range(len(images)):
            mask = (
                (obj_scores[img_idx] > self.config.objectness_threshold)
                & (text_probs[img_idx] >= self.config.text_score_threshold)
            )
            indices = torch.where(mask)[0]

            if len(indices) > self.config.max_proposals_per_image:
                topk = torch.topk(
                    combined_scores[img_idx, indices],
                    self.config.max_proposals_per_image,
                )
                indices = indices[topk.indices]

            boxes_np = boxes_xyxy[img_idx, indices].detach().cpu().numpy()
            scores_np = combined_scores[img_idx, indices].detach().cpu().numpy()
            text_np = text_cosines[img_idx, indices].detach().cpu().numpy()
            obj_np = obj_scores[img_idx, indices].detach().cpu().numpy()
            embeddings_np = class_embeds[img_idx, indices].detach().cpu().numpy()

            for j, patch_idx in enumerate(indices.detach().cpu().numpy()):
                proposal = FlameProposal(
                    box=boxes_np[j],
                    score=float(scores_np[j]),
                    text_similarity=float(text_np[j]),
                    objectness=float(obj_np[j]),
                    embedding=embeddings_np[j],
                    image_idx=img_idx,
                    patch_idx=int(patch_idx),
                )
                self.proposals.append(proposal)

        if self.config.verbose:
            print(f"[FLAME] Generated {len(self.proposals)} proposals from {len(images)} images")
        return self.proposals

    def _build_augmented_embeddings(self, proposals: Optional[list[FlameProposal]] = None) -> np.ndarray:
        """
        Build augmented embeddings (visual embedding ⊕ text cosine similarity).

        These are used **only** for the active-selection step (PCA + KDE), per
        Algorithm 1, lines 1–4 of the paper. The classifier itself is trained
        on raw visual embeddings — see ``_build_classifier_embeddings``.
        """
        if proposals is None:
            proposals = self.proposals

        embeddings = np.stack([p.embedding for p in proposals])          # [N, D]
        text_sims = np.array([p.text_similarity for p in proposals])[:, None]  # [N, 1]
        augmented = np.concatenate([embeddings, text_sims], axis=1)      # [N, D+1]
        return augmented

    def _build_classifier_embeddings(self, proposals: Optional[list[FlameProposal]] = None) -> np.ndarray:
        """
        Build the feature matrix used to train and query the refiner.

        Per the paper's lemmas, the classifier operates on the raw visual
        embeddings ``xi`` (not the augmented ``[xi, ci]`` used for selection).
        Including the text-similarity scalar here would let the refiner re-fit
        the original detector's score, defeating the purpose of correcting it.
        """
        if proposals is None:
            proposals = self.proposals
        return np.stack([p.embedding for p in proposals])  # [N, D]

    def select_annotation_candidates(self) -> list[int]:
        """
        Stage 3: FLAME active sample selection.
        Returns indices into self.proposals of the K most informative samples.

        Steps:
            1. Build augmented embeddings (visual + text similarity)
            2. Standardize + PCA projection
            3. Kernel density estimation
            4. Select marginal (uncertain) samples from the density tails
            5. Cluster marginals and pick one per cluster for diversity
        """
        cfg = self.config

        if len(self.proposals) < cfg.num_clusters:
            # Not enough proposals — just return all
            self._candidate_indices = list(range(len(self.proposals)))
            return self._candidate_indices

        # Step 1: Augmented embeddings
        augmented = self._build_augmented_embeddings()

        # Step 2: Standardize + PCA
        self.scaler = StandardScaler()
        augmented_scaled = self.scaler.fit_transform(augmented)

        n_components = min(cfg.pca_components, augmented_scaled.shape[1], augmented_scaled.shape[0])
        self.pca = PCA(n_components=n_components)
        projected = self.pca.fit_transform(augmented_scaled)  # [N, pca_components]

        # Step 3: Kernel density estimation
        kde = KernelDensity(bandwidth=cfg.kde_bandwidth, kernel='gaussian')
        kde.fit(projected)
        log_densities = kde.score_samples(projected)  # [N]

        # Step 4: Find marginal samples — the band of points whose density
        # lies between rl*f̂(s*) and ru*f̂(s*), where s* = argmax f̂(s).
        # Working in log-space avoids underflow: log f̂(sL) = log(rl) + log f̂(s*).
        rl = float(cfg.marginal_density_ratio_low)
        ru = float(cfg.marginal_density_ratio_high)
        if not (0.0 < rl < ru < 1.0):
            raise ValueError(
                "marginal_density_ratio_low/high must satisfy 0 < rl < ru < 1, "
                f"got rl={rl}, ru={ru}"
            )

        log_mode = float(log_densities.max())
        log_lower = np.log(rl) + log_mode  # outer band edge (lower density)
        log_upper = np.log(ru) + log_mode  # inner band edge (closer to mode)
        marginal_mask = (log_densities >= log_lower) & (log_densities <= log_upper)
        marginal_indices = np.where(marginal_mask)[0]

        if len(marginal_indices) < cfg.num_clusters:
            # Fallback: relax the band but still prefer points whose density is
            # closest to the band rather than absolute outliers. Rank by
            # distance (in log-density) to the band midpoint and take the top-N.
            log_band_mid = 0.5 * (log_lower + log_upper)
            dist_to_band = np.abs(log_densities - log_band_mid)
            n_take = min(max(cfg.num_clusters * 3, cfg.num_clusters), len(log_densities))
            marginal_indices = np.argsort(dist_to_band)[:n_take]

        # Step 5: Cluster marginals for diversity in the original visual
        # embedding space, matching Algorithm 1's k-means on X_marginal.
        n_clusters = min(cfg.num_clusters, len(marginal_indices))
        classifier_features = self._build_classifier_embeddings()
        marginal_features = classifier_features[marginal_indices]

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(marginal_features)

        selected = []
        for c in range(n_clusters):
            cluster_members = marginal_indices[cluster_labels == c]
            # Pick the one closest to the cluster center
            center = kmeans.cluster_centers_[c]
            dists = np.linalg.norm(marginal_features[cluster_labels == c] - center, axis=1)
            best = cluster_members[np.argmin(dists)]
            selected.append(int(best))

        self._candidate_indices = selected
        if self.config.verbose:
            print(f"[FLAME] Selected {len(selected)} annotation candidates from {len(self.proposals)} proposals")
        return selected

    def get_candidate_crops(
        self,
        images: list,
        candidate_indices: Optional[list[int]] = None,
    ) -> list[tuple[int, np.ndarray]]:
        """
        Utility: extract image crops for the selected candidates so the user
        can view and label them.

        Returns list of (candidate_index, crop_array) tuples.
        """
        from PIL import Image as PILImage

        if candidate_indices is None:
            candidate_indices = self._candidate_indices

        crops = []
        # Cache loaded images
        loaded = {}
        for idx in candidate_indices:
            p = self.proposals[idx]
            if p.image_idx not in loaded:
                img = images[p.image_idx]
                if isinstance(img, str):
                    img = PILImage.open(img)
                loaded[p.image_idx] = np.array(img)

            img_arr = loaded[p.image_idx]
            x1, y1, x2, y2 = p.box.astype(int)
            h, w = img_arr.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = img_arr[y1:y2, x1:x2]
            crops.append((idx, crop))

        return crops

    def label_candidates(self, candidate_indices: list[int], labels: list[bool]):
        """
        Stage 4: Receive user annotations for the selected candidates.

        Args:
            candidate_indices: indices into self.proposals
            labels: True = positive (correct detection), False = negative (false positive)
        """
        classifier_features = self._build_classifier_embeddings()
        self._append_training_samples(classifier_features[candidate_indices], labels)
        self._print_training_label_stats(len(labels), prefix="Labeled")

    def _append_training_samples(self, embeddings: np.ndarray, labels: list[bool] | np.ndarray):
        """Append new labeled samples to the accumulated training set."""
        labels_array = np.asarray(labels, dtype=np.int32)
        embeddings = np.asarray(embeddings)

        if len(labels_array) != len(embeddings):
            raise ValueError("embeddings and labels must have the same length")

        if len(labels_array) == 0:
            return

        if self._train_embeddings is None or self._train_labels is None:
            self._train_embeddings = embeddings.copy()
            self._train_labels = labels_array.copy()
            return

        self._train_embeddings = np.concatenate([self._train_embeddings, embeddings], axis=0)
        self._train_labels = np.concatenate([self._train_labels, labels_array], axis=0)

    def _print_training_label_stats(self, num_added: int, prefix: str):
        """Log training label counts after new labels are appended."""
        if self._train_labels is None:
            print(f"[FLAME] {prefix} 0 candidates")
            return

        n_pos = int(self._train_labels.sum())
        n_neg = int(len(self._train_labels) - n_pos)
        print(
            f"[FLAME] {prefix} {num_added} candidates: "
            f"{n_pos} positive, {n_neg} negative total"
        )

    def add_feedback_labels(self, detection_indices: list[int], labels: list[bool]):
        """
        Append labels for detections returned by the most recent detect() call.

        Args:
            detection_indices: indices into the flattened detections from the most
                recent detect() call, as returned in each result's
                'detection_indices' field.
            labels: True = positive (correct detection), False = negative (false positive)
        """
        if self._detection_classifier_embeddings is None:
            raise RuntimeError("No detection feedback available. Call detect() first.")

        if len(detection_indices) != len(labels):
            raise ValueError("detection_indices and labels must have the same length")

        if len(detection_indices) == 0:
            return

        max_index = len(self._detection_classifier_embeddings) - 1
        if min(detection_indices) < 0 or max(detection_indices) > max_index:
            raise IndexError("detection index out of range for the most recent detect() call")

        feedback_embeddings = self._detection_classifier_embeddings[detection_indices]
        self._append_training_samples(feedback_embeddings, labels)
        self._print_training_label_stats(len(labels), prefix="Added feedback labels for")

    def train_refiner(self):
        """
        Stage 5: Train the lightweight refiner classifier on labeled samples.
        Optionally uses SMOTE for class balancing (if imbalanced-learn is available).
        """
        cfg = self.config

        if self._train_embeddings is None or self._train_labels is None:
            raise RuntimeError("No labeled data. Call label_candidates() first.")

        X = self._train_embeddings.copy()
        y = self._train_labels.copy()

        # Try SMOTE if classes are imbalanced and imblearn is available
        n_pos, n_neg = y.sum(), len(y) - y.sum()
        if n_pos > 0 and n_neg > 0 and min(n_pos, n_neg) / max(n_pos, n_neg) < 0.5:
            try:
                from imblearn.over_sampling import SVMSMOTE
                smote = SVMSMOTE(random_state=42, k_neighbors=min(3, min(n_pos, n_neg) - 1))
                X, y = smote.fit_resample(X, y)
                print(f"[FLAME] Applied SVM-SMOTE: {len(y)} samples ({y.sum()} pos, {len(y) - y.sum()} neg)")
            except ImportError:
                print("[FLAME] imbalanced-learn not installed, skipping SMOTE")

        # Standardize
        self._refiner_scaler = StandardScaler()
        X_scaled = self._refiner_scaler.fit_transform(X)

        # Train classifier
        if cfg.classifier_type == "svm":
            # No probability=True: Platt-scaling on ~30 points yields an
            # uncalibrated predict_proba that destroys ranking quality
            # (which is what COCO mAP measures). We rank by decision_function
            # at inference time instead — see detect().
            # class_weight="balanced" handles label skew without resorting to
            # SMOTE for small imbalances.
            self.refiner = SVC(
                kernel=cfg.svm_kernel,
                C=cfg.svm_C,
                class_weight="balanced",
                random_state=42,
            )
        elif cfg.classifier_type == "mlp":
            self.refiner = MLPClassifier(
                hidden_layer_sizes=cfg.mlp_hidden,
                max_iter=cfg.mlp_max_iter,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown classifier_type: {cfg.classifier_type}")

        self.refiner.fit(X_scaled, y)
        train_acc = self.refiner.score(X_scaled, y)
        print(f"[FLAME] Trained {cfg.classifier_type} refiner — training accuracy: {train_acc:.3f}")

    def _refiner_scores(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Map refiner outputs to a [0, 1] positive-class score.

        For SVM: sigmoid(decision_function). Monotonic in the SVM's signed
        margin, so AP ranking is identical to ranking by decision_function,
        and a threshold of 0.5 corresponds to decision_function >= 0
        (the SVM's natural decision boundary).

        For MLP: predict_proba[:, 1].
        """
        if isinstance(self.refiner, SVC):
            margins = self.refiner.decision_function(X_scaled)
            # Numerically stable sigmoid.
            return np.where(
                margins >= 0,
                1.0 / (1.0 + np.exp(-margins)),
                np.exp(margins) / (1.0 + np.exp(margins)),
            )
        return self.refiner.predict_proba(X_scaled)[:, 1]

    @torch.no_grad()
    def detect(
        self,
        images: list,
        text_query: str,
        image_sizes: Optional[list[tuple[int, int]]] = None,
    ) -> list[dict]:
        """
        Stage 6: Run the full FLAME cascade on new images.

        1. OWLv2 generates high-recall proposals
        2. Refiner classifier filters false positives

        Returns:
            List of dicts per image, each with keys:
                'boxes': np.ndarray [N, 4] xyxy
                'scores': np.ndarray [N]
                'refiner_scores': np.ndarray [N]  (probability from refiner)
                'proposal_indices': np.ndarray [N] (indices into detect() proposals)
                'detection_indices': np.ndarray [N] (indices for add_feedback_labels())
        """
        if self.refiner is None:
            raise RuntimeError("Refiner not trained. Call train_refiner() first.")

        # Generate proposals (reuse the same method)
        old_proposals = self.proposals
        proposals = self.generate_proposals(images, text_query, image_sizes)

        num_images = len(images)
        results = [
            {
                "boxes": [],
                "scores": [],
                "refiner_scores": [],
                "proposal_indices": [],
                "detection_indices": [],
            }
            for _ in range(num_images)
        ]

        # Short-circuit: if no proposals survived objectness filtering across
        # the whole batch there's nothing to classify (and ``np.stack([])``
        # would otherwise blow up inside the feature builder).
        if not proposals:
            self._detection_proposals = []
            self._detection_classifier_embeddings = None
            for r in results:
                r["boxes"] = np.zeros((0, 4))
                r["scores"] = np.zeros(0)
                r["refiner_scores"] = np.zeros(0)
                r["proposal_indices"] = np.zeros(0, dtype=np.int32)
                r["detection_indices"] = np.zeros(0, dtype=np.int32)
            self.proposals = old_proposals
            if self.config.verbose:
                print(f"[FLAME] Refined: 0/0 proposals kept across {num_images} images")
            return results

        # Build classifier features (raw visual embeddings — must match training)
        classifier_features = self._build_classifier_embeddings(proposals)
        X_scaled = self._refiner_scaler.transform(classifier_features)

        # Score proposals. For SVM we deliberately avoid predict_proba (Platt
        # scaling on ~30 points is unreliable) and instead pass the signed
        # margin through a sigmoid: this preserves the SVM's ranking — what
        # COCO mAP actually evaluates — and yields a [0, 1] score that
        # composes cleanly with the original detector score (e.g. via
        # `--score-combine product`). MLP keeps predict_proba.
        refiner_probs = self._refiner_scores(X_scaled)

        self._detection_proposals = []
        kept_features: list[np.ndarray] = []

        for i, (proposal, prob) in enumerate(zip(proposals, refiner_probs)):
            if prob >= self.config.classifier_threshold:
                r = results[proposal.image_idx]
                detection_idx = len(self._detection_proposals)
                r["boxes"].append(proposal.box)
                r["scores"].append(proposal.score)
                r["refiner_scores"].append(prob)
                r["proposal_indices"].append(i)
                r["detection_indices"].append(detection_idx)
                self._detection_proposals.append(proposal)
                kept_features.append(classifier_features[i])

        if kept_features:
            self._detection_classifier_embeddings = np.stack(kept_features)
        else:
            self._detection_classifier_embeddings = None

        # Convert to arrays
        for r in results:
            if r["boxes"]:
                r["boxes"] = np.stack(r["boxes"])
                r["scores"] = np.array(r["scores"])
                r["refiner_scores"] = np.array(r["refiner_scores"])
                r["proposal_indices"] = np.array(r["proposal_indices"], dtype=np.int32)
                r["detection_indices"] = np.array(r["detection_indices"], dtype=np.int32)
            else:
                r["boxes"] = np.zeros((0, 4))
                r["scores"] = np.zeros(0)
                r["refiner_scores"] = np.zeros(0)
                r["proposal_indices"] = np.zeros(0, dtype=np.int32)
                r["detection_indices"] = np.zeros(0, dtype=np.int32)

        # Restore original proposals
        self.proposals = old_proposals

        total_detections = sum(len(r["boxes"]) for r in results)
        total_proposals = len(proposals)
        if self.config.verbose:
            print(f"[FLAME] Refined: {total_detections}/{total_proposals} proposals kept across {num_images} images")

        return results

    def save(self, path: str):
        """Save the trained refiner and associated state."""
        import pickle
        state = {
            "config": self.config,
            "refiner": self.refiner,
            "refiner_scaler": self._refiner_scaler,
            "scaler": self.scaler,
            "pca": self.pca,
            "train_embeddings": self._train_embeddings,
            "train_labels": self._train_labels,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"[FLAME] Saved refiner to {path}")

    def load(self, path: str):
        """Load a previously trained refiner."""
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.config = state["config"]
        self.refiner = state["refiner"]
        self._refiner_scaler = state["refiner_scaler"]
        self.scaler = state["scaler"]
        self.pca = state["pca"]
        self._train_embeddings = state.get("train_embeddings")
        self._train_labels = state.get("train_labels")
        self._detection_proposals = []
        self._detection_classifier_embeddings = None
        print(f"[FLAME] Loaded refiner from {path}")
