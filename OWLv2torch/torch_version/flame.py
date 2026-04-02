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
    text_score_threshold: float = 0.0       # keep all text-matched proposals
    max_proposals_per_image: int = 500      # cap proposals per image

    # Active sample selection
    pca_components: int = 32                # PCA dimensionality for density estimation
    kde_bandwidth: float = 0.5              # KDE bandwidth
    marginal_quantile_low: float = 0.05     # lower density quantile for marginal band
    marginal_quantile_high: float = 0.40    # upper density quantile for marginal band
    num_clusters: int = 30                  # number of clusters (= num shots to annotate)

    # Classifier
    classifier_type: str = "svm"            # "svm" or "mlp"
    svm_kernel: str = "rbf"
    svm_C: float = 10.0
    mlp_hidden: tuple = (128, 64)
    mlp_max_iter: int = 500

    # Inference
    classifier_threshold: float = 0.5       # threshold for the refiner classifier


@dataclass
class FlameProposal:
    """A single detection proposal with its features."""
    box: np.ndarray              # [4] xyxy in pixel coords
    score: float                 # objectness * text similarity
    text_similarity: float       # raw text similarity logit
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
        self._detection_augmented_embeddings: Optional[np.ndarray] = None

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
        device = next(self.model.parameters()).device

        token_ids = tokenize([text_query], context_length=16, truncate=True).to(device)
        attention_mask = (token_ids == 0).to(device)

        for img_idx, image in enumerate(images):
            # Preprocess
            pixel_values = self.model.preprocess_image(image).to(device)

            # Run detection
            pred_logits, objectness_logits, pred_boxes, class_embeds, _ = \
                self.model.forward_object_detection(pixel_values, token_ids, attention_mask)

            # pred_logits: [1, P*P, 1], objectness: [1, P*P], boxes: [1, P*P, 4]
            # class_embeds: [1, P*P, D]
            text_scores = pred_logits[0, :, 0]             # [P*P]
            obj_scores = torch.sigmoid(objectness_logits[0])  # [P*P]
            combined_scores = torch.sigmoid(text_scores) * obj_scores

            # Get image size for box postprocessing
            if image_sizes is not None:
                img_size = torch.tensor([image_sizes[img_idx]], dtype=torch.float32)
            else:
                from PIL import Image as PILImage
                if isinstance(image, str):
                    pil_img = PILImage.open(image)
                else:
                    pil_img = image
                img_size = torch.tensor([[pil_img.height, pil_img.width]], dtype=torch.float32)

            # Postprocess boxes to xyxy pixel coords
            boxes_xyxy = self.model.postprocess_boxes(pred_boxes, img_size.to(device))  # [1, P*P, 4]
            boxes_xyxy = boxes_xyxy[0]  # [P*P, 4]

            # Filter by objectness threshold
            mask = obj_scores > self.config.objectness_threshold
            indices = torch.where(mask)[0]

            # Cap proposals
            if len(indices) > self.config.max_proposals_per_image:
                topk = torch.topk(combined_scores[indices], self.config.max_proposals_per_image)
                indices = indices[topk.indices]

            embeddings = class_embeds[0]  # [P*P, D] — projected image features

            for idx in indices:
                i = idx.item()
                proposal = FlameProposal(
                    box=boxes_xyxy[i].cpu().numpy(),
                    score=combined_scores[i].item(),
                    text_similarity=text_scores[i].item(),
                    objectness=obj_scores[i].item(),
                    embedding=embeddings[i].cpu().numpy(),
                    image_idx=img_idx,
                    patch_idx=i,
                )
                self.proposals.append(proposal)

        print(f"[FLAME] Generated {len(self.proposals)} proposals from {len(images)} images")
        return self.proposals

    def _build_augmented_embeddings(self, proposals: Optional[list[FlameProposal]] = None) -> np.ndarray:
        """
        Stage 2: Build augmented embeddings by concatenating visual embedding
        with text similarity score.
        """
        if proposals is None:
            proposals = self.proposals

        embeddings = np.stack([p.embedding for p in proposals])          # [N, D]
        text_sims = np.array([p.text_similarity for p in proposals])[:, None]  # [N, 1]
        augmented = np.concatenate([embeddings, text_sims], axis=1)      # [N, D+1]
        return augmented

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

        # Step 4: Find marginal samples (between quantile thresholds)
        low_q = np.quantile(log_densities, cfg.marginal_quantile_low)
        high_q = np.quantile(log_densities, cfg.marginal_quantile_high)
        marginal_mask = (log_densities >= low_q) & (log_densities <= high_q)
        marginal_indices = np.where(marginal_mask)[0]

        if len(marginal_indices) < cfg.num_clusters:
            # Expand marginal band if too few samples
            sorted_indices = np.argsort(log_densities)
            n_take = min(cfg.num_clusters * 3, len(sorted_indices))
            marginal_indices = sorted_indices[:n_take]

        # Step 5: Cluster marginals for diversity, pick one per cluster
        n_clusters = min(cfg.num_clusters, len(marginal_indices))
        marginal_features = projected[marginal_indices]

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
        augmented = self._build_augmented_embeddings()
        self._append_training_samples(augmented[candidate_indices], labels)
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
        if self._detection_augmented_embeddings is None:
            raise RuntimeError("No detection feedback available. Call detect() first.")

        if len(detection_indices) != len(labels):
            raise ValueError("detection_indices and labels must have the same length")

        if len(detection_indices) == 0:
            return

        max_index = len(self._detection_augmented_embeddings) - 1
        if min(detection_indices) < 0 or max(detection_indices) > max_index:
            raise IndexError("detection index out of range for the most recent detect() call")

        feedback_embeddings = self._detection_augmented_embeddings[detection_indices]
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
            self.refiner = SVC(
                kernel=cfg.svm_kernel,
                C=cfg.svm_C,
                probability=True,
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

        # Build augmented embeddings
        augmented = self._build_augmented_embeddings(proposals)
        X_scaled = self._refiner_scaler.transform(augmented)

        # Classify
        refiner_probs = self.refiner.predict_proba(X_scaled)[:, 1]  # P(positive)

        self._detection_proposals = []
        self._detection_augmented_embeddings = []

        # Group results by image
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
                self._detection_augmented_embeddings.append(augmented[i])

        if self._detection_augmented_embeddings:
            self._detection_augmented_embeddings = np.stack(self._detection_augmented_embeddings)
        else:
            self._detection_augmented_embeddings = None

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
        self._detection_augmented_embeddings = None
        print(f"[FLAME] Loaded refiner from {path}")
