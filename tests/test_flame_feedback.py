import types

import numpy as np
import torch

from OWLv2torch.torch_version.flame import FlameConfig, FlamePipeline, FlameProposal


class DummyModel:
    def __init__(self):
        self._param = torch.nn.Parameter(torch.zeros(1))

    def eval(self):
        return self

    def parameters(self):
        return iter([self._param])


class IdentityScaler:
    def transform(self, X):
        return np.asarray(X)


class FixedProbabilityRefiner:
    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=np.float32)

    def predict_proba(self, X):
        X = np.asarray(X)
        assert len(X) == len(self.probs)
        return np.column_stack([1.0 - self.probs, self.probs])


def make_proposal(image_idx: int, embedding: list[float], text_similarity: float, patch_idx: int):
    return FlameProposal(
        box=np.array([patch_idx, patch_idx, patch_idx + 1, patch_idx + 1], dtype=np.float32),
        score=0.5 + 0.1 * patch_idx,
        text_similarity=text_similarity,
        objectness=0.8,
        embedding=np.array(embedding, dtype=np.float32),
        image_idx=image_idx,
        patch_idx=patch_idx,
    )


def test_flame_feedback_labels_append_and_persist(tmp_path):
    pipeline = FlamePipeline(DummyModel(), FlameConfig(classifier_threshold=0.5))

    pipeline.proposals = [
        make_proposal(0, [1.0, 1.5], 0.1, 0),
        make_proposal(0, [2.0, 2.5], 0.2, 1),
    ]
    pipeline.label_candidates([0, 1], [True, False])

    pipeline.proposals = [
        make_proposal(0, [3.0, 3.5], 0.3, 2),
    ]
    pipeline.label_candidates([0], [True])

    assert pipeline._train_embeddings.shape == (3, 3)
    assert np.array_equal(pipeline._train_labels, np.array([1, 0, 1], dtype=np.int32))
    assert np.allclose(
        pipeline._train_embeddings,
        np.array(
            [
                [1.0, 1.5, 0.1],
                [2.0, 2.5, 0.2],
                [3.0, 3.5, 0.3],
            ],
            dtype=np.float32,
        ),
    )

    original_proposals = pipeline.proposals
    detection_proposals = [
        make_proposal(0, [10.0, 10.5], 1.1, 3),
        make_proposal(0, [20.0, 20.5], 1.2, 4),
        make_proposal(1, [30.0, 30.5], 1.3, 5),
    ]

    def fake_generate_proposals(self, images, text_query, image_sizes=None):
        self.proposals = detection_proposals
        return detection_proposals

    pipeline.generate_proposals = types.MethodType(fake_generate_proposals, pipeline)
    pipeline._refiner_scaler = IdentityScaler()
    pipeline.refiner = FixedProbabilityRefiner([0.9, 0.2, 0.8])

    results = pipeline.detect(["image-0", "image-1"], "chimney")

    assert pipeline.proposals is original_proposals
    assert len(pipeline._detection_proposals) == 2
    assert np.array_equal(results[0]["proposal_indices"], np.array([0], dtype=np.int32))
    assert np.array_equal(results[0]["detection_indices"], np.array([0], dtype=np.int32))
    assert np.array_equal(results[1]["proposal_indices"], np.array([2], dtype=np.int32))
    assert np.array_equal(results[1]["detection_indices"], np.array([1], dtype=np.int32))

    pipeline.add_feedback_labels([1], [False])

    assert pipeline._train_embeddings.shape == (4, 3)
    assert np.array_equal(pipeline._train_labels, np.array([1, 0, 1, 0], dtype=np.int32))
    assert np.allclose(pipeline._train_embeddings[-1], np.array([30.0, 30.5, 1.3], dtype=np.float32))

    save_path = tmp_path / "flame.pkl"
    pipeline.save(str(save_path))

    loaded = FlamePipeline(DummyModel())
    loaded.load(str(save_path))

    assert np.allclose(loaded._train_embeddings, pipeline._train_embeddings)
    assert np.array_equal(loaded._train_labels, pipeline._train_labels)
