import math

import numpy as np
import pytest
import torch
from sklearn.svm import SVC

from OWLv2torch import Flame
from OWLv2torch.torch_version.flame_features import balanced_landmarks, fit_approximate, rbf_kernel


def examples():
    rng = np.random.default_rng(10)
    x = torch.from_numpy(rng.normal(size=(180, 5))).float()
    y = (x[:, 0].square() + x[:, 1].square() < 1).long()
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    held_out = torch.from_numpy(rng.normal(size=(80, 5))).float()
    return x, y, held_out


def test_landmarks_preserve_rare_class_and_seed():
    y = torch.cat((torch.ones(7), torch.zeros(1000)))
    selected = balanced_landmarks(y, 32, 17)
    assert len(selected.unique()) == 32
    assert int(y[selected].sum()) == 7
    assert torch.equal(selected, balanced_landmarks(y, 32, 17))
    assert not torch.equal(selected, balanced_landmarks(y, 32, 18))


@pytest.mark.parametrize("backend", ["nystrom", "rff"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device unavailable"))])
def test_fitted_margins_match_svc_in_the_same_feature_space(backend, device):
    torch.set_num_threads(4)
    x, y, held_out = examples()
    gamma = 1 / (x.shape[1] * float(x.double().var(correction=0)))
    count = len(x) if backend == "nystrom" else 256
    fitted = fit_approximate(x, y, backend=backend, gamma=gamma, c=1,
                             n_components=count, device=torch.device(device), seed=0, max_iter=400)
    if backend == "nystrom":
        reference = SVC(gamma=gamma, class_weight="balanced").fit(x.numpy(), y.numpy())
        expected = reference.decision_function(held_out.numpy())
        actual = rbf_kernel(held_out, fitted["vectors"], gamma) @ fitted["coef"] + fitted["bias"]
    else:
        def transform(value):
            return torch.cos(value @ fitted["frequencies"] + fitted["phases"]) * math.sqrt(2 / count)

        reference = SVC(kernel="linear", class_weight="balanced").fit(transform(x).numpy(), y.numpy())
        expected = reference.decision_function(transform(held_out).numpy())
        actual = (torch.cos(held_out @ fitted["frequencies"] + fitted["phases"])
                  @ fitted["coef"] + fitted["bias"])
    assert np.abs(actual.numpy() - expected).mean() < 0.025
    assert np.mean((actual.numpy() > 0) != (expected > 0)) < 0.04


class TextOnlyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def get_text_features(self, tokens, mask):
        return torch.zeros(len(tokens), 5, device=tokens.device)


@pytest.mark.parametrize("backend", ["sklearn", "nystrom", "rff"])
def test_flame_scores_and_saved_backend_round_trip(tmp_path, backend):
    x, y, held_out = examples()
    model = TextOnlyModel()
    flame = Flame(model, ["object", "unlabelled"], svm_backend=backend,
                  n_components=64, svm_seed=23, svm_max_iter=200)
    flame.examples[0] = [(x.half(), y.to(torch.int8))]
    flame.fit()
    logits, objectness = torch.zeros(len(held_out), 2), torch.ones(len(held_out))
    scores = flame.scores(held_out, logits, objectness)
    assert torch.allclose(scores[:, 1], torch.sigmoid(objectness) / 2)
    path = tmp_path / "flame.pt"
    flame.save(path)
    restored = Flame.load(path, model)
    assert restored.svm_backend == backend
    assert restored.n_components == 64
    assert restored.svm_seed == 23
    assert torch.allclose(scores, restored.scores(held_out, logits, objectness), atol=1e-6)


def test_load_older_flame_checkpoint(tmp_path):
    x, y, _ = examples()
    path = tmp_path / "legacy.pt"
    torch.save({"queries": ["object"], "alpha": 8.0, "svm_c": 1.0,
                "examples": [[(x.half(), y.to(torch.int8))]]}, path)
    restored = Flame.load(path, TextOnlyModel())
    assert restored.svm_backend == "sklearn"
    assert restored.svms[0] is not None
