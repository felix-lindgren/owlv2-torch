import torch
import torch.nn.functional as F

from OWLv2torch.torch_version.loss import _hard_negative_indices, _objectness_loss


def test_hard_negative_indices_keep_top_detection_scores():
    class_logits = torch.tensor(
        [
            [8.0, -8.0],
            [9.0, 9.0],
            [4.0, -4.0],
            [2.0, -2.0],
            [-8.0, -8.0],
        ]
    )
    objectness_logits = torch.tensor([-8.0, 9.0, 4.0, 6.0, 8.0])
    matched_indices = torch.tensor([1])

    indices = _hard_negative_indices(
        class_logits,
        objectness_logits,
        matched_indices,
        num_positives=1,
        negative_ratio=2,
        max_negatives=512,
    )

    assert indices.tolist() == [2, 3]


def test_objectness_loss_uses_positives_and_mined_negatives_only():
    objectness_logits = torch.tensor([20.0, 0.5, -20.0, -0.5])
    positive_indices = torch.tensor([1])
    negative_indices = torch.tensor([3])

    loss = _objectness_loss(objectness_logits, positive_indices, negative_indices)

    expected = F.binary_cross_entropy_with_logits(
        torch.tensor([0.5, -0.5]),
        torch.tensor([1.0, 0.0]),
    )
    assert torch.allclose(loss, expected)
