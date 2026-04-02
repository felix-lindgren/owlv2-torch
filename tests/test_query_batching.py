import types

import torch

from OWLv2torch.torch_version.owlv2 import OwlV2
from OWLv2torch.utils.query_batching import normalize_detection_queries


def test_normalize_detection_queries_repeats_shared_queries_for_batch():
    token_ids = torch.arange(16, dtype=torch.long).reshape(1, 16)
    attention_mask = token_ids == 0

    flat_token_ids, flat_attention_mask, num_queries = normalize_detection_queries(
        token_ids, attention_mask, batch_size=4
    )

    assert num_queries == 1
    assert flat_token_ids.shape == (4, 16)
    assert flat_attention_mask.shape == (4, 16)
    assert torch.equal(flat_token_ids, token_ids.repeat(4, 1))
    assert torch.equal(flat_attention_mask, attention_mask.repeat(4, 1))


def test_normalize_detection_queries_preserves_flattened_per_image_queries():
    token_ids = torch.arange(4 * 2 * 8, dtype=torch.long).reshape(8, 8)
    attention_mask = token_ids == 0

    flat_token_ids, flat_attention_mask, num_queries = normalize_detection_queries(
        token_ids, attention_mask, batch_size=4
    )

    assert num_queries == 2
    assert torch.equal(flat_token_ids, token_ids)
    assert torch.equal(flat_attention_mask, attention_mask)


def test_normalize_detection_queries_flattens_explicit_batched_queries():
    token_ids = torch.arange(4 * 3 * 8, dtype=torch.long).reshape(4, 3, 8)
    attention_mask = token_ids == 0

    flat_token_ids, flat_attention_mask, num_queries = normalize_detection_queries(
        token_ids, attention_mask, batch_size=4
    )

    assert num_queries == 3
    assert flat_token_ids.shape == (12, 8)
    assert flat_attention_mask.shape == (12, 8)
    assert torch.equal(flat_token_ids, token_ids.reshape(12, 8))
    assert torch.equal(flat_attention_mask, attention_mask.reshape(12, 8))


def test_forward_object_detection_accepts_shared_queries_for_batched_images():
    batch_size = 4
    seq_len = 16
    hidden_dim = 8
    text_dim = 6
    num_patches = 2
    num_locations = num_patches * num_patches

    class DummyClassHead:
        def __call__(self, image_feats, text_features, query_mask):
            pred_logits = torch.zeros(
                image_feats.shape[0], image_feats.shape[1], text_features.shape[1]
            )
            return pred_logits, image_feats

    class DummyObjectnessHead:
        def __call__(self, image_feats):
            return torch.zeros(image_feats.shape[0], image_feats.shape[1], 1)

    class DummyBoxHead:
        def __call__(self, image_feats):
            return torch.zeros(image_feats.shape[0], image_feats.shape[1], 4)

    dummy = types.SimpleNamespace(
        sqrt_num_patches=num_patches,
        vision_model=types.SimpleNamespace(post_layernorm=torch.nn.Identity()),
        layer_norm=torch.nn.Identity(),
        class_head=DummyClassHead(),
        objectness_head=DummyObjectnessHead(),
        box_head=DummyBoxHead(),
        box_bias=torch.zeros(num_locations, 4),
    )

    def fake_forward(self, pixel_values, token_ids, attention_mask):
        assert token_ids.shape == (batch_size, seq_len)
        vision_full = torch.zeros(batch_size, 1 + num_locations, hidden_dim)
        text_features = torch.zeros(batch_size, text_dim)
        return None, None, None, text_features, vision_full, text_features

    dummy.forward = types.MethodType(fake_forward, dummy)

    pixel_values = torch.zeros(batch_size, 3, 32, 32)
    token_ids = torch.ones(1, seq_len, dtype=torch.long)
    attention_mask = token_ids == 0

    pred_logits, objectness_logits, pred_boxes, class_embeds, _ = OwlV2.forward_object_detection(
        dummy, pixel_values, token_ids, attention_mask
    )

    assert pred_logits.shape == (batch_size, num_locations, 1)
    assert objectness_logits.shape == (batch_size, num_locations)
    assert pred_boxes.shape == (batch_size, num_locations, 4)
    assert class_embeds.shape == (batch_size, num_locations, hidden_dim)
