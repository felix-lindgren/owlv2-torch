from typing import Optional

import torch


def normalize_detection_queries(
    token_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    batch_size: int,
    *,
    repeat_shared: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor], int]:
    """
    Normalize detection-query layout and optionally flatten/repeat it to
    ``[batch_size * num_queries, seq_len]``.

    A 2D query tensor is always a shared query set. A 3D query tensor is always an
    explicitly batched query set. This avoids silently interpreting shared queries
    as per-image queries when ``num_queries`` happens to be divisible by
    ``batch_size``.

    Set ``repeat_shared=False`` when shared queries will be encoded once and the
    resulting embeddings expanded across the image batch. The default retains the
    flattened output layout for lower-level callers.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if token_ids.ndim not in (2, 3):
        raise ValueError(
            f"token_ids must have shape [num_queries, seq_len] or "
            f"[batch_size, num_queries, seq_len], got {tuple(token_ids.shape)}"
        )

    if token_ids.ndim == 3:
        if token_ids.shape[0] != batch_size:
            raise ValueError(
                f"token_ids batch dimension {token_ids.shape[0]} does not match "
                f"pixel batch size {batch_size}"
            )
        num_queries = token_ids.shape[1]
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                if attention_mask.shape != token_ids.shape[1:]:
                    raise ValueError(
                        f"attention_mask shape {tuple(attention_mask.shape)} does not match "
                        f"shared query shape {tuple(token_ids.shape[1:])}"
                    )
                attention_mask = attention_mask.unsqueeze(0).expand(batch_size, -1, -1)
            elif attention_mask.ndim != 3 or attention_mask.shape != token_ids.shape:
                raise ValueError(
                    f"attention_mask must match token_ids shape {tuple(token_ids.shape)}, "
                    f"got {tuple(attention_mask.shape)}"
                )
            attention_mask = attention_mask.reshape(batch_size * num_queries, token_ids.shape[-1])
        return token_ids.reshape(batch_size * num_queries, token_ids.shape[-1]), attention_mask, num_queries

    if attention_mask is not None:
        same_token_shape = (
            attention_mask.ndim == 2 and attention_mask.shape == token_ids.shape
        )
        additive_attention_shape = (
            attention_mask.ndim >= 3
            and attention_mask.shape[0] == token_ids.shape[0]
            and attention_mask.shape[-2:] == (token_ids.shape[1], token_ids.shape[1])
        )
        if not same_token_shape and not additive_attention_shape:
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} is incompatible "
                f"with token_ids shape {tuple(token_ids.shape)}"
            )

    num_query_rows = token_ids.shape[0]
    if num_query_rows == 0:
        raise ValueError("token_ids must contain at least one query")

    if repeat_shared:
        token_ids = token_ids.repeat(batch_size, 1)
        if attention_mask is not None:
            repeats = (batch_size,) + (1,) * (attention_mask.ndim - 1)
            attention_mask = attention_mask.repeat(repeats)
    return token_ids, attention_mask, num_query_rows
