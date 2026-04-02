from typing import Optional

import torch


def normalize_detection_queries(
    token_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    batch_size: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor], int]:
    """
    Normalize detection queries to the flattened [batch_size * num_queries, seq_len]
    layout expected by the OWLv2 detection heads.

    A 2D query tensor whose first dimension is not divisible by batch_size is treated
    as a shared query set and repeated for each image in the batch. Callers with an
    explicitly batched query tensor can pass [batch_size, num_queries, seq_len] to
    avoid ambiguity when a shared query count is divisible by batch_size.
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

    if attention_mask is not None and attention_mask.shape != token_ids.shape:
        raise ValueError(
            f"attention_mask shape {tuple(attention_mask.shape)} does not match "
            f"token_ids shape {tuple(token_ids.shape)}"
        )

    num_query_rows = token_ids.shape[0]
    if num_query_rows == 0:
        raise ValueError("token_ids must contain at least one query")

    if num_query_rows % batch_size == 0:
        return token_ids, attention_mask, num_query_rows // batch_size

    token_ids = token_ids.repeat(batch_size, 1)
    if attention_mask is not None:
        attention_mask = attention_mask.repeat(batch_size, 1)
    return token_ids, attention_mask, num_query_rows
