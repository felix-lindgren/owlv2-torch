"""Batched rectangular linear assignment on the GPU, without host syncs.

The kernel is the Triton backend of ``torch_linear_assignment`` as used by
rf-detr, unchanged apart from the launch wrapper. It reproduces SciPy's
rectangular LSAP (shortest augmenting path with the same tie order). One Triton
program solves one batch item, so a batch of matching problems is one launch.

The upstream wrapper validates its input and checks the per-item infeasibility
flag with ``.item()``. Both force a device sync, which is what this module exists
to avoid, so the wrapper here returns the flag-free assignment instead: a row the
kernel could not assign is left at ``-1`` and callers must mask it.

Upstream launches with ``num_warps=1`` because one of its multi-warp test cases
reported false infeasibility. With 3600 OWLv2 patch columns a single warp is
slower than SciPy (6.0 vs 2.9 ms per batch of 8), while four warps take 0.6 ms
and matched SciPy's total cost on all 80 LV-MHP problems checked. A false
infeasibility would surface as a ``-1`` row, which callers count and report.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _linear_assignment_kernel(
    cost,
    u,
    v,
    col4row,
    row4col,
    infeasible,
    NUM_ROWS: tl.constexpr,
    NUM_COLUMNS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_FP64: tl.constexpr,
):
    """Solve one rectangular assignment problem in a single Triton program."""
    batch_index = tl.program_id(0)
    lane = tl.arange(0, BLOCK_N)
    column_mask = lane < NUM_COLUMNS
    row_mask = lane < NUM_ROWS

    cost = cost + batch_index * NUM_ROWS * NUM_COLUMNS
    u = u + batch_index * NUM_ROWS
    v = v + batch_index * NUM_COLUMNS
    col4row = col4row + batch_index * NUM_ROWS
    row4col = row4col + batch_index * NUM_COLUMNS

    for current_row in tl.range(0, NUM_ROWS, num_stages=1):
        scanned_rows = tl.zeros((BLOCK_N,), tl.int1)
        scanned_columns = tl.zeros((BLOCK_N,), tl.int1)
        active_columns = column_mask
        scan_positions = NUM_COLUMNS - lane - 1
        path = tl.full((BLOCK_N,), -1, tl.int32)
        if IS_FP64:
            shortest_path_costs = tl.full((BLOCK_N,), float("inf"), tl.float64)
            scanned_row_costs = tl.zeros((BLOCK_N,), tl.float64)
        else:
            shortest_path_costs = tl.full((BLOCK_N,), float("inf"), tl.float32)
            scanned_row_costs = tl.zeros((BLOCK_N,), tl.float32)

        sink = -1
        current_potential = tl.load(u + current_row)
        # A workspace-derived zero keeps this loop-carried value FP32 or FP64.
        min_value = current_potential - current_potential
        search_row = current_row
        num_remaining = NUM_COLUMNS

        # Wide matrices usually reach an unmatched sink early; do not execute
        # the remaining masked column-search iterations after that point.
        searching = True
        while searching:
            scanned_rows = scanned_rows | (row_mask & (lane == search_row) & searching)
            candidate_mask = active_columns & searching
            previous_cost = shortest_path_costs
            reduced_cost = (
                min_value
                - tl.load(u + search_row)
                + tl.load(
                    cost + search_row * NUM_COLUMNS + lane,
                    mask=candidate_mask,
                    other=float("inf"),
                )
                - tl.load(v + lane, mask=candidate_mask, other=0.0)
            )
            improved = candidate_mask & (reduced_cost < previous_cost)
            candidate_cost = tl.where(improved, reduced_cost, previous_cost)
            shortest_path_costs = candidate_cost
            path = tl.where(improved, search_row, path)

            # The legacy solver scans a swap-removed ``remaining`` array. Track
            # each column's current scan position locally to preserve its exact
            # last-unmatched/first-matched tie rule without shared scratch writes.
            lowest = tl.min(
                tl.where(candidate_mask, candidate_cost, float("inf")),
                axis=0,
            )
            has_candidate = lowest != float("inf")
            step_valid = searching & has_candidate
            tied = candidate_mask & (candidate_cost == lowest)
            matched_rows = tl.load(row4col + lane, mask=candidate_mask, other=0)
            last_unmatched_position = tl.max(
                tl.where(tied & (matched_rows == -1), scan_positions, -1),
                axis=0,
            )
            first_tied_position = tl.min(
                tl.where(tied, scan_positions, NUM_COLUMNS),
                axis=0,
            )
            selected_position = tl.where(
                last_unmatched_position >= 0,
                last_unmatched_position,
                first_tied_position,
            )
            selected_column = tl.max(
                tl.where(candidate_mask & (scan_positions == selected_position), lane, -1),
                axis=0,
            )
            selected_column = tl.where(step_valid, selected_column, 0)
            selected_row = tl.load(row4col + selected_column)

            # A matched row enters the search tree through ``selected_column``.
            # Save that column's shortest cost by row now, avoiding a dynamic
            # gather from the loop-carried column vector after the search.
            matched_step = step_valid & (selected_row != -1)
            scanned_row_costs = tl.where(
                row_mask & (lane == selected_row) & matched_step,
                lowest,
                scanned_row_costs,
            )
            sink = tl.where(step_valid & (selected_row == -1), selected_column, sink)
            search_row = tl.where(
                matched_step,
                selected_row,
                search_row,
            )
            scanned_columns = scanned_columns | (column_mask & (lane == selected_column) & step_valid)
            last_index = num_remaining - 1
            last_column = tl.max(
                tl.where(active_columns & (scan_positions == last_index), lane, -1),
                axis=0,
            )
            scan_positions = tl.where(
                active_columns & (lane == last_column) & step_valid,
                selected_position,
                scan_positions,
            )
            active_columns = active_columns & ~((lane == selected_column) & step_valid)
            num_remaining = tl.where(step_valid, num_remaining - 1, num_remaining)
            min_value = tl.where(step_valid, lowest, min_value)
            tl.store(infeasible + batch_index, 1, mask=searching & ~has_candidate)
            searching = matched_step

        solved = sink != -1
        tl.store(infeasible + batch_index, 1, mask=~solved)

        tl.store(u + current_row, current_potential + min_value, mask=solved)
        visited_rows = scanned_rows
        update_row = solved & row_mask & visited_rows & (lane != current_row)
        tl.store(
            u + lane,
            tl.load(u + lane, mask=row_mask, other=0.0) + min_value - scanned_row_costs,
            mask=update_row,
        )

        update_column = solved & column_mask & scanned_columns
        tl.store(
            v + lane,
            tl.load(v + lane, mask=column_mask, other=0.0) - min_value + shortest_path_costs,
            mask=update_column,
        )

        augmenting = solved
        augmenting_column = sink
        augmentation_steps = 0
        # Recover only the actual path while retaining the row-count safety cap.
        while augmenting & (augmentation_steps < NUM_ROWS):
            safe_column = tl.where(augmenting, augmenting_column, 0)
            augmenting_row = tl.max(
                tl.where(column_mask & (lane == safe_column), path, -1),
                axis=0,
            )
            safe_row = tl.where(augmenting, augmenting_row, 0)
            previous_column = tl.load(col4row + safe_row)
            tl.store(row4col + safe_column, augmenting_row, mask=augmenting)
            tl.store(col4row + safe_row, augmenting_column, mask=augmenting)
            augmenting = augmenting & (augmenting_row != current_row)
            augmenting_column = tl.where(augmenting, previous_column, augmenting_column)
            augmentation_steps += 1

        # The next row consumes the assignments and potentials written above.
        tl.debug_barrier()


def batch_linear_assignment(cost: torch.Tensor, num_warps: int = 4) -> torch.Tensor:
    """Minimise each ``[rows, columns]`` problem of a ``[B, rows, columns]`` CUDA cost.

    Requires ``rows <= columns``. Returns the column assigned to each row as a
    ``[B, rows]`` long tensor, ``-1`` where the kernel failed to assign the row.
    ``rows`` and ``columns`` are compile-time constants of the kernel, so keep
    them fixed (pad the rows) to avoid a recompile for every new shape.
    """
    if cost.ndim != 3 or not cost.is_cuda:
        raise ValueError("Need a 3-dimensional CUDA cost tensor of shape (B, rows, columns)")
    batch_size, rows, columns = cost.shape
    if rows > columns:
        raise ValueError(f"Need rows <= columns, got {rows} rows and {columns} columns")
    cost = cost.float().contiguous()
    u = torch.zeros((batch_size, rows), device=cost.device, dtype=cost.dtype)
    v = torch.zeros((batch_size, columns), device=cost.device, dtype=cost.dtype)
    col4row = torch.full((batch_size, rows), -1, device=cost.device, dtype=torch.int32)
    row4col = torch.full((batch_size, columns), -1, device=cost.device, dtype=torch.int32)
    infeasible = torch.zeros((batch_size,), device=cost.device, dtype=torch.int32)
    _linear_assignment_kernel[(batch_size,)](
        cost,
        u,
        v,
        col4row,
        row4col,
        infeasible,
        NUM_ROWS=rows,
        NUM_COLUMNS=columns,
        BLOCK_N=triton.next_power_of_2(columns),
        IS_FP64=False,
        num_warps=num_warps,
    )
    return col4row.long()
