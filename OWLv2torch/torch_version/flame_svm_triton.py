"""Fused local SMO updates for FLAME's full-kernel C-SVC (imported only on CUDA)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _solve(Gram, Alpha, Signs, Penalties, H, Result, Delta, Steps,
           N, TOL, MAX_STEPS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.arange(0, BLOCK)
    valid = index < N
    a = tl.load(Alpha + index, valid, 0)
    original = a
    y = tl.load(Signs + index, valid, 0)
    c = tl.load(Penalties + index, valid, 0)
    h = tl.load(H + index, valid, 0)
    count = 0
    up = valid & (((y > 0) & (a < c)) | ((y < 0) & (a > 0)))
    low = valid & (((y > 0) & (a > 0)) | ((y < 0) & (a < c)))
    high = tl.max(tl.where(up, h, -float("inf")), 0)
    bottom = tl.min(tl.where(low, h, float("inf")), 0)
    while (count < MAX_STEPS) & (high - bottom > TOL):
        i = tl.min(tl.where(up & (h == high), index, BLOCK), 0)
        kernel_i = tl.load(Gram + i * N + index, valid, 0).to(tl.float64)
        curvature = tl.maximum(2.0 - 2.0 * kernel_i, 1e-12)
        difference = high - h
        gain = tl.where(low & (difference > 0), difference * difference / curvature, -1.0)
        best = tl.max(gain, 0)
        j = tl.min(tl.where(valid & (gain == best), index, BLOCK), 0)
        room_i = tl.sum(tl.where(index == i, tl.where(y > 0, c - a, a), 0.0), 0)
        room_j = tl.sum(tl.where(index == j, tl.where(y > 0, a, c - a), 0.0), 0)
        optimum = tl.sum(tl.where(index == j, difference / curvature, 0.0), 0)
        step = tl.minimum(tl.minimum(room_i, room_j), optimum)
        change = tl.where(index == i, y * step, tl.where(index == j, -y * step, 0.0))
        a = tl.minimum(c, tl.maximum(0.0, a + change))
        kernel_j = tl.load(Gram + j * N + index, valid, 0).to(tl.float64)
        h -= step * (kernel_i - kernel_j)
        count += 1
        up = valid & (((y > 0) & (a < c)) | ((y < 0) & (a > 0)))
        low = valid & (((y > 0) & (a > 0)) | ((y < 0) & (a < c)))
        high = tl.max(tl.where(up, h, -float("inf")), 0)
        bottom = tl.min(tl.where(low, h, float("inf")), 0)
    tl.store(Result + index, a, valid)
    tl.store(Delta + index, (a - original) * y, valid)
    tl.store(Steps, count)


def solve_working_set(gram, alpha, signs, penalties, h, tol, max_steps):
    result, delta = torch.empty_like(alpha), torch.empty_like(alpha)
    steps = alpha.new_empty(())
    # Runtime size and tolerance avoid recompiling for every working set and
    # changing outer-loop KKT gap.
    with torch.cuda.device(gram.device):
        _solve[(1,)](gram, alpha, signs, penalties, h, result, delta, steps,
                     len(alpha), tol, max_steps, triton.next_power_of_2(len(alpha)),
                     num_warps=4, enable_fp_fusion=False)
    return result, delta, steps
