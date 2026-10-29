"""Full-kernel, balanced binary C-SVC on the Torch device.

Hierarchical SMO solves the ordinary SVM dual, including its equality constraint
and unregularized intercept. Working sets limit memory, not the kernel's rank:
the gradient and final KKT check always cover every training example. CUDA uses
a fused Triton inner solver; the Torch inner loop also works on CPU.
"""

from __future__ import annotations

import math
import warnings
from contextlib import contextmanager

import torch


@contextmanager
def _kernel_precision(device):
    # A caller's autocast/TF32 settings must not change the SVM kernel.
    precision = torch.get_float32_matmul_precision()
    try:
        if device.type == "cuda":
            torch.set_float32_matmul_precision("highest")
        with torch.autocast(device.type, enabled=False):
            yield
    finally:
        if device.type == "cuda":
            torch.set_float32_matmul_precision(precision)


class _KernelRows:
    """Reuse exact RBF rows across working sets within a fixed memory budget."""

    def __init__(self, x, gamma, size):
        self.x, self.gamma = x, gamma
        self.norms = x.square().sum(1)
        self.rows = x.new_zeros((size, len(x)))
        self.slots: dict[int, int] = {}
        self.evaluations = 0

    def compute(self, indices):
        distance = self.norms[indices, None] + self.norms[None, :]
        distance.addmm_(self.x[indices], self.x.T, beta=1, alpha=-2)
        distance.clamp_min_(0).mul_(-self.gamma).exp_()
        # RBF diagonals are exactly one, even if the dot product rounds.
        distance[torch.arange(len(indices), device=self.x.device), indices] = 1
        self.evaluations += len(indices) * len(self.x)
        return distance

    def get(self, indices):
        requested = indices.cpu().tolist()
        wanted = set(requested)
        self.slots = {i: slot for i, slot in self.slots.items() if i in wanted}
        available = iter(sorted(set(range(len(self.rows))) - set(self.slots.values())))
        missing = [i for i in requested if i not in self.slots]
        for i in missing:
            self.slots[i] = next(available)
        # Bound construction scratch independently of the number of cached rows.
        for start in range(0, len(missing), 32):
            batch = missing[start:start + 32]
            selected = torch.tensor(batch, device=self.x.device)
            slots = torch.tensor([self.slots[i] for i in batch], device=self.x.device)
            self.rows.index_copy_(0, slots, self.compute(selected))
        return torch.tensor([self.slots[i] for i in requested], device=self.x.device)

    def refresh(self, signs, alpha):
        """Reconstruct y - K(alpha*y), with double-precision accumulation."""
        h = signs.clone()
        support = (alpha > 0).nonzero()[:, 0]
        for indices in support.split(32):
            h.addmv_(self.compute(indices).double().T, alpha[indices] * signs[indices], alpha=-1)
        return h


def _bounds(alpha, signs, penalties, h):
    up = ((signs > 0) & (alpha < penalties)) | ((signs < 0) & (alpha > 0))
    low = ((signs > 0) & (alpha > 0)) | ((signs < 0) & (alpha < penalties))
    return h.masked_fill(~up, -torch.inf), h.masked_fill(~low, torch.inf)


def _working_set(up, low, size, previous):
    count = max(1, size // (4 if previous is not None else 2))
    high, high_indices = up.topk(count)
    bottom, low_indices = low.topk(count, largest=False)
    selected = torch.cat((high_indices[torch.isfinite(high)], low_indices[torch.isfinite(bottom)])).unique()
    if previous is not None:
        retained = previous[~torch.isin(previous, selected)][:size - len(selected)]
        selected = torch.cat((selected, retained))
    return selected


def _solve_torch(gram, alpha, signs, penalties, h, tol, max_steps):
    """Reference SMO loop; CUDA uses the same updates in a single GPU program."""
    alpha, h = alpha.clone(), h.clone()
    original = alpha.clone()
    steps = 0
    for _ in range(max_steps):
        up, low = _bounds(alpha, signs, penalties, h)
        i = int(up.argmax())
        if float(up[i] - low.min()) <= tol:
            break
        difference = h[i] - h
        curvature = (2 - 2 * gram[i].double()).clamp_min(1e-12)
        gain = (difference.square() / curvature).masked_fill(~(torch.isfinite(low) & (difference > 0)), -1)
        j = int(gain.argmax())
        room_i = torch.where(signs[i] > 0, penalties[i] - alpha[i], alpha[i])
        room_j = torch.where(signs[j] > 0, alpha[j], penalties[j] - alpha[j])
        step = torch.minimum(torch.minimum(room_i, room_j), difference[j] / curvature[j])
        alpha[i] = (alpha[i] + signs[i] * step).clamp(0, penalties[i])
        alpha[j] = (alpha[j] - signs[j] * step).clamp(0, penalties[j])
        h -= step * (gram[i].double() - gram[j].double())
        steps += 1
    return alpha, (alpha - original) * signs, alpha.new_tensor(steps)


def fit_rbf(
    x: torch.Tensor, y: torch.Tensor, *, gamma: float, c: float = 1.0,
    device: torch.device | str = "cpu", max_iter: int = 300, tol: float = 1e-3,
    cache_size_mb: float = 256, working_set_size: int = 512,
) -> dict:
    """Fit an exact RBF C-SVC and return the CPU-resident FLAME head.

    ``max_iter`` counts outer working-set solves, each allowing up to
    ``max(1000, 20 * working_set_size)`` pair updates. ``tol`` bounds the global
    KKT gap. Cache size is in MiB and bounds stored kernel rows; input features,
    optimizer state, local Gram matrix and construction scratch are additional.
    Reaching the budget without convergence emits a warning and is recorded in
    ``fit``; it never silently switches to an approximate objective.
    """
    device = torch.device(device)
    if x.ndim != 2 or len(x) < 2 or x.shape[1] == 0 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Expected x [N, D] and y [N], with N >= 2 and D >= 1")
    if not all(math.isfinite(v) and v > 0 for v in (gamma, c, tol, cache_size_mb)):
        raise ValueError("gamma, c, tol and cache_size_mb must be finite and positive")
    if max_iter < 1 or working_set_size < 2:
        raise ValueError("max_iter must be positive and working_set_size must be at least 2")
    if not bool(torch.isfinite(x).all()) or not bool(((y == 0) | (y == 1)).all()):
        raise ValueError("Features must be finite and targets must be 0 or 1")
    counts = torch.bincount(y.long().cpu(), minlength=2)
    if not bool((counts > 0).all()):
        raise ValueError("RBF C-SVC requires both target classes")
    size = min(working_set_size, len(x), int(cache_size_mb * 2**20 / (4 * len(x))))
    if size < 2:
        raise ValueError("cache_size_mb must hold at least two float32 kernel rows")

    solve = _solve_torch
    if device.type == "cuda":
        try:
            from .flame_svm_triton import solve_working_set
        except ImportError:
            warnings.warn("Triton is unavailable; using the slower Torch SMO inner loop on CUDA", RuntimeWarning)
        else:
            solve = solve_working_set

    with torch.inference_mode(False), torch.no_grad(), _kernel_precision(device):
        x = x.detach().to(device=device, dtype=torch.float32).clone()
        labels = y.to(device=device, dtype=torch.long).clone()
        signs = labels.double() * 2 - 1
        penalties = c * (len(y) / (2 * counts.to(device)[labels].double()))
        if not bool(torch.isfinite(penalties).all()):
            raise ValueError("Class-balanced penalties overflow; reduce c")
        alpha = torch.zeros_like(signs)
        h = signs.clone()  # -y * dual gradient = y - K(alpha*y)
        cache = _KernelRows(x, gamma, size)
        previous, iterations = None, 0
        pair_updates = alpha.new_zeros(())
        refreshed = False
        for _ in range(max_iter):
            up, low = _bounds(alpha, signs, penalties, h)
            violation = float(up.max() - low.min())
            if violation <= tol:
                # Incremental float32 GEMVs can drift. Verify against the full
                # expansion before accepting convergence, and continue if needed.
                h = cache.refresh(signs, alpha)
                up, low = _bounds(alpha, signs, penalties, h)
                violation = float(up.max() - low.min())
                if violation <= tol:
                    refreshed = True
                    break
            selected = _working_set(up, low, size, previous)
            slots = cache.get(selected)
            gram = cache.rows[slots[:, None], selected[None, :]].contiguous()
            new_alpha, delta, steps = solve(
                gram, alpha[selected], signs[selected], penalties[selected], h[selected],
                max(tol * 0.5, violation * 0.1), max(1000, 20 * size),
            )
            alpha[selected] = new_alpha
            changes = cache.rows.new_zeros(size)
            changes[slots] = delta.float()
            h -= cache.rows.T @ changes
            pair_updates += steps
            previous = selected
            iterations += 1

        if not refreshed:
            h = cache.refresh(signs, alpha)
        up, low = _bounds(alpha, signs, penalties, h)
        violation = float(up.max() - low.min())
        violation = max(0.0, violation) if math.isfinite(violation) else math.inf
        free = (alpha > 0) & (alpha < penalties)
        bias = float(h[free].mean() if bool(free.any()) else (up.max() + low.min()) / 2)
        support = alpha > 0
        beta = alpha * signs
        norm_squared = beta @ (signs - h)
        primal = norm_squared / 2 + (penalties * (1 - signs * (signs - h + bias)).clamp_min(0)).sum()
        dual = alpha.sum() - norm_squared / 2
        converged = math.isfinite(violation) and violation <= tol
        diagnostics = {
            "solver": "hierarchical_smo", "inner_solver": "triton" if solve is not _solve_torch else "torch",
            "iterations": iterations, "pair_updates": int(pair_updates), "kkt_violation": violation,
            "tolerance": tol, "objective": float(primal), "dual_objective": float(dual),
            "duality_gap": float(primal - dual), "equality_residual": float(beta.sum().abs()),
            "converged": converged, "stop_reason": "kkt_tolerance" if converged else "iteration_limit",
            "support_vectors": int(support.sum()), "working_set_size": size,
            "kernel_cache_mb": cache.rows.numel() * cache.rows.element_size() / 2**20,
            "kernel_evaluations": cache.evaluations,
        }
        if not converged:
            warnings.warn(f"RBF SVM did not converge after {iterations} working-set iterations "
                          f"(KKT gap {violation:.6g} > {tol:g}); increase svm_max_iter", RuntimeWarning)
        return {"backend": "torch", "gamma": gamma, "vectors": x[support].cpu(),
                "coef": beta[support].float().cpu(), "bias": bias, "fit": diagnostics}
