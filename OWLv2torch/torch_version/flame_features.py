"""Torch-only approximate RBF SVMs for FLAME, fitted on the model's device.

Both feature maps use the same standardized inputs, RBF gamma and balanced
class penalties as SVC. The linear classifier minimizes L2 regularization plus
a Huber-smoothed hinge loss (width 0.01), with an unregularized intercept.
Nyström weights are folded back into an ordinary RBF expansion for inference.
"""

from __future__ import annotations

import math

import torch


def rbf_kernel(x: torch.Tensor, centers: torch.Tensor, gamma: float) -> torch.Tensor:
    distance = x.square().sum(1, keepdim=True) + centers.square().sum(1) - 2 * x @ centers.T
    return torch.exp(-gamma * distance.clamp_min(0))


def balanced_landmarks(y: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    """Keep rare positives represented instead of sampling mostly background."""
    y = y.cpu()
    count = min(count, len(y))
    if count == len(y):
        return torch.arange(len(y))
    rng = torch.Generator().manual_seed(seed)
    positive, negative = (y == 1).nonzero()[:, 0], (y == 0).nonzero()[:, 0]
    n_pos = min(len(positive), count // 2)
    n_neg = min(len(negative), count - n_pos)
    n_pos = min(len(positive), count - n_neg)
    return torch.cat((positive[torch.randperm(len(positive), generator=rng)[:n_pos]],
                      negative[torch.randperm(len(negative), generator=rng)[:n_neg]]))


def linear_svm(
    features: torch.Tensor, y: torch.Tensor, c: float, max_iter: int, *, smoothing: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Fit balanced, summed-loss C-SVC using damped Newton steps.

    Dividing the objective by C*N improves numerical scaling without changing
    its optimum. Smoothing bounds the change to each hinge loss by width/2.
    """
    # Double precision keeps the small, regularized Hessian well conditioned
    # numerically when a rare positive carries a very large balanced penalty.
    with torch.inference_mode(False), torch.no_grad():
        features = features.detach().double()
        y = y.to(device=features.device, dtype=torch.long).clone()
        signs = 2 * y.double() - 1
        counts = torch.bincount(y, minlength=2)
        penalties = len(y) / (2 * counts[y].double())
        regularization = 1 / (c * len(y))
        theta = features.new_zeros(features.shape[1] + 1)
        scores = features.new_zeros(len(y))

        def objective(parameters, predictions):
            residual = (1 - signs * predictions).clamp_min(0)
            hinge = torch.where(residual < smoothing, residual.square() / (2 * smoothing),
                                residual - smoothing / 2)
            return regularization * parameters[:-1].square().sum() / 2 + (penalties * hinge).mean()

        loss = objective(theta, scores)
        evaluations, iterations = 1, 0
        reason = "iteration_limit"
        for _ in range(max_iter):
            residual = 1 - signs * scores
            derivative = -penalties * signs * (residual / smoothing).clamp(0, 1) / len(y)
            gradient = torch.cat((features.T @ derivative + regularization * theta[:-1],
                                  derivative.sum()[None]))
            gradient_max = float(gradient.abs().max())
            if gradient_max < 1e-7:
                reason = "gradient_tolerance"
                break
            active = (residual > 0) & (residual < smoothing)
            hessian = torch.eye(len(theta), device=features.device, dtype=torch.float64) * regularization
            if bool(active.any()):
                active_features = torch.cat((features[active], features.new_ones((int(active.sum()), 1))), 1)
                hessian[-1, -1] = 0
                hessian += active_features.T @ (active_features * (penalties[active] / (len(y) * smoothing))[:, None])
            # When no point lies in the smoothing band, the temporary intercept
            # damping above makes the Newton system invertible. The objective
            # itself never penalizes the intercept.
            factor = torch.linalg.cholesky(hessian)
            direction = torch.cholesky_solve(-gradient[:, None], factor)[:, 0]
            slope = gradient @ direction
            score_step = features @ direction[:-1] + direction[-1]
            step = 1.0
            for _ in range(50):
                candidate, candidate_scores = theta + step * direction, scores + step * score_step
                candidate_loss = objective(candidate, candidate_scores)
                evaluations += 1
                if bool(candidate_loss <= loss + 1e-4 * step * slope):
                    break
                step *= 0.5
            else:
                reason = "line_search_stalled"
                break
            theta, scores, loss = candidate, candidate_scores, candidate_loss
            iterations += 1

        residual = 1 - signs * scores
        derivative = -penalties * signs * (residual / smoothing).clamp(0, 1) / len(y)
        gradient = torch.cat((features.T @ derivative + regularization * theta[:-1], derivative.sum()[None]))
        gradient_max = float(gradient.abs().max())
        if gradient_max < 1e-7:
            reason = "gradient_tolerance"
        diagnostics = {"solver": "newton", "iterations": iterations, "evaluations": evaluations,
                       "objective": float(loss) * c * len(y), "gradient_max": gradient_max,
                       "smoothing": smoothing, "converged": gradient_max < 1e-7, "stop_reason": reason}
        return theta[:-1].float(), theta[-1].float(), diagnostics


def fit_approximate(
    x: torch.Tensor, y: torch.Tensor, *, backend: str, gamma: float, c: float,
    n_components: int, device: torch.device, seed: int, max_iter: int,
) -> dict:
    """Return a CPU-resident FLAME classifier, with no sklearn dependency."""
    with torch.inference_mode(False), torch.no_grad():
        x = x.to(device=device, dtype=torch.float32).clone()
        if backend == "nystrom":
            indices = balanced_landmarks(y, n_components, seed).to(device)
            centers = x[indices]
            gram = rbf_kernel(centers, centers, gamma)
            eigenvalues, eigenvectors = torch.linalg.eigh(gram.double())
            basis = (eigenvectors * eigenvalues.clamp_min(1e-7).rsqrt()).float()
            features = torch.cat([rbf_kernel(chunk, centers, gamma) @ basis for chunk in x.split(4096)])
            state = {"vectors": centers.cpu()}
        elif backend == "rff":
            rng = torch.Generator().manual_seed(seed)
            frequencies = (torch.randn(x.shape[1], n_components, generator=rng)
                           * math.sqrt(2 * gamma)).to(device)
            phases = (torch.rand(n_components, generator=rng) * (2 * math.pi)).to(device)
            features = torch.cos(x @ frequencies + phases) * math.sqrt(2 / n_components)
            state = {"frequencies": frequencies.cpu(), "phases": phases.cpu()}
        else:
            raise ValueError(f"Unknown approximate SVM backend: {backend}")

    weight, bias, diagnostics = linear_svm(features, y, c, max_iter)
    with torch.no_grad():
        if backend == "nystrom":
            state["coef"] = (basis @ weight).cpu()
        else:
            state["coef"] = (weight * math.sqrt(2 / n_components)).cpu()
    return {**state, "backend": backend, "gamma": gamma, "bias": float(bias),
            "fit": diagnostics, "n_components": features.shape[1]}
