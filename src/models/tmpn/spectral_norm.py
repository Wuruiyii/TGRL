import torch
import torch.nn as nn


def _spectral_norm_param(W: torch.Tensor, bound: float = 1.0,
                         n_power_iterations: int = 1) -> torch.Tensor:
    """Clamp spectral norm of W to <= bound via power iteration."""
    with torch.no_grad():
        u = torch.randn(W.size(0), device=W.device)
        u = u / u.norm()
        for _ in range(n_power_iterations):
            v = W.t() @ u
            v = v / v.norm().clamp(min=1e-12)
            u = W @ v
            u = u / u.norm().clamp(min=1e-12)
        sigma = u @ W @ v
    if sigma > bound:
        W = W * (bound / sigma)
    return W


def apply_spectral_norm_to_relation_weights(module: nn.Module,
                                             bound: float = 1.0):
    """Apply spectral normalisation to W_r parameters in TypedMeanAggregator."""
    if hasattr(module, "W_r"):
        with torch.no_grad():
            for i, w in enumerate(module.W_r):
                module.W_r[i] = nn.Parameter(
                    _spectral_norm_param(w.data, bound)
                )


class SpectralNormHook:
    """Training hook that re-normalises after each optimiser step."""

    def __init__(self, model, bound: float = 1.0):
        self.model = model
        self.bound = bound

    def __call__(self):
        for m in self.model.modules():
            apply_spectral_norm_to_relation_weights(m, self.bound)
