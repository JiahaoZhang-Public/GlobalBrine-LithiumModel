from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class IntegratedGradientsResult:
    attributions: torch.Tensor
    delta: torch.Tensor


def integrated_gradients(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    inputs: torch.Tensor,
    baseline: torch.Tensor,
    target_index: int,
    steps: int = 50,
) -> IntegratedGradientsResult:
    """Integrated Gradients for a single sample and a single target dimension.

    Args:
        forward_fn: maps a batch of inputs [N, D] -> outputs [N, K]
        inputs: [D]
        baseline: [D]
        target_index: which output dimension to explain
        steps: number of steps for the path integral (>= 1)
    """
    if inputs.ndim != 1 or baseline.ndim != 1:
        raise ValueError("inputs and baseline must be 1D tensors [D].")
    if inputs.shape != baseline.shape:
        raise ValueError("inputs and baseline must have the same shape.")
    if steps < 1:
        raise ValueError("steps must be >= 1.")

    device = inputs.device
    dtype = inputs.dtype
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=dtype)
    scaled = baseline.unsqueeze(0) + alphas.unsqueeze(1) * (inputs - baseline).unsqueeze(0)
    scaled.requires_grad_(True)

    outputs = forward_fn(scaled)
    if outputs.ndim != 2:
        raise ValueError("forward_fn must return a 2D tensor [N, K].")
    if not (0 <= target_index < outputs.shape[1]):
        raise ValueError("target_index out of range.")

    target = outputs[:, target_index].sum()
    grads = torch.autograd.grad(target, scaled, create_graph=False, retain_graph=False)[0]

    avg_grads = (grads[:-1] + grads[1:]) * 0.5
    integral = avg_grads.mean(dim=0)
    attributions = (inputs - baseline) * integral

    delta = outputs[-1, target_index] - outputs[0, target_index] - attributions.sum()
    return IntegratedGradientsResult(attributions=attributions.detach(), delta=delta.detach())

