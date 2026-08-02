"""Forward Explorative Modeling for record-aligned autoregressive training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class XMSelection:
    candidate_losses: torch.Tensor
    winner_candidate_indices: torch.Tensor
    winner_mode_indices: torch.Tensor


def select_xm_candidate_losses(
    candidate_losses: torch.Tensor,
    token_counts: torch.Tensor,
):
    """Select one candidate per sample and preserve baseline token weighting."""
    if candidate_losses.ndim != 2:
        raise ValueError("candidate_losses must have shape (K, B)")
    if token_counts.shape != candidate_losses.shape[1:]:
        raise ValueError("token_counts must have shape (B,)")
    best_losses, winner_indices = candidate_losses.min(dim=0)
    token_counts = token_counts.to(dtype=best_losses.dtype)
    loss = (best_losses * token_counts).sum() / token_counts.sum().clamp_min(1)
    return loss, winner_indices


def xm_best_of_k_forward(
    model,
    idx: torch.Tensor,
    cond_vec,
    targets: torch.Tensor,
    start_indices_batch,
    best_of_k: int,
    *,
    candidate_mode_indices: Optional[torch.Tensor] = None,
    return_selection: bool = False,
):
    """Run memory-saving Forward XM with shared non-explored randomness."""
    if best_of_k < 1:
        raise ValueError("xm_best_of_k must be >= 1")
    if best_of_k == 1:
        result = model(idx, cond_vec, targets, start_indices_batch)
        if return_selection:
            return (*result, None)
        return result

    batch_size = idx.size(0)
    if len(start_indices_batch) != batch_size or any(
        len(starts) != 1 or int(starts[0]) != 0
        for starts in start_indices_batch
    ):
        raise ValueError(
            "Forward XM requires one record starting at index 0 per batch row"
        )
    if candidate_mode_indices is None:
        candidate_mode_indices = torch.randint(
            best_of_k,
            (best_of_k, batch_size),
            device=idx.device,
        )
    else:
        candidate_mode_indices = candidate_mode_indices.to(
            device=idx.device,
            dtype=torch.long,
        )
        if candidate_mode_indices.shape != (best_of_k, batch_size):
            raise ValueError(
                "candidate_mode_indices must have shape (xm_best_of_k, batch_size)"
            )
        if torch.any(candidate_mode_indices < 0) or torch.any(
            candidate_mode_indices >= best_of_k
        ):
            raise ValueError("candidate mode index is outside the XM embedding range")

    shared_rng_state = _capture_rng_state(idx.device)
    candidate_losses = []
    for mode_indices in candidate_mode_indices:
        _restore_rng_state(shared_rng_state, idx.device)
        with torch.no_grad():
            _, per_sample_loss = model(
                idx,
                cond_vec,
                targets,
                start_indices_batch,
                xm_mode_indices=mode_indices,
                return_per_sample_loss=True,
            )
        candidate_losses.append(per_sample_loss)

    candidate_losses = torch.stack(candidate_losses, dim=0)
    token_counts = targets.ne(-1).sum(dim=1)
    _, winner_candidate_indices = select_xm_candidate_losses(
        candidate_losses,
        token_counts,
    )
    winner_mode_indices = candidate_mode_indices.gather(
        0,
        winner_candidate_indices.unsqueeze(0),
    ).squeeze(0)

    _restore_rng_state(shared_rng_state, idx.device)
    torch.clear_autocast_cache()
    logits, winner_losses = model(
        idx,
        cond_vec,
        targets,
        start_indices_batch,
        xm_mode_indices=winner_mode_indices,
        return_per_sample_loss=True,
    )
    loss, _ = select_xm_candidate_losses(
        winner_losses.unsqueeze(0),
        token_counts,
    )
    if return_selection:
        selection = XMSelection(
            candidate_losses=candidate_losses,
            winner_candidate_indices=winner_candidate_indices,
            winner_mode_indices=winner_mode_indices,
        )
        return logits, loss, selection
    return logits, loss


def _capture_rng_state(device: torch.device):
    cpu_state = torch.random.get_rng_state()
    cuda_state = None
    if device.type == "cuda":
        cuda_state = torch.cuda.get_rng_state(device)
    return cpu_state, cuda_state


def _restore_rng_state(state, device: torch.device):
    cpu_state, cuda_state = state
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)
