# server/aggregation.py

import torch
from typing import List


def multi_krum(
    updates: List[torch.Tensor],
    f: int,
) -> List[int]:
    """
    Simplified implementation of Multi-Krum.
    Returns the indices of the selected updates.
    """
    n = len(updates)
    assert n > 2 * f + 2, "Multi-Krum condition not met"
    updates_stack = torch.stack(updates)  # [n, d]
    dists = torch.cdist(updates_stack, updates_stack, p=2) ** 2  # [n, n]

    scores = []
    m = n - f - 2
    for i in range(n):
        d_i = dists[i].clone()
        d_i_sorted, _ = torch.sort(d_i)
        scores.append(d_i_sorted[1 : m + 1].sum().item())  # ignoring distance from oneself

    scores = torch.tensor(scores)
    # For this PoC we choose n - f, but we can adjust
    k = n - f
    _, selected_indices = torch.topk(-scores, k)  # Lower score => more reliable
    return selected_indices.tolist()


def clipped_mean(
    updates: List[torch.Tensor],
    clip_norm: float = 10.0,
) -> torch.Tensor:
    clipped = []
    for u in updates:
        norm = torch.norm(u, p=2)
        if norm > clip_norm:
            u = u * (clip_norm / norm)
        clipped.append(u)
    return torch.stack(clipped).mean(dim=0)
