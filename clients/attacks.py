# clients/attacks.py

import torch
from typing import Literal

AttackMode = Literal["honest", "scale", "random", "sign_flip"]


def apply_attack(
    update: torch.Tensor,
    mode: AttackMode = "honest",
    scale_factor: float = 10.0,
) -> torch.Tensor:
    """
    Simple Byzantine attack models.
    """
    if mode == "honest":
        return update
    if mode == "scale":
        return scale_factor * update
    if mode == "random":
        return torch.randn_like(update) * update.std().clamp_min(1e-6)
    if mode == "sign_flip":
        return -update
    raise ValueError(f"Unknown attack mode: {mode}")
