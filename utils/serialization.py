# utils/serialization.py

import json
from pathlib import Path
from typing import Dict, Any
import torch
import numpy as np


def tensor_to_list(t: torch.Tensor) -> list:
    return t.detach().cpu().numpy().tolist()


def list_to_tensor(lst: list, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(lst, dtype=dtype)


def quantize_tensor(t: torch.Tensor, scale: float = 1e4) -> np.ndarray:
    """
    Toy encoding: Q(x) = round(x * scale), 64-bit signed integer.
    aligned with the encoding scheme in F_p..
    """
    arr = t.detach().cpu().numpy()
    q = np.round(arr * scale).astype(np.int64)
    return q


def save_update_for_winterfell(
    client_id: int,
    round_idx: int,
    w_vec: torch.Tensor,
    g_vec: torch.Tensor,
    out_dir: Path,
    eta: float,
) -> Path:
    """
    Serialize (w, g, eta) for Winterfell in a JSON.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    q_w = quantize_tensor(w_vec)
    q_g = quantize_tensor(g_vec)
    payload: Dict[str, Any] = {
        "client_id": client_id,
        "round": round_idx,
        "eta": eta,
        "w": q_w.tolist(),
        "g": q_g.tolist(),
    }
    path = out_dir / f"client_{client_id}_round_{round_idx}.json"
    with path.open("w") as f:
        json.dump(payload, f)
    return path



def save_proof(client_id: int, round_idx: int, proof_bytes: bytes, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"client_{client_id}_round_{round_idx}.proof"
    with path.open("wb") as f:
        f.write(proof_bytes)
    return path

# utils/serialization.py


def flatten_model_params(model: torch.nn.Module) -> torch.Tensor:
    params = [p.detach().cpu().flatten() for p in model.parameters()]
    return torch.cat(params)
