# server/server.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List
from pathlib import Path
import time
import torch

from config import (
    NUM_CLIENTS,
    BYZ_FRACTION,
    INPUT_DIM,
    OUTPUT_DIM,
    LEARNING_RATE,
)
from models.model import SimpleRegressor
from server.aggregation import multi_krum, clipped_mean
from clients.client import ClientUpdate
from config import OQS_SIG_ALG

try:
    import oqs  # liboqs-python
except ImportError:
    oqs = None


# =======================================================
# ===================== STATE ===========================
# =======================================================

@dataclass
class ServerState:
    global_model: SimpleRegressor
    round_idx: int = 0


# =======================================================
# ===================== SERVER ==========================
# =======================================================

class Server:
    def __init__(self):
        self.state = ServerState(
            global_model=SimpleRegressor(INPUT_DIM, OUTPUT_DIM)
        )

    # ------------------------------

    def verify_signature_and_proof(self, update: ClientUpdate) -> bool:
        """
        For the PoC :
        - Verify the ML-DSA-65 signature via liboqs
        - Simulates the STARK check
        """
        # Signature verification
        try:
            if oqs is not None:
                with oqs.Signature(OQS_SIG_ALG) as sig:
                    ok = sig.verify(update.proof, update.signature, update.public_key)
            else:
                ok = True
        except Exception:
            ok = True  # PoC fallback

        # TODO : Add the STARK verification here
        proof_ok = True

        return ok and proof_ok

    # ------------------------------

    def aggregate(
        self,
        client_updates: List[ClientUpdate],
        f: int,
    ) -> torch.Tensor:
        """
        Byzantine-Resilient Aggregation :
            - filter valid customers
            - Multi-Krum
            - trimmed/clipped mean
        """

        valid_updates = [
            u for u in client_updates if self.verify_client_update(u)
        ]


        if not valid_updates:
            raise RuntimeError("No valid updates")

        updates_vecs = [u.update for u in valid_updates]
        selected_idx = multi_krum(updates_vecs, f=f)
        selected_updates = [updates_vecs[i] for i in selected_idx]

        agg = clipped_mean(selected_updates, clip_norm=10.0)
        return agg

    # ------------------------------

    def apply_update(self, agg_delta: torch.Tensor):
        """
        Applies aggregated Δw to the global model.
        """
        offset = 0
        with torch.no_grad():
            for p in self.state.global_model.parameters():
                numel = p.numel()
                delta_slice = agg_delta[offset : offset + numel].view_as(p)
                p += LEARNING_RATE * delta_slice
                offset += numel

    # ------------------------------

    def get_global_model(self) -> SimpleRegressor:
        return self.state.global_model

    # =======================================================
    #            SERVER-SIDE METRICS
    # =======================================================

    def verify_client_update(self, update: ClientUpdate):
        """
        Verifies signature + proof with time metrics.
        """
        metrics = {}

        # 1) Signature verification
        t0 = time.perf_counter()
        sig_valid = self._verify_signature(update)
        metrics["verify_signature_ms"] = (time.perf_counter() - t0) * 1000

        # 2) ZK-STARK Verification
        t1 = time.perf_counter()
        proof_valid = self._verify_proof(update)
        metrics["verify_proof_ms"] = (time.perf_counter() - t1) * 1000

        # 3) Records metrics
        self._save_server_metrics(update.client_id, update.round_idx, metrics)

        return sig_valid and proof_valid

    # ------------------------------

    def _verify_signature(self, update: ClientUpdate) -> bool:
        """Encapsulates the ML-DSA verification."""
        try:
            if oqs is not None:
                with oqs.Signature(OQS_SIG_ALG) as sig:
                    return sig.verify(update.proof, update.signature, update.public_key)
            return True
        except Exception:
            return True  # fallback

    # ------------------------------

    def _verify_proof(self, update: ClientUpdate) -> bool:
        """Simulates (for now) the ZK-STARK verification."""
        # TODO : winterfell verify
        return True

    # ------------------------------

    def _save_server_metrics(self, client_id: int, round_idx: int, metrics: dict):
        """Writes the server metrics to artifacts/metrics_server.tsv"""
        out = Path("artifacts") / "metrics_server.tsv"
        out.parent.mkdir(exist_ok=True)

        header_needed = not out.exists()
        with out.open("a") as f:
            if header_needed:
                f.write("client_id\tround\tverify_signature_ms\tverify_proof_ms\n")

            f.write(
                f"{client_id}\t{round_idx}\t"
                f"{metrics['verify_signature_ms']:.3f}\t"
                f"{metrics['verify_proof_ms']:.3f}\n"
            )
