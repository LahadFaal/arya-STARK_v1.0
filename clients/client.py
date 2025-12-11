# clients/client.py

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader

from config import (
    LEARNING_RATE,
    LOCAL_EPOCHS,
    DEVICE,
    WINTERFELL_PROVER_BIN,
    TRACE_DIR,
    PROOF_DIR,
    OQS_SIG_ALG,
)
from models.model import SimpleRegressor
from utils.data import make_dataloader
from utils.serialization import save_update_for_winterfell, flatten_model_params
from clients.attacks import apply_attack, AttackMode

try:
    import oqs  # liboqs-python
except ImportError:
    oqs = None


# =========================
# ======= STRUCTURES ======
# =========================

@dataclass
class PQKeys:
    """Stores the OQS Signature object and its associated public key."""
    signature: "oqs.Signature"
    public_key: bytes


@dataclass
class ClientUpdate:
    client_id: int
    round_idx: int
    update: torch.Tensor
    proof: bytes
    signature: bytes
    public_key: bytes


# =========================
# ========= CLIENT ========
# =========================

class Client:
    def __init__(
        self,
        client_id: int,
        dataset,
        is_byzantine: bool = False,
        attack_mode: AttackMode = "honest",
    ):
        self.id = client_id
        self.dataset = dataset
        self.is_byzantine = is_byzantine
        self.attack_mode = attack_mode
        self.model = None
        self._pq_keys: Optional[PQKeys] = None

        # ---------- OQS KEY GENERATION ----------
        if oqs is not None:
            sig = oqs.Signature(OQS_SIG_ALG)
            pub = sig.generate_keypair()
            self._pq_keys = PQKeys(signature=sig, public_key=pub)

    # ---------------------

    def _init_model_from_global(self, global_model: SimpleRegressor):
        self.model = SimpleRegressor(global_model.linear.in_features).to(DEVICE)
        self.model.load_state_dict(global_model.state_dict())

    # ---------------------

    def local_train(
        self,
        global_model: SimpleRegressor,
        round_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        
        Local training and return from (delta_vec, w_global_flat).
        """
        w_global_flat = flatten_model_params(global_model)

        self._init_model_from_global(global_model)
        self.model.train()

        loader: DataLoader = make_dataloader(self.dataset)
        opt = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE)

        for _ in range(LOCAL_EPOCHS):
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                loss = self.model.loss_fn(self.model(x), y)
                loss.backward()
                opt.step()

        deltas = []
        with torch.no_grad():
            for p_g, p_l in zip(global_model.parameters(), self.model.parameters()):
                deltas.append((p_l - p_g).flatten())

        delta_vec = torch.cat(deltas)

        if self.is_byzantine:
            delta_vec = apply_attack(delta_vec, self.attack_mode)

        return delta_vec.detach().cpu(), w_global_flat

    # ---------------------

    def _call_winterfell_prover(self, trace_path: Path, proof_out: Path) -> bytes:
        """Appeal to the Winterfell proverbial or dummy proof if binary is missing."""
        proof_out.parent.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(
                [
                    WINTERFELL_PROVER_BIN,
                    "--input", str(trace_path),
                    "--output", str(proof_out),
                ],
                check=True,
            )
            proof_bytes = proof_out.read_bytes()
        except Exception:
            proof_bytes = (
                f"DUMMY_PROOF_CLIENT_{self.id}_ROUND_{trace_path.stem}"
            ).encode()
            proof_out.write_bytes(proof_bytes)

        return proof_bytes

    # ---------------------

    def _sign_proof(self, proof: bytes) -> Tuple[bytes, bytes]:
        """Signature ML-DSA-65 (liboqs)."""
        if oqs is None or self._pq_keys is None:
            return b"DUMMY_SIG", b"DUMMY_PUB"

        signature = self._pq_keys.signature.sign(proof)
        return signature, self._pq_keys.public_key

    # ---------------------

    def prepare_update(
        self,
        global_model: SimpleRegressor,
        round_idx: int,
    ) -> ClientUpdate:
        """
        Full customer-side pipeline + metrics.
        """
        metrics = {}

        # 1) Local training
        t0 = time.perf_counter()
        delta_vec, w_global_flat = self.local_train(global_model, round_idx)
        metrics["local_train_ms"] = (time.perf_counter() - t0) * 1000

        # 2) Gradient encoding
        eta = LEARNING_RATE
        t1 = time.perf_counter()
        g_vec = -delta_vec / eta
        metrics["encode_gradient_ms"] = (time.perf_counter() - t1) * 1000

        # 3) Proof generation
        t2 = time.perf_counter()
        trace_path = save_update_for_winterfell(
            self.id,
            round_idx,
            w_global_flat,
            g_vec,
            Path(TRACE_DIR),
            eta=eta,
        )
        proof_path = Path(PROOF_DIR) / f"client_{self.id}_round_{round_idx}.proof"
        proof_bytes = self._call_winterfell_prover(trace_path, proof_path)
        metrics["prove_ms"] = (time.perf_counter() - t2) * 1000
        metrics["proof_size_bytes"] = len(proof_bytes)

        # 4) Signature
        t3 = time.perf_counter()
        signature, pubkey = self._sign_proof(proof_bytes)
        metrics["sign_ms"] = (time.perf_counter() - t3) * 1000
        metrics["signature_size_bytes"] = len(signature)

        # 5) Backup Metrics
        self._save_client_metrics(metrics, round_idx)

        return ClientUpdate(
            client_id=self.id,
            round_idx=round_idx,
            update=delta_vec,
            proof=proof_bytes,
            signature=signature,
            public_key=pubkey,
        )

    # ---------------------

    def _save_client_metrics(self, metrics: dict, round_idx: int):
        """Write the customer metrics in artifacts/metrics_clients.tsv"""
        out = Path("artifacts") / "metrics_clients.tsv"
        out.parent.mkdir(exist_ok=True)

        header_needed = not out.exists()

        with out.open("a") as f:
            if header_needed:
                f.write(
                    "client_id\tround\tlocal_train_ms\tencode_gradient_ms\t"
                    "prove_ms\tproof_size_bytes\tsign_ms\tsignature_size_bytes\n"
                )
            f.write(
                f"{self.id}\t{round_idx}\t"
                f"{metrics['local_train_ms']:.3f}\t"
                f"{metrics['encode_gradient_ms']:.3f}\t"
                f"{metrics['prove_ms']:.3f}\t"
                f"{metrics['proof_size_bytes']}\t"
                f"{metrics['sign_ms']:.3f}\t"
                f"{metrics['signature_size_bytes']}\n"
            )
