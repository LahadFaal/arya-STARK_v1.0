# main.py

from __future__ import annotations

import random
from typing import List

import torch

from config import NUM_CLIENTS, BYZ_FRACTION, NUM_ROUNDS, RANDOM_SEED
from utils.data import generate_synthetic_data, split_dataset_among_clients
from clients.client import Client
from server.server import Server


def build_clients(datasets) -> List[Client]:
    num_byz = int(NUM_CLIENTS * BYZ_FRACTION)
    byz_ids = set(random.sample(range(NUM_CLIENTS), num_byz))
    clients: List[Client] = []
    for i, ds in enumerate(datasets):
        is_byz = i in byz_ids
        attack_mode = "random" if is_byz else "honest"
        clients.append(Client(i, ds, is_byzantine=is_byz, attack_mode=attack_mode))
    return clients


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # 1) Data + clients
    X, y = generate_synthetic_data()
    datasets = split_dataset_among_clients(X, y)
    clients = build_clients(datasets)

    # 2) Server + global model
    server = Server()

    for t in range(NUM_ROUNDS):
        print(f"\n=== Round {t} ===")
        global_model = server.get_global_model()

        # 3) Each client prepares their signed and verified update
        updates = []
        for c in clients:
            u = c.prepare_update(global_model, round_idx=t)
            updates.append(u)

        # 4) Server-side Byzantine-resilient aggregation
        f = int(NUM_CLIENTS * BYZ_FRACTION)
        agg_delta = server.aggregate(updates, f=f)

        # 5) Global model update
        server.apply_update(agg_delta)

        # Simple log: update standard
        print(f"||Δw_agg||_2 = {agg_delta.norm().item():.4f}")

    print("\nTraining session finished.")


if __name__ == "__main__":
    main()
