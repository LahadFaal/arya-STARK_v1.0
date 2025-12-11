# utils/data.py

import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Tuple
from config import INPUT_DIM, NUM_CLIENTS, BATCH_SIZE, RANDOM_SEED

torch.manual_seed(RANDOM_SEED)


def generate_synthetic_data(
    n_samples: int = 10_000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Toy dataset : y = Wx + noise
    """
    X = torch.randn(n_samples, INPUT_DIM)
    true_w = torch.randn(INPUT_DIM, 1)
    y = X @ true_w + 0.1 * torch.randn(n_samples, 1)
    return X, y


def split_dataset_among_clients(
    X: torch.Tensor, y: torch.Tensor, num_clients: int = NUM_CLIENTS
) -> List[TensorDataset]:
    n = X.size(0)
    assert n >= num_clients
    per_client = n // num_clients
    datasets = []
    for i in range(num_clients):
        start = i * per_client
        end = (i + 1) * per_client if i < num_clients - 1 else n
        datasets.append(TensorDataset(X[start:end], y[start:end]))
    return datasets


def make_dataloader(dataset: TensorDataset) -> DataLoader:
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
