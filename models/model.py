# models/model.py

import torch
import torch.nn as nn


class SimpleRegressor(nn.Module):
    """
    Ultra-simple model for the PoC :
    y = Wx + b (linear regression).
    """

    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def loss_fn(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return nn.MSELoss()(y_pred, y_true)
