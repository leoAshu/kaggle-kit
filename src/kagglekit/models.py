from collections.abc import Sequence

import torch
from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(X)

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        logits = self(X)

        return (logits >= 0).long().squeeze(-1)


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int | Sequence[int] = 8,
    ) -> None:
        super().__init__()

        # An int is one hidden layer of that width; a sequence gives one layer per entry.
        if isinstance(hidden_features, int):
            hidden_features = [hidden_features]

        features = [in_features, *hidden_features, out_features]

        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(features[:-2], features[1:-1], strict=True):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])

        # No ReLU after the output layer: forward returns logits.
        layers.append(nn.Linear(features[-2], features[-1]))

        self.net = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        logits = self(X)

        return (logits >= 0).long().squeeze(-1)
