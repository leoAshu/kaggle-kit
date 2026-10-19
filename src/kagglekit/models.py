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
        hidden_features: int = 8,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        logits = self(X)

        return (logits >= 0).long().squeeze(-1)
