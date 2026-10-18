import torch
from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(X)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self(x)
        
        return (logits >= 0).long().squeeze(-1)
