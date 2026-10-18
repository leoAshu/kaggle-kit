import torch


def accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)

    if predictions.numel() != targets.numel():
        raise ValueError('Predictions and targets must have equal sizes.')

    return (predictions == targets).float().mean().item()
