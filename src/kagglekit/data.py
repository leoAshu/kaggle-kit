import torch


class Dataset:
    def __init__(
        self, 
        X_train: torch.Tensor, 
        Y_train: torch.Tensor,
        X_val: torch.Tensor | None = None,
        Y_val: torch.Tensor | None = None,
        batch_size: int = 32
    ) -> None:
        self.batch_size = batch_size
        
        self.train_loader = self.get_tensorloader(X_train, Y_train, True)
        self.val_loader = None
        if X_val is not None and Y_val is not None:
            self.val_loader = self.get_tensorloader(X_val, Y_val, False)

    def get_tensorloader(self, X: torch.Tensor, Y: torch.Tensor, shuffle=False) -> torch.utils.data.DataLoader:
        tensor_dataset = torch.utils.data.TensorDataset(X, Y)
        
        return torch.utils.data.DataLoader(tensor_dataset, self.batch_size, shuffle)
