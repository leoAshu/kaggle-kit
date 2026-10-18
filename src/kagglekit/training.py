import pandas as pd
import torch
from torch import nn

from kagglekit.data import Dataset
from kagglekit.metrics import accuracy


class Trainer:
    def __init__(
        self,
        loss_function: nn.Module,
        optimizer: torch.optim.Optimizer,
        max_epochs: int = 100,
    ) -> None:
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.max_epochs = max_epochs

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    def prepare_model(self, model):
        self.model = model.to(self.device)

    def prepare_batch(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        X, Y = batch
        
        return X.to(self.device), Y.to(self.device)
    
    def fit(self, model: nn.Module, data: Dataset):
        self.prepare_model(model)
        
        train_loader = data.train_loader
        val_loader = data.val_loader
        
        train_losses = []
        val_losses = []
        val_accs = []
        for epoch in range(self.max_epochs):
            self.model.train()

            train_loss = 0.0
            for batch in train_loader:
                X_batch, Y_batch = self.prepare_batch(batch)
                
                logits = self.model(X_batch)
    
                loss = self.loss_function(logits, Y_batch)

                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

                train_loss += loss.item() * len(X_batch)
            
            train_losses.append(train_loss / len(train_loader.dataset))

            if val_loader is not None:
                self.model.eval()
                
                val_loss = 0.0
                val_acc = 0.0
                for batch in val_loader:
                    X_batch, Y_batch = self.prepare_batch(batch)
                    
                    with torch.no_grad():
                        logits = self.model(X_batch)
                        loss = self.loss_function(logits, Y_batch)

                        val_loss += loss.item() * len(X_batch)
                        val_acc += accuracy(self.model.predict(X_batch), Y_batch) * len(X_batch)
                
                val_losses.append(val_loss / len(val_loader.dataset))
                val_accs.append(val_acc / len(val_loader.dataset))
        
        history = {'train_loss': train_losses}
        if val_loader is not None:
            history['val_loss'] = val_losses
            history['val_accuracy'] = val_accs

        return pd.DataFrame(history)
