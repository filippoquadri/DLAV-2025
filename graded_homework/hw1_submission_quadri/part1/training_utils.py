# **************************************************************************** #
#                                                                              #
#                                                                              #
#    training_utils.py                                                         #
#                                                                              #
#    By: Filippo Quadri <filippo.quadri@epfl.ch>                               #
#                                                                              #
#    Created: 2025/03/10 11:52:30 by Filippo Quadri                            #
#    Updated: 2025/03/16 09:59:15 by Filippo Quadri                            #
#                                                                              #
# **************************************************************************** #

import torch

import torch.optim as optim
import torch.nn as nn

from typing import Optional
import time

def train_model(
        model: nn.Module, 
        train_loader: torch.utils.data.DataLoader, 
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        epochs: int = 10, 
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        early_stopping: bool = False,
        patience: int = 10
):
    """
    Trains a CNN on CIFAR-10.
    
    Parameters:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        device (torch.device): Device to train on (CPU or GPU).
        epochs (int): Number of training epochs.
        criterion (nn.Module): Loss function (default: nn.CrossEntropyLoss).
        optimizer (optim.Optimizer): Optimizer (default: Adam, lr=0.001).
        scheduler (optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
        early_stopping (bool): Whether to use early stopping.
        patience (int): Number of epochs to wait before stopping.

    Returns:
        tuple: (loss_history, acc_history) lists of training loss and validation accuracy.
    """

    if early_stopping:
        assert patience > 0, "Patience must be greater than 0 for early stopping."
        print(f"Using early stopping with patience {patience}.")

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.to(device)

    loss_history = []
    train_history = []
    acc_history = []

    best_val_acc = -1
    patience_counter = 0
    best_model = None

    for epoch in range(epochs):
        start = time.time()
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss)
            else:
                scheduler.step()

        avg_train_loss = running_loss / len(train_loader)
        loss_history.append(avg_train_loss)

        # training accuracy
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_history.append(train_acc)

        # Validation
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        acc_history.append(val_acc)
        
        end = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch {(epoch+1):2d}/{epochs}] Training Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_acc:.2f}%, Training Accuracy: {train_acc:.2f}%, Time: {end-start:.2f}s, lr: {current_lr:.6f}")

        # Early Stopping
        if early_stopping:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model = model.state_dict() 
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}. Restoring best model. Best validation accuracy: {best_val_acc:.2f}%")
                model.load_state_dict(best_model)
                break
    

    final_acc = best_val_acc if early_stopping else val_acc
    print(f"End of the training... Final Validation Accuracy: {final_acc:.2f}")
    if best_model:
        print("Loading in the current model the best version of it during training")
        model.load_state_dict(best_model) 

    return loss_history, acc_history, train_history
