# **************************************************************************** #
#                                                                              #
#                                                                              #
#    data_utils.py                                                             #
#                                                                              #
#    By: Filippo Quadri <filippo.quadri@epfl.ch>                               #
#                                                                              #
#    Created: 2025/03/10 11:57:31 by Filippo Quadri                            #
#    Updated: 2025/03/11 07:40:08 by Filippo Quadri                            #
#                                                                              #
# **************************************************************************** #

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def get_cifar10_dataloader(batch_size=64, augment=False, num_workers=2):
    """
    Returns DataLoaders for the CIFAR-10 dataset, split into training (40k) and validation (10k) sets.
    
    Parameters:
        batch_size (int): Batch size for the DataLoader.
        augment (bool): If True, applies data augmentation to the training set.
        num_workers (int): Number of worker threads for data loading.

    Returns:
        tuple: (train_loader, val_loader) DataLoaders for training and validation.
    """
    transform_list = []
    
    if augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
        ])
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    transform = transforms.Compose(transform_list)
    
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    
    train_dataset, val_dataset = random_split(dataset, [40000, 10000])
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader

def visualize_dataloader(dataloader):
    """
    Visualizes a batch of images from a DataLoader.
    
    Parameters:
        dataloader (DataLoader): DataLoader to visualize.
    """
    # Get a batch of training data
    inputs, classes = next(iter(dataloader))

    # Denormalize the images
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.247, 0.243, 0.261])
    inputs = inputs.numpy().transpose((0, 2, 3, 1))  # Convert to HWC format
    inputs = std * inputs + mean  # Denormalize
    inputs = np.clip(inputs, 0, 1)  # Clip to valid range

    # Make a grid from batch
    out = torchvision.utils.make_grid(torch.tensor(inputs).permute(0, 3, 1, 2))

    # Plot the images
    plt.figure(figsize=(10, 10))
    plt.imshow(out.permute(1, 2, 0))  # Convert to HWC format for imshow
    plt.axis("off")
    
    # Add labels
    for i in range(inputs.shape[0]):
        plt.text(
            i % 8 * 34 + 16,  # x coordinate
            i // 8 * 34 + 8,  # y coordinate
            str(classes[i].item()),  # label
            color="white",
            fontsize=12,
            ha="center",
            va="center",
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
        )
    
    plt.show()

def compare_losses(losses: List[Tuple[str, List[float], List[float], List[float]]]):
    """
    Plots multiple loss curves and accuracy curves on separate subplots with consistent coloring and aesthetic improvements.
    
    Parameters:
        losses (List[Tuple[str, List[float], List[float], List[float]]]): List of tuples (label, train_loss, val_acc, train_acc).
    """
    # Set color palette for consistency
    sns.set_palette("Paired")  # You can use other palettes like "deep", "dark", etc.

    # Create two subplots, one for loss and one for accuracy
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Loss subplot (top)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Accuracy subplot (bottom)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)

    # Plot the data
    for i, (label, train_loss, val_acc, train_acc) in enumerate(losses):
        epochs = range(1, len(train_loss) + 1)

        # Alternate colors based on index
        color_1 = sns.color_palette()[(2*i + 1) % len(sns.color_palette())]
        color_2 = sns.color_palette()[2*i % len(sns.color_palette())]

        # Plot the loss curve on the first subplot (top)
        ax1.plot(epochs, train_loss, color=color_1, label=f"{label} Train Loss", linewidth=2)
        
        # Plot the accuracy curve on the second subplot (bottom)
        ax2.plot(epochs, val_acc, color=color_1, label=f"{label} Val Acc", linewidth=2)
        ax2.plot(epochs, train_acc, color=color_2, label=f"{label} Train Acc", linewidth=2)

    # Add gridlines for better readability
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Make x-axis only integer
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Adjust legend positioning to avoid overlap
    fig.tight_layout()
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=len(losses), fontsize=10)

    plt.show()

def save_model(model: torch.nn.Module, model_name: str, save_dir: str = "models"):
    """
    Saves a PyTorch model inside the specified directory.

    Parameters:
        model (torch.nn.Module): The model to save.
        model_name (str): The filename for the saved model (e.g., "my_model").
        save_dir (str, optional): The directory where the model will be saved. Defaults to "models".

    Returns:
        str: The full path of the saved model file.
    """

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    if not model_name.endswith(".pth"):
        model_name += ".pth"

    # Full path of the saved model
    model_path = os.path.join(save_dir, model_name)

    # Save the model state dictionary
    torch.save(model.state_dict(), model_path)

    print(f"Model saved to {model_path}")
    return model_path

def load_model(model: torch.nn.Module, model_path: str):
    """
    Loads a PyTorch model from the specified file.
    """
    model.load_state_dict(torch.load(model_path))
    return model
