from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from torchvision import datasets, transforms

def load_cifar10_dataset(
    root,
    train=False,
    download=True
): 
    """
    Load CIFAR-10 with pixel values in [0, 1]
    """
    root = Path(root)

    transform = transforms.ToTensor()

    return datasets.CIFAR10(
        root=root,
        train=train,
        transform=transform,
        download=download
    )

def create_cifar10_datasets(
    root,
    val_size=5000,
    seed=42,
    download=True,
):
    """
    Create CIFAR-10 train and test datasets
    """
    root = Path(root)

    train_transform = transforms.Compose([
        transforms.RandomCrop(
            32, 
            padding=4
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    eval_transform = transforms.ToTensor()

    train_full = datasets.CIFAR10(
        root=root,
        train=True,
        transform=train_transform,
        download=download,
    )

    val_full = datasets.CIFAR10(
        root=root,
        train=True,
        transform=eval_transform,
        download=False,
    )

    test_dataset = datasets.CIFAR10(
        root=root,
        train=False,
        transform=eval_transform,
        download=download,
    )

    generator = torch.Generator().manual_seed(seed)


    indices = torch.randperm(
        len(train_full),
        generator=generator,
    )

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = Subset(
        train_full,
        train_indices.tolist(),
    )

    val_dataset = Subset(
        val_full,
        val_indices.tolist(),
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )

def create_cifar10_loaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size,
    num_workers,
    pin_memory=False,
    seed=42,
):
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )