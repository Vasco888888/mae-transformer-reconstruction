import torch
from torchvision import datasets, transforms

def build_dataset(args, is_train=True):
    """
    Build CIFAR-10 set with augmentations for training or deterministic for validation.
    """
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    dataset = datasets.CIFAR10(
        root='./data', 
        train=is_train, 
        download=True, 
        transform=transform
    )

    return dataset
