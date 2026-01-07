import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

def get_dataloaders(data_dir="./data", batch_size=128, num_workers=2, val_ratio=0.1, seed=42):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),     # 위치 약간 이동
        transforms.RandomHorizontalFlip(),        # 좌우반전
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    full_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_set   = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size

    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=g)

    # validation은 augmentation 없이 평가하는 게 보통 더 공정함 → transform 교체
    val_set.dataset.transform = test_tf

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
