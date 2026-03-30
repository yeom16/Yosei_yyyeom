import os
from typing import Tuple
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

def build_train_transform(args):
    return transforms.Compose([
        transforms.RandomResizedCrop(args.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=args.randaugment_n, magnitude=args.randaugment_m),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def build_val_transform(args):
    return transforms.Compose([
        transforms.Resize(int(args.input_size / 0.875)),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def build_imagenet_loaders(args) -> Tuple[DataLoader, DataLoader, object]:
    train_dataset = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=build_train_transform(args))
    val_dataset = datasets.ImageFolder(os.path.join(args.data_path, "val"), transform=build_val_transform(args))

    if getattr(args, "distributed", False):
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    return train_loader, val_loader, train_sampler
