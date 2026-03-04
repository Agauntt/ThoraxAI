import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split


def make_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.2)),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # minor patient positioning shifts
        transforms.ColorJitter(brightness=0.2, contrast=0.2),        # exposure/contrast variation between machines
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3), # varies image clarity
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.2)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    return train_tf, eval_tf


def make_loaders(data_dir: str, img_size: int, batch_size: int, num_workers: int):
    train_tf, eval_tf = make_transforms(img_size)

    full_train = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_tf)
    # val_ds   = datasets.ImageFolder(root=f"{data_dir}/val",   transform=eval_tf)
    test_ds  = datasets.ImageFolder(root=f"{data_dir}/test",  transform=eval_tf)

    # Split full_train into train and val
    val_size = int(0.15 * len(full_train))
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # Weighted sampling to handle class imbalance
    train_targets = [full_train.targets[i] for i in train_ds.indices]
    class_counts = torch.bincount(torch.tensor(train_targets))
    class_weights = 1.0 / (class_counts.float())
    sample_weights = class_weights[train_targets]
    sampler = WeightedRandomSampler(sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, full_train.class_to_idx