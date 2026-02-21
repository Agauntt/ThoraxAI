from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def make_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    return train_tf, eval_tf


def make_loaders(data_dir: str, img_size: int, batch_size: int, num_workers: int):
    train_tf, eval_tf = make_transforms(img_size)

    train_ds = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_tf)
    val_ds   = datasets.ImageFolder(root=f"{data_dir}/val",   transform=eval_tf)
    test_ds  = datasets.ImageFolder(root=f"{data_dir}/test",  transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, train_ds.class_to_idx