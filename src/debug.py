import matplotlib.pyplot as plt
import torch
import os

from torchvision import datasets
from PIL import Image

from data import make_transforms


train_tf, eval_tf = make_transforms(224)
val_ds = datasets.ImageFolder(root=f"data/val", transform=eval_tf)

for i in range(5):
    img, label = val_ds[i]

    img = img.clone()
    img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    img = img.permute(1, 2, 0).numpy()
    # img = img.clamp(0, 1)

    plt.imshow(img)
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

img_path, _, = val_ds.samples[0]

original = Image.open(img_path).convert('RGB')
transformed = eval_tf(original)

transformed = transformed.clone()
transformed = transformed * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
transformed = transformed * torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
transformed = transformed.permute(1, 2, 0).numpy()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original)
plt.title("Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(transformed)
plt.title("After Transform")
plt.axis('off')
plt.show()