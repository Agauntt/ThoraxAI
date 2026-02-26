import argparse, yaml, time
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torchmetrics.classification import BinaryAUROC

from utils import set_seed, ensure_dir
from data import make_loaders
from model import build_model


def run_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    auroc = BinaryAUROC().to(device)

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
           logits = model(x)
           loss = loss_fn(logits, y)

           if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1]
        auroc.update(probs, y)
    
    avg_loss = total_loss / len(loader.dataset)
    auc = float(auroc.compute().detach().cpu())
    return avg_loss, auc


def main(cfg):
    set_seed(cfg["seed"])
    ensure_dir(cfg["output_dir"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, _, class_to_idx = make_loaders(cfg["data_dir"], cfg["img_size"],
                                                            cfg["batch_size"], cfg["num_workers"])
    model = build_model(cfg["model_name"], cfg["pretrained"], num_classes=2).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best_auc = -1.0
    best_path = f'{cfg["output_dir"]}/best.pt'
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-6)

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_auc = run_epoch(model, train_loader, optimizer, device, train=True)
        va_loss, va_auc = run_epoch(model, val_loader, optimizer, device, train=False)
        scheduler.step()

        print(f"Epoch {epoch:02d} | "
              f"train loss {tr_loss:.4f} auc {tr_auc:.4f} | "
              f"val loss {va_loss:.4f} auc {va_auc:.4f} | "
              f"{time.time()-t0:.1f}s")
        
        if va_auc > best_auc:
            best_auc = va_auc
            torch.save({'model_state': model.state_dict(), 'cfg': cfg, 'class_to_idx': class_to_idx}, best_path)
            

    print(f"Best val AUROC: {best_auc:.4f} saved to {best_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    main(cfg)