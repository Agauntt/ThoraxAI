import argparse, yaml, torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from data import make_loaders

def main(cfg, ckpt_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, _, test_loader, class_to_idx = make_loaders(cfg["data_dir"], cfg["img_size"],
                                                    cfg["batch_size"], cfg["num_workers"])
    ckpt = torch.load(ckpt_path, map_location=device)
    from model import build_model
    model = build_model(cfg["model_name"], cfg["pretrained"], num_classes=2).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_probs, all_y = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_y.extend(y.numpy().tolist())

    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    print("AUC:", roc_auc_score(all_y, all_probs))
    print("Confusion Matrix:\n", confusion_matrix(all_y, preds))
    print(classification_report(all_y, preds, digits=4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    args = ap.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    main(cfg, args.ckpt)