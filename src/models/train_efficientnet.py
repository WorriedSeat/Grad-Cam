from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.dataset import get_rafdb_splits, RAFDBDataset
from src.models.efficientemotionnet import EfficientEmotionNet

CFG = {
    "val_ratio":       0.1,
    "seed":            42,
    "input_size":      224,
    "batch_size":      32,
    "num_workers":     0,
    "num_classes":     7,
    "dropout":         0.4,
    "lr_backbone":     1e-4,
    "lr_head":         3e-4,
    "weight_decay":    1e-2,
    "label_smoothing": 0.1,
    "epochs":          50,
    "t_max":           30,
    "eta_min":         1e-6,
    "patience":        10,
    "save_path":       "best_efficientnet_emotion.pth",
}

EMOTION_LABELS = {
    0: "Anger", 1: "Disgust", 2: "Fear",
    3: "Happy", 4: "Sad",    5: "Surprise", 6: "Neutral",
}


sz = CFG["input_size"]

DATA_TRANSFORMS = {
    "train": transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.10),
                                 ratio=(0.3, 3.3), value=0),
    ]),
    "val": transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]),
}

def train(model, epochs: int, max_patience: int, device,
          train_loader, val_loader,
          criterion, optimizer, scheduler,
          save_path: str):

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    best_acc = 0.0
    patience = 0

    print("Training started...")
    print("=" * 75)

    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader,
                                   desc=f"Epoch {epoch+1} [Train]",
                                   leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        correct, total, val_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss_sum += criterion(outputs, labels).item() * inputs.size(0)
                correct += (outputs.max(1)[1] == labels).sum().item()
                total   += labels.size(0)

        val_acc  = correct / total
        val_loss = val_loss_sum / len(val_loader.dataset)
        cur_lr   = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(cur_lr)

        print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"LR: {cur_lr:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"   >>> Best model saved — val_acc={best_acc:.4f}")
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping (patience={max_patience})")
                break

        if scheduler is not None:
            scheduler.step()

    print("=" * 75)
    print(f"Best Val Acc: {best_acc:.4f}")
    return history

def efficientnet_train():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_hf, val_hf, test_hf = get_rafdb_splits(CFG["val_ratio"], CFG["seed"])
    print(f"Train: {len(train_hf)}, Val: {len(val_hf)}, Test: {len(test_hf)}")

    train_dataset = RAFDBDataset(train_hf, DATA_TRANSFORMS["train"])
    val_dataset   = RAFDBDataset(val_hf,   DATA_TRANSFORMS["val"])

    train_loader = DataLoader(train_dataset, batch_size=CFG["batch_size"],
                              shuffle=True,  num_workers=CFG["num_workers"],
                              pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=CFG["batch_size"],
                              shuffle=False, num_workers=CFG["num_workers"],
                              pin_memory=True)

    model = EfficientEmotionNet(
        num_classes=CFG["num_classes"],
        dropout=CFG["dropout"],
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,} ({100 * trainable / total:.1f}%)")

    sample_size  = min(5000, len(train_dataset))
    train_labels = [train_dataset[i][1] for i in range(sample_size)]
    cw = compute_class_weight("balanced",
                              classes=np.unique(train_labels),
                              y=train_labels)
    cw_tensor = torch.tensor(cw, dtype=torch.float32).to(device)
    print("Class weights:", {EMOTION_LABELS[i]: f"{w:.3f}" for i, w in enumerate(cw)})

    criterion = nn.CrossEntropyLoss(weight=cw_tensor,
                                    label_smoothing=CFG["label_smoothing"])

    optimizer = optim.AdamW([
        {"params": model.model.features.parameters(),   "lr": CFG["lr_backbone"]},
        {"params": model.model.classifier.parameters(), "lr": CFG["lr_head"]},
    ], weight_decay=CFG["weight_decay"])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["t_max"], eta_min=CFG["eta_min"]
    )

    train(model, CFG["epochs"], CFG["patience"], device,
          train_loader, val_loader,
          criterion, optimizer, scheduler,
          CFG["save_path"])


if __name__ == "__main__":
    efficientnet_train()