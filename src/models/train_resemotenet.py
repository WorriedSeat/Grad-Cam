from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from tqdm.auto import tqdm
from sklearn.utils.class_weight import compute_class_weight

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.dataset import (
    _load_config,
    get_rafdb_splits,
    RAFDBDataset,
    make_dataloader,
)
from src.models.resemotenet import ResEmoteNet

DATA_TRANSFORMS = {
    "train": transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]),
    "val": transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]),
}


def train(
    model,
    epochs: int,
    max_patience: int,
    device,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    save_path: str,
):
    best_acc = 0.0
    patience = 0

    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / max(1, len(train_loader.dataset))

        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, pred = outputs.max(1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        val_acc = correct / total if total > 0 else 0.0
        val_loss /= max(1, len(val_loader.dataset))

        print(
            f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_full = Path(save_path)
            if not save_full.is_absolute():
                save_full = REPO_ROOT / save_full
            save_full.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_full)
            print("   >>> Best model saved")
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print("Early Stopping")
                break

        if scheduler is not None:
            scheduler.step(val_acc)


def resemotenet_train():
    BATCH_SIZE = 16
    EPOCHS = 80
    MAX_PATIENCE = 12
    config = _load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_hf, val_hf, test_hf = get_rafdb_splits(val_ratio=0.1, seed=42)

    train_dataset = RAFDBDataset(train_hf, DATA_TRANSFORMS["train"])
    val_dataset = RAFDBDataset(val_hf, DATA_TRANSFORMS["val"])
    test_dataset = RAFDBDataset(test_hf, DATA_TRANSFORMS["val"])

    train_loader = make_dataloader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = make_dataloader(val_dataset, BATCH_SIZE, shuffle=False)
    test_loader = make_dataloader(test_dataset, BATCH_SIZE, shuffle=False)

    model = ResEmoteNet(num_classes=7, input_size=100)
    model = model.to(device)

    train_labels = [train_dataset[i][1] for i in range(min(5000, len(train_dataset)))]
    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(train_labels),
        y=train_labels,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=5
    )

    save_path = config["models"]["resemotenet"]
    train(
        model,
        EPOCHS,
        MAX_PATIENCE,
        device,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        save_path,
    )

if __name__ == "__main__":
    resemotenet_train()
