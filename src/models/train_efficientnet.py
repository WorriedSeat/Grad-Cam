import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from sklearn.utils.class_weight import compute_class_weight

from src.dataset.dataset import _load_config, get_data_splits, FERDataset

DATA_TRANSFORMS = {
    "train":transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandAugment(),                    
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ]),
    "val":transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
}

def train(model, epochs:int, max_patience:int, device,
          train_loader, val_loader, test_loader,
          criterion, optimizer, scheduler,
          save_path:str):

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

        train_loss = running_loss / len(train_loader)

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

        val_acc = correct / total
        val_loss /=len(val_loader)

        print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print("   >>> Best model saved")
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print("Early Stopping")
                break
        
        if scheduler != None:
            scheduler.step()

def efficientnet_train():
    BATCH_SIZE = 64
    EPOCHS = 100
    MAX_PATIENCE = 12
    config = _load_config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_df, val_df, test_df = get_data_splits()

    train_dataset = FERDataset(train_df, DATA_TRANSFORMS.get("train"))
    val_dataset   = FERDataset(val_df,   DATA_TRANSFORMS.get("val"))
    test_dataset  = FERDataset(test_df,  DATA_TRANSFORMS.get("val"))

    train_loader = FERDataset.dataloader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = FERDataset.dataloader(val_dataset, BATCH_SIZE, shuffle=False)
    test_loader = FERDataset.dataloader(test_dataset, BATCH_SIZE, shuffle=False)

    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
    model = model.to(device)

    class_weights = compute_class_weight('balanced', classes=np.unique(train_df['emotion']), y=train_df['emotion'])
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    train(model, EPOCHS, MAX_PATIENCE, device, 
          train_loader, val_loader, test_loader, 
          criterion, optimizer, scheduler,
          config["models"]["efficientnet"])

if __name__ == "__main__":
    efficientnet_train()