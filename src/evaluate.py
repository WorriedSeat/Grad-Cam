import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    top_k_accuracy_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.efficientemotionnet import EfficientEmotionNet
from src.dataset.dataset import _load_config, get_rafdb_splits, RAFDBDataset, make_dataloader

BATCH_SIZE = 64
NUM_CLASSES = 7
EMOTION_LABELS = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

config = _load_config()
model_path = config["models"]["efficientnet"]
if not Path(model_path).is_absolute():
    model_path = REPO_ROOT / model_path

model = EfficientEmotionNet(num_classes=NUM_CLASSES, dropout=0.4)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()
print(f"Loaded model from: {model_path}")

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

print("Loading RAF-DB test split...")
_, _, test_hf = get_rafdb_splits()
test_dataset = RAFDBDataset(test_hf, transform=val_transform)
test_loader = make_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Test samples: {len(test_dataset)}")

all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

overall_acc = (all_preds == all_labels).mean()
top2_acc = top_k_accuracy_score(all_labels, all_probs, k=2)

print("\n" + "=" * 55)
print(f"  Overall Accuracy : {overall_acc:.4f}  ({overall_acc:.2%})")
print(f"  Top-2 Accuracy   : {top2_acc:.4f}  ({top2_acc:.2%})")
print("=" * 55)

print("\nPer-class Report:")
print(classification_report(all_labels, all_preds, target_names=EMOTION_LABELS, digits=4))

cm = confusion_matrix(all_labels, all_preds)
per_class_acc = cm.diagonal() / cm.sum(axis=1)
print("Per-class Accuracy:")
for label, acc in zip(EMOTION_LABELS, per_class_acc):
    print(f"  {label:<10} {acc:.4f}  ({acc:.2%})")

fig, ax = plt.subplots(figsize=(9, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMOTION_LABELS)
disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
ax.set_title("Confusion Matrix — EfficientEmotionNet on RAF-DB Test Set")
plt.tight_layout()

out_path = REPO_ROOT / "confusion_matrix.png"
plt.savefig(out_path, dpi=150)
print(f"\nConfusion matrix saved to: {out_path}")
plt.show()
