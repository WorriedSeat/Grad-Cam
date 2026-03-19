import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.gradcam.grad_cam import compute_gradcam
from src.models.resemotenet import ResEmoteNet
from src.dataset.dataset import _load_config


# ── Устройство ──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# ── Модель ResEmoteNet ──────────────────────────────────────────────────────
config = _load_config()
model_path = config["models"]["resemotenet"]
if not Path(model_path).is_absolute():
    model_path = REPO_ROOT / model_path

model = ResEmoteNet(num_classes=7, input_size=100)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

emotion_labels = {
    0: "Anger", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Sad", 5: "Surprise", 6: "Neutral",
}

your_image_path = "sup2.jpg"

img_pil = Image.open(your_image_path).convert("RGB")

# ResEmoteNet expects 100x100 input
display_transform = transforms.Compose([
    transforms.Resize((100, 100)),
])
img_display = np.array(display_transform(img_pil))

val_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

input_tensor = val_transform(img_pil).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(input_tensor)
    predicted_class = torch.argmax(logits, dim=1).item()
    probabilities = torch.softmax(logits, dim=1)[0]

print(f"\nПРЕДСКАЗАННАЯ ЭМОЦИЯ: {emotion_labels[predicted_class]}")
print(f"Уверенность модели: {probabilities[predicted_class]:.1%}")
print("\nВсе вероятности:")
for i, prob in enumerate(probabilities):
    marker = " ◄" if i == predicted_class else ""
    print(f"  {emotion_labels[i]:8s}  {prob:.1%}{marker}")

cam = compute_gradcam(
    model,
    input_tensor,
    predicted_class,
    target_layer_name=None,
)
cam_np = cam.cpu().numpy()

colormap = cm.get_cmap("jet")
heatmap_rgb = colormap(cam_np)[:, :, :3]
img_norm = img_display.astype(np.float32) / 255

alpha = 0.45
overlay = (1 - alpha) * img_norm + alpha * heatmap_rgb
overlay = np.clip(overlay, 0, 1)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

axes[0].imshow(img_display)
axes[0].set_title("Исходное фото", fontsize=12)
axes[0].axis("off")

im = axes[1].imshow(cam_np, cmap="jet", vmin=0, vmax=1)
axes[1].set_title("Grad-CAM (тепловая карта)", fontsize=12)
axes[1].axis("off")
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

axes[2].imshow(overlay)
axes[2].set_title(
    f"Overlay — {emotion_labels[predicted_class]} "
    f"({probabilities[predicted_class]:.1%})",
    fontsize=12,
)
axes[2].axis("off")

plt.tight_layout()
plt.show()
