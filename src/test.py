import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.gradcam.grad_cam import GradCAM
from src.models.efficientemotionnet import EfficientEmotionNet
from src.dataset.dataset import _load_config

INPUT_SIZE = 224
EMOTION_LABELS = {
    0: "Anger", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Neutral", 5: "Sad", 6: "Surprise",
}


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM on EfficientNet emotion classifier")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--out", type=str, default="gradcam_result.png",
                        help="Path to save the visualization (default: gradcam_result.png)")
    parser.add_argument("--show", action="store_true", help="Try to open a GUI window")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = _load_config()
    model_path = Path(config["models"]["efficientnet"])
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path

    model = EfficientEmotionNet(num_classes=7, dropout=0.4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    img_pil = Image.open(args.image).convert("RGB")

    display_transform = transforms.Resize((INPUT_SIZE, INPUT_SIZE))
    img_display = np.array(display_transform(img_pil))

    val_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = val_transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        predicted_class = torch.argmax(logits, dim=1).item()
        probabilities = torch.softmax(logits, dim=1)[0]

    print(f"\nPREDICTED: {EMOTION_LABELS[predicted_class]}")
    print(f"Confidence: {probabilities[predicted_class]:.1%}")
    print("\nAll probabilities:")
    for i, prob in enumerate(probabilities):
        marker = " <-" if i == predicted_class else ""
        print(f"  {EMOTION_LABELS[i]:8s}  {prob:.1%}{marker}")

    # Target layer for EfficientNet-B0: last features block
    target_layer = model.model.features[8]
    gradcam = GradCAM(model, target_layer)

    input_for_cam = val_transform(img_pil).unsqueeze(0).to(device)
    cam_np = gradcam(input_for_cam, target_class=predicted_class)
    gradcam.remove_hooks()

    # Upsample CAM to display size
    from PIL import Image as PILImage
    cam_img = PILImage.fromarray((cam_np * 255).astype(np.uint8)).resize(
        (INPUT_SIZE, INPUT_SIZE), PILImage.BILINEAR
    )
    cam_resized = np.array(cam_img).astype(np.float32) / 255.0

    colormap = plt.get_cmap("jet")
    heatmap_rgb = colormap(cam_resized)[:, :, :3]
    img_norm = img_display.astype(np.float32) / 255
    overlay = np.clip(0.55 * img_norm + 0.45 * heatmap_rgb, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(img_display)
    axes[0].set_title("Input")
    axes[0].axis("off")

    im = axes[1].imshow(cam_resized, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title(
        f"{EMOTION_LABELS[predicted_class]} ({probabilities[predicted_class]:.1%})"
    )
    axes[2].axis("off")

    plt.tight_layout()
    out_path = Path(args.out)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path.resolve()}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
