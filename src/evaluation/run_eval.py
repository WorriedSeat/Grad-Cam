from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset.dataset import _load_config, get_rafdb_splits, RAFDBDataset
from src.gradcam.grad_cam import GradCAM
from src.models.efficientemotionnet import EfficientEmotionNet
from src.evaluation.au_masks import (
    EMOTION_NAMES,
    NEUTRAL_IDX,
    build_emotion_mask,
    build_face_mask,
)
from src.evaluation.landmarks import FaceMeshDetector
from src.evaluation.metrics import au_mass_ratio, bg_leakage, pointing_game


INPUT_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _upsample_cam(cam: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (size, size), Image.BILINEAR
    )
    return np.array(img).astype(np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser(
        description="Landmark-aware Grad-CAM evaluation on RAF-DB."
    )
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit to N test samples (for a smoke run)")
    parser.add_argument("--out-csv", type=str, default="eval_results.csv")
    parser.add_argument("--out-per-sample", type=str, default="eval_per_sample.csv")
    parser.add_argument("--out-plot", type=str, default="eval_results.png")
    parser.add_argument("--use-true-label", action="store_true",
                        help="Compute CAM for ground-truth class instead of predicted")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = _load_config()
    model_path = Path(config["models"]["efficientnet"])
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path

    model = EfficientEmotionNet(num_classes=7, dropout=0.4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()

    print("Loading RAF-DB splits...")
    _, _, test_hf = get_rafdb_splits()
    test_ds = RAFDBDataset(test_hf, transform=None)
    print(f"Test set: {len(test_ds)} samples")

    val_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    display_resize = transforms.Resize((INPUT_SIZE, INPUT_SIZE))

    detector = FaceMeshDetector()
    gradcam = GradCAM(model, model.model.features[8])

    n = len(test_ds) if args.num_samples is None else min(args.num_samples, len(test_ds))

    per_class = defaultdict(lambda: defaultdict(list))
    per_sample = []
    skipped = {"no_face": 0, "neutral": 0}
    correct_count = 0
    evaluated = 0

    for i in tqdm(range(n)):
        img_pil, true_label = test_ds[i]
        img_rgb = np.array(display_resize(img_pil))

        input_tensor = val_transform(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            pred_label = int(torch.argmax(logits, dim=1).item())

        target = int(true_label) if args.use_true_label else pred_label
        if target == NEUTRAL_IDX:
            skipped["neutral"] += 1
            continue

        landmarks = detector.detect(img_rgb)
        if landmarks is None:
            skipped["no_face"] += 1
            continue

        cam_input = val_transform(img_pil).unsqueeze(0).to(device)
        cam_small = gradcam(cam_input, target_class=target)
        cam = _upsample_cam(cam_small, INPUT_SIZE)

        emotion_mask = build_emotion_mask(landmarks, target, (INPUT_SIZE, INPUT_SIZE))
        face_mask = build_face_mask(landmarks, (INPUT_SIZE, INPUT_SIZE))

        ratio = au_mass_ratio(cam, emotion_mask)
        leak = bg_leakage(cam, face_mask)
        hit = pointing_game(cam, emotion_mask)
        correct = int(pred_label == int(true_label))

        per_class[target]["au_mass"].append(ratio)
        per_class[target]["leakage"].append(leak)
        per_class[target]["pointing"].append(float(hit))
        per_class[target]["correct"].append(correct)

        per_sample.append({
            "idx": i,
            "true_label": int(true_label),
            "pred_label": pred_label,
            "target": target,
            "correct": correct,
            "au_mass": ratio,
            "leakage": leak,
            "pointing_hit": int(hit),
        })

        evaluated += 1
        correct_count += correct

    gradcam.remove_hooks()
    detector.close()

    print(f"\nEvaluated: {evaluated} | Skipped no-face: {skipped['no_face']} | "
          f"Skipped neutral: {skipped['neutral']}")

    if evaluated == 0:
        print("No samples evaluated — nothing to aggregate.")
        return

    # Per-class summary
    classes = sorted(per_class.keys())
    rows = []
    for c in classes:
        rows.append({
            "class_id": c,
            "class_name": EMOTION_NAMES[c],
            "n": len(per_class[c]["au_mass"]),
            "au_mass_mean": float(np.mean(per_class[c]["au_mass"])),
            "au_mass_std": float(np.std(per_class[c]["au_mass"])),
            "leakage_mean": float(np.mean(per_class[c]["leakage"])),
            "pointing_acc": float(np.mean(per_class[c]["pointing"])),
            "clf_acc": float(np.mean(per_class[c]["correct"])),
        })
    rows.append({
        "class_id": "-",
        "class_name": "OVERALL",
        "n": evaluated,
        "au_mass_mean": float(np.mean([r["au_mass_mean"] for r in rows])),
        "au_mass_std": float(np.mean([r["au_mass_std"] for r in rows])),
        "leakage_mean": float(np.mean([r["leakage_mean"] for r in rows])),
        "pointing_acc": float(np.mean([r["pointing_acc"] for r in rows])),
        "clf_acc": correct_count / evaluated,
    })

    out_csv = Path(args.out_csv)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved summary: {out_csv.resolve()}")

    out_per = Path(args.out_per_sample)
    with out_per.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_sample[0].keys()))
        w.writeheader()
        w.writerows(per_sample)
    print(f"Saved per-sample: {out_per.resolve()}")

    # Bar chart
    import matplotlib.pyplot as plt

    names = [EMOTION_NAMES[c] for c in classes]
    au_means = [np.mean(per_class[c]["au_mass"]) for c in classes]
    au_stds = [np.std(per_class[c]["au_mass"]) for c in classes]
    pt_means = [np.mean(per_class[c]["pointing"]) for c in classes]
    overall_au = float(np.mean([np.mean(per_class[c]["au_mass"]) for c in classes]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(names, au_means, yerr=au_stds, color="steelblue", capsize=4)
    axes[0].axhline(y=overall_au, color="red", linestyle="--", label=f"overall={overall_au:.2f}")
    axes[0].set_title("AU-mass ratio by class")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(names, pt_means, color="teal")
    axes[1].set_title("Pointing game accuracy by class")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out_plot = Path(args.out_plot)
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_plot.resolve()}")


if __name__ == "__main__":
    main()
