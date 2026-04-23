from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
from scipy.ndimage import binary_dilation


# MediaPipe FaceMesh landmark indices per anatomical region.
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
NOSE = [1, 2, 4, 5, 6, 19, 94, 125, 141, 168, 188, 195, 197, 198, 236, 174, 399]
MOUTH = [61, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 146, 91, 181]
LEFT_CHEEK = [205, 50, 36, 101, 137, 177, 147, 187]
RIGHT_CHEEK = [425, 280, 266, 330, 366, 401, 376, 411]
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109,
]

REGIONS: dict[str, list[int]] = {
    "brows": LEFT_EYEBROW + RIGHT_EYEBROW,
    "eyes": LEFT_EYE + RIGHT_EYE,
    "nose": NOSE,
    "mouth": MOUTH,
    "cheeks": LEFT_CHEEK + RIGHT_CHEEK,
}

# EMFACS-simplified: which anatomical regions should dominate the explanation for each class.
# HF dataset deanngkl/raf-db-7emotions order:
# 0 Anger, 1 Disgust, 2 Fear, 3 Happy, 4 Neutral, 5 Sad, 6 Surprise.
EMOTION_REGIONS: dict[int, list[str]] = {
    0: ["brows", "eyes", "mouth"],
    1: ["nose", "mouth"],
    2: ["brows", "eyes", "mouth"],
    3: ["mouth", "cheeks"],
    4: [],
    5: ["brows", "mouth"],
    6: ["brows", "eyes", "mouth"],
}

EMOTION_NAMES: dict[int, str] = {
    0: "Anger", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Neutral", 5: "Sad", 6: "Surprise",
}

NEUTRAL_IDX = 4


def _hull_mask(pts: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(pts) < 3:
        return mask.astype(bool)
    hull = cv2.convexHull(pts.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(bool)


def build_region_mask(
    landmarks: np.ndarray,
    region_names: Iterable[str],
    hw: tuple[int, int],
    dilate_iters: int = 6,
) -> np.ndarray:
    mask = np.zeros(hw, dtype=bool)
    for name in region_names:
        mask |= _hull_mask(landmarks[REGIONS[name]], hw)
    if dilate_iters > 0:
        mask = binary_dilation(mask, iterations=dilate_iters)
    return mask


def build_emotion_mask(
    landmarks: np.ndarray,
    emotion_idx: int,
    hw: tuple[int, int],
    dilate_iters: int = 6,
):
    regions = EMOTION_REGIONS[emotion_idx]
    if not regions:
        return None
    return build_region_mask(landmarks, regions, hw, dilate_iters)


def build_face_mask(landmarks: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    return _hull_mask(landmarks[FACE_OVAL], hw)
