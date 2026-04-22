from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def _ensure_model(model_path: Path) -> Path:
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face_landmarker model to {model_path} ...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


class FaceMeshDetector:
    """MediaPipe Tasks FaceLandmarker wrapper.

    Returns (468, 2) pixel-space landmark coordinates, or None if no face.
    Uses the first 468 of the model's 478 points (the last 10 are iris).
    """

    def __init__(self, model_path: Path | None = None):
        if model_path is None:
            cache_dir = Path.home() / ".cache" / "mediapipe"
            model_path = cache_dir / "face_landmarker.task"
        model_path = _ensure_model(Path(model_path))

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, image_rgb: np.ndarray):
        if image_rgb.dtype != np.uint8:
            raise ValueError("FaceMeshDetector expects uint8 RGB image")
        h, w = image_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self._detector.detect(mp_image)
        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0]
        pts = np.array([[p.x * w, p.y * h] for p in lm[:468]], dtype=np.float32)
        return pts

    def close(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
