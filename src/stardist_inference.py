"""
stardist_inference.py
---------------------
Handles everything related to StarDist:
  - loading the model
  - normalizing the volume
  - running inference

Automatically switches between:
  - predict_instances_big  for large full volumes (tiled)
  - predict_instances      for small ROIs (direct)
"""
from pathlib import Path

import numpy as np
from csbdeep.utils import normalize
from stardist.models import StarDist3D

MODELS_DIR = Path(__file__).parent.parent / "models"

# predict_instances_big requires every dim >= block_size
# if any dim is smaller, use predict_instances directly
BLOCK_SIZE = (64, 128, 128)


def load_model(model_name: str = "3D_demo") -> StarDist3D:
    model_path = MODELS_DIR / model_name

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model folder not found:\n{model_path}\n"
            f"Put your StarDist model folder inside the models/ directory."
        )

    print(f"Loading model from: {model_path}")
    model = StarDist3D(None, name=model_name, basedir=str(MODELS_DIR))
    print("Model loaded successfully.")
    return model


def run_inference(
    model: StarDist3D,
    volume: np.ndarray,
    prob_thresh: float = None,
    nms_thresh: float = None,
) -> np.ndarray:
    """
    Normalize and run StarDist3D on a 3D volume (z, y, x).

    Automatically uses tiling for large volumes and direct inference
    for small ROIs.
    """
    print(f"  Volume shape: {volume.shape}")
    print("  Normalizing...")
    img = normalize(volume.astype(np.float32))

    kwargs = {}
    if prob_thresh is not None:
        kwargs["prob_thresh"] = prob_thresh
    if nms_thresh is not None:
        kwargs["nms_thresh"] = nms_thresh

    needs_tiling = all(
        volume.shape[i] >= BLOCK_SIZE[i]
        for i in range(3)
    )

    if needs_tiling:
        print("  Large volume → tiled inference (predict_instances_big)...")
        labels, _ = model.predict_instances_big(
            img,
            axes="ZYX",
            block_size=BLOCK_SIZE,
            min_overlap=(4, 16, 16),
            **kwargs,
        )
    else:
        print("  Small volume / ROI → direct inference (predict_instances)...")
        labels, _ = model.predict_instances(img, **kwargs)

    print(f"  Detected {int(labels.max())} objects.")
    return labels