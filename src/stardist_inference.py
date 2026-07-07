"""
stardist_inference.py
---------------------
Handles everything related to StarDist:
  - loading the model (downloads to local models/ folder if not there yet)
  - normalizing the volume
  - running inference and returning labels

Nothing here knows about napari or any GUI.
"""
from pathlib import Path

import numpy as np
from csbdeep.utils import normalize
from stardist.models import StarDist3D

# models/ folder lives at the project root (one level above src/)
MODELS_DIR = Path(__file__).parent.parent / "models"

# predict_instances_big requires each dimension >= block_size
# if the volume is smaller (e.g. a small ROI), use predict_instances directly
BLOCK_SIZE = (64, 128, 128)


def load_model(model_name="3D_demo"):
    """
    Load a StarDist model from the local models folder.
    """
    model_path = MODELS_DIR / model_name

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model folder not found:\n{model_path}"
        )

    print(f"Loading model from:\n{model_path}")

    model = StarDist3D(
        None,
        name=model_name,
        basedir=str(MODELS_DIR)
    )

    print("Model loaded successfully!")
    return model


def run_inference(
    model,
    volume,
    prob_thresh=None,
    nms_thresh=None,
):
    print(f"Volume shape: {volume.shape}")
    print("Normalizing image...")
    img = normalize(volume.astype(np.float32))

    kwargs = {}
    if prob_thresh is not None:
        kwargs["prob_thresh"] = prob_thresh
    if nms_thresh is not None:
        kwargs["nms_thresh"] = nms_thresh

    # Check if volume is large enough for tiled inference
    # All 3 dims must be >= block_size, otherwise predict_instances_big crashes
    needs_tiling = all(
        volume.shape[i] >= BLOCK_SIZE[i]
        for i in range(3)
    )

    if needs_tiling:
        print("Large volume → tiled inference (predict_instances_big)...")
        labels, _ = model.predict_instances_big(
            img,
            'ZYX',
            block_size=BLOCK_SIZE,
            min_overlap=(4, 16, 16),
            **kwargs,
        )
    else:
        print("Small volume / ROI → direct inference (predict_instances)...")
        labels, _ = model.predict_instances(img, **kwargs)

    print(f"Detected {int(labels.max())} objects")
    return labels