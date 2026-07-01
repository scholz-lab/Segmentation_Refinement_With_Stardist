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
    volume,
    model,
    prob_thresh=None,
    nms_thresh=None,
):

    print("Normalizing image...")

    img = normalize(volume.astype(np.float32))

    kwargs = {}

    if prob_thresh is not None:
        kwargs["prob_thresh"] = prob_thresh

    if nms_thresh is not None:
        kwargs["nms_thresh"] = nms_thresh

    print("Running StarDist...")

    labels, _ = model.predict_instances(
        img,
        **kwargs,
    )

    print(f"Detected {labels.max()} objects")

    return labels