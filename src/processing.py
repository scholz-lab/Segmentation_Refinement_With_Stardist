import numpy as np
from stardist.models import StarDist3D
from csbdeep.utils import normalize
import os


def load_model(model_name='3D_demo'):
    # Pointing to the local folder where you extracted the zip
    model_dir = "C:/Users/pyaasa/Documents/Git_Project/Segmentation_Refinement_With_Stardist/stardist_model"
    model_path = os.path.abspath(os.path.join(model_dir, model_name))
    return StarDist3D(None, name=model_name, basedir=model_dir)


def run_stardist(model, image, prob_thresh=0.5, nms_thresh=0.3):
    # StarDist strictly requires normalization
    # Axis=(0, 1, 2) assumes ZYX order.
    # If your image has channels, adjust axis=(0, 1, 2) to match spatial dims.
    img_norm = normalize(image, 1, 99.8, axis=(0, 1, 2))

    # Standard prediction call as per StarDist examples
    labels, details = model.predict_instances(
        img_norm,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh
    )
    return labels


def integrate_labels(global_labels, local_labels):
    # Ensure they are the same shape
    if global_labels.shape != local_labels.shape:
        return global_labels

    mask = local_labels > 0
    # Shift local IDs to avoid overlap with existing global IDs
    offset = global_labels.max()
    global_labels[mask] = local_labels[mask] + offset
    return global_labels