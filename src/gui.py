from magicgui import magic_factory
import napari
from .processing import run_stardist, integrate_labels, load_model

model = load_model()


@magic_factory(call_button="Run Coarse Segmentation",
               prob_thresh={"min": 0.1, "max": 0.9, "step": 0.1},
               nms_thresh={"min": 0.1, "max": 0.9, "step": 0.1})
def stardist_widget(viewer: napari.Viewer,
                    image_layer: napari.layers.Image,
                    prob_thresh: float = 0.5,
                    nms_thresh: float = 0.3):
    def run():
        labels = run_stardist(model, image_layer.data, prob_thresh, nms_thresh)
        viewer.add_labels(labels, name="Global_Labels")

    return run


@magic_factory(call_button="Integrate Local to Global")
def integration_widget(viewer: napari.Viewer,
                       global_layer: napari.layers.Labels,
                       local_layer: napari.layers.Labels):
    def integrate():
        global_layer.data = integrate_labels(global_layer.data, local_layer.data)
        viewer.layers.remove(local_layer)  # Remove temp layer after merge

    return integrate