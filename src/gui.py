import numpy as np
from magicgui import magicgui
import napari


def open_viewer(images, labels):

    names = list(images.keys())

    image_stack = np.stack(
        [images[name] for name in names],
        axis=0,
    )

    label_stack = np.stack(
        [labels[name] for name in names],
        axis=0,
    )

    viewer = napari.Viewer(title="Neuron Segmentation")

    viewer.add_image(
        image_stack,
        name="Images",
    )

    viewer.add_labels(
        label_stack,
        name="Labels",
    )

    print("\nVolume index:")

    for i, name in enumerate(names):
        print(f"{i} -> {name}")

    return viewer


def create_stardist_widget(model, viewer):
    @magicgui(call_button="Run StarDist (Current Frame)")
    def stardist_widget(image_layer: napari.layers.Image):
        # 1. Get the current slider position (the frame index)
        # viewer.dims.current_step[0] gives the current index of the N dimension
        frame_idx = viewer.dims.current_step[0]

        # 2. Extract only the 3D volume for that frame
        # image_layer.data is (N, Z, Y, X), so we grab image_layer.data[frame_idx]
        current_volume = image_layer.data[frame_idx]

        print(f"Running inference on frame {frame_idx}...")

        # 3. Run inference using your existing stardist_inference logic
        # Make sure this function matches the one you imported
        from src.stardist_inference import run_inference
        labels = run_inference(model, current_volume)

        # 4. Add the labels to the viewer
        # We name it based on the frame index so you know which is which
        viewer.add_labels(labels, name=f"Labels_Frame_{frame_idx}")

    return stardist_widget