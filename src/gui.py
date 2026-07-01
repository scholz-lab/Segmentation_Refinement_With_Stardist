import numpy as np
from magicgui import magicgui
from stardist_inference import run_inference
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
    @magicgui(call_button="Run StarDist (New Layer)")
    def stardist_widget(image_layer: napari.layers.Image):
        # 1. Get the current slider position
        frame_idx = viewer.dims.current_step[0]

        # 2. Extract only the 3D volume for the current frame
        current_volume = image_layer.data[frame_idx]

        print(f"Running inference on frame {frame_idx}...")

        # 3. Run inference
        labels = run_inference(model, current_volume)

        # 4. Create a new layer with a unique name
        # We include the frame index in the name so you know what it is
        new_layer_name = f"Labels_Frame_{frame_idx}"

        # Add the new labels as a fresh layer
        viewer.add_labels(labels, name=new_layer_name)

        print(f"Created new layer: {new_layer_name}")

    return stardist_widget