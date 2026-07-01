import napari
import numpy as np
from data_loader import load_data
from gui import create_stardist_widget
from stardist_inference import load_model

def main():
    # 1. Load all images
    print("Loading images from folder...")
    images_dict = load_data()
    volumes = list(images_dict.values())

    # 2. Stack images into (N, z, y, x)
    stack = np.stack(volumes, axis=0)
    print(f"\nStack shape: {stack.shape}")

    # 3. Load StarDist model (Inference is NOT run here)
    print("Loading StarDist model...")
    model = load_model('3D_demo')
    print("Model loaded successfully!")

    # 4. Open napari
    viewer = napari.Viewer()

    # Add Image stack
    viewer.add_image(stack, name="Image_Stack")
    viewer.dims.ndisplay = 3

    # Create the widget and add it to the right side
    widget = create_stardist_widget(model, stack)
    viewer.window.add_dock_widget(widget, area='right')

    print("\nNapari launched.")
    # Only call run if we aren't in an interactive environment
    import sys
    if 'ipykernel' not in sys.modules:
        napari.run()

if __name__ == "__main__":
    main()