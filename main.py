import napari
from src.gui import stardist_widget, integration_widget


def main():
    viewer = napari.Viewer()

    # Add our custom tools
    viewer.window.add_dock_widget(stardist_widget(), area='right', name="StarDist Control")
    viewer.window.add_dock_widget(integration_widget(), area='right', name="Integration Tool")

    napari.run()


if __name__ == "__main__":
    main()