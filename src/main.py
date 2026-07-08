"""
main.py
-------
Entry point. Run with:
    python src/main.py
"""
from pathlib import Path

import numpy as np
import napari
from tifffile import imread

from stardist_inference import load_model
from gui import open_viewer


def read_data_folder() -> Path:
    txt_file = Path(__file__).parent.parent / "data_path.txt"
    if not txt_file.exists():
        raise FileNotFoundError(
            f"data_path.txt not found at: {txt_file}\n"
            "Create it and put your data folder path inside."
        )
    with open(txt_file) as f:
        folder = Path(f.read().strip().strip('"').strip("'"))
    if not folder.exists():
        raise FileNotFoundError(
            f"Data folder not found: {folder}\n"
            "Check that data_path.txt contains the correct folder path."
        )
    return folder


def load_tiff_stack(data_folder: Path):
    tiff_files = sorted(
        list(data_folder.glob("*.tif")) +
        list(data_folder.glob("*.tiff"))
    )
    if len(tiff_files) == 0:
        raise FileNotFoundError(
            f"No .tif or .tiff files found in: {data_folder}"
        )

    print(f"\nFound {len(tiff_files)} TIFF file(s):")
    for f in tiff_files:
        print(f"  {f.name}")

    volumes, names = [], []
    for tiff_path in tiff_files:
        print(f"Loading: {tiff_path.name} ...", end=" ", flush=True)
        vol = imread(str(tiff_path))
        print(f"shape {vol.shape}")
        volumes.append(vol)
        names.append(tiff_path.stem)

    shapes = [v.shape for v in volumes]
    if len(set(shapes)) > 1:
        raise ValueError(
            "Cannot stack — volumes have different shapes:\n" +
            "\n".join(f"  {n}: {s}" for n, s in zip(names, shapes))
        )

    stack = np.stack(volumes, axis=0)   # (N, z, y, x)
    print(f"\nStack shape: {stack.shape}")
    print("Volume index → filename:")
    for i, name in enumerate(names):
        print(f"  [{i}]  {name}")

    return stack, names


def main():
    print("=" * 50)
    print("  Neuron Segmentation")
    print("=" * 50)

    data_folder = read_data_folder()
    print(f"\nData folder: {data_folder}")

    stack, names = load_tiff_stack(data_folder)

    print("\nLoading StarDist model...")
    model = load_model("3D_demo")

    print("\nLaunching napari...")
    viewer = open_viewer(stack, names, model)

    napari.run()


if __name__ == "__main__":
    main()