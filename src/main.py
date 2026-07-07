"""
main.py
-------
Entry point.

Run with:
    python src/main.py

Flow:
    1. Read data folder from data_path.txt
    2. Find and load all .tif / .tiff files
    3. Stack into (N, z, y, x)
    4. Load StarDist model
    5. Open napari
"""
from pathlib import Path

import numpy as np
import napari
from tifffile import imread

from stardist_inference import load_model
from gui import open_viewer


def read_data_folder() -> Path:
    """Read the data folder path from data_path.txt at the project root."""
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
    """
    Find all .tif / .tiff files in data_folder,
    load each as a 3D volume, and stack into (N, z, y, x).

    Returns
    -------
    stack : np.ndarray  shape (N, z, y, x)
    names : list of str  filenames without extension
    """
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

    volumes = []
    names   = []

    for tiff_path in tiff_files:
        print(f"\nLoading: {tiff_path.name} ...", end=" ", flush=True)
        vol = imread(str(tiff_path))
        print(f"shape {vol.shape}")
        volumes.append(vol)
        names.append(tiff_path.stem)

    # Check all volumes have the same shape before stacking
    shapes = [v.shape for v in volumes]
    if len(set(shapes)) > 1:
        raise ValueError(
            "Cannot stack — volumes have different shapes:\n" +
            "\n".join(f"  {n}: {s}" for n, s in zip(names, shapes))
        )

    stack = np.stack(volumes, axis=0)   # (N, z, y, x)

    print(f"\nStack shape: {stack.shape}")
    print("  axis 0 = volume index (top slider in napari)")
    print("  axis 1 = z slice")
    print("  axis 2 = y")
    print("  axis 3 = x")

    print("\nVolume index → filename:")
    for i, name in enumerate(names):
        print(f"  [{i}]  {name}")

    return stack, names


def main():
    print("=" * 50)
    print("  Neuron Segmentation")
    print("=" * 50)

    # 1. Read data folder
    data_folder = read_data_folder()
    print(f"\nData folder: {data_folder}")

    # 2. Load and stack all tiff files
    stack, names = load_tiff_stack(data_folder)

    # 3. Load StarDist model
    print("\nLoading StarDist model...")
    model = load_model("3D_demo")

    # 4. Open napari
    print("\nLaunching napari...")
    viewer = open_viewer(stack, names, model)

    napari.run()


if __name__ == "__main__":
    main()