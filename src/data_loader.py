"""
data_loader.py
--------------
Loads all TIFF volumes from the folder specified in data_path.txt.

Returns
-------
images : dict
    Dictionary of the form:
    {
        "volume_name": numpy.ndarray (z, y, x),
        ...
    }
"""

from pathlib import Path
from tifffile import imread


def load_data():
    # Project root
    project_root = Path(__file__).parent.parent

    # Path to the text file containing the data folder
    txt_file = project_root / "data_path.txt"

    if not txt_file.exists():
        raise FileNotFoundError(
            f"Could not find {txt_file}"
        )

    # Read folder path
    with open(txt_file, "r") as f:
        data_folder = Path(f.read().strip().strip('"').strip("'"))

    if not data_folder.exists():
        raise FileNotFoundError(
            f"Data folder does not exist:\n{data_folder}"
        )

    # Find all TIFF files
    tiff_files = sorted(list(data_folder.glob("*.tif")) +
                        list(data_folder.glob("*.tiff")))

    if len(tiff_files) == 0:
        raise FileNotFoundError(
            f"No TIFF files found in:\n{data_folder}"
        )

    print(f"\nFound {len(tiff_files)} TIFF files")

    images = {}
    shapes = []

    for file in tiff_files:
        print(f"Loading {file.name}...")

        volume = imread(file)

        images[file.stem] = volume
        shapes.append(volume.shape)

        print(f"Shape: {volume.shape}")

    # Check all images have the same shape
    if len(set(shapes)) != 1:
        raise ValueError(
            "All volumes must have the same shape.\n"
            + "\n".join(
                f"{name}: {img.shape}"
                for name, img in images.items()
            )
        )

    print("\nFinished loading images.")

    return images