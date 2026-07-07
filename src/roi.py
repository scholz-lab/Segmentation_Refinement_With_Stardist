"""
roi.py
------
Handles everything related to ROI selection:
  - Converting a napari Shapes rectangle into a TRUE 3D bounding box
  - Cropping a 3D volume to that bounding box
  - Storing the bbox so the merge step knows where to paste back

When you draw a rectangle in napari's 3D view, each vertex already has
a z coordinate. We use the actual min/max z from those vertices directly
— no center+margin guessing needed.

The margin is still available for y and x padding if you want extra context
around the drawn area.
"""
import numpy as np


class ROIManager:
    def __init__(self, margin: int = 0):
        """
        Parameters
        ----------
        margin : int
            Extra voxels added around the ROI in ALL directions (z, y, x).
            Useful because StarDist performs better with a little context
            around the neurons.
        """
        self.margin = margin
        self.last_bbox = None   # saved after each crop so merge can use it

    def shapes_to_bbox(
        self,
        shape_data: np.ndarray,
        volume_shape: tuple,
    ) -> dict:
        """
        Convert a single napari Shapes rectangle drawn in 3D view into a
        true 3D bounding box.

        napari vertex layout for a 4D stack (N, z, y, x):
            shape_data has shape (4, 4)
            columns: [frame, z, y, x]

        napari vertex layout for a 3D volume (z, y, x):
            shape_data has shape (4, 3)
            columns: [z, y, x]

        The min/max of z, y, x are taken directly from the drawn vertices.
        The margin is added around all sides and clipped to the volume bounds.

        Parameters
        ----------
        shape_data   : (4, ndim) array of rectangle vertices from napari
        volume_shape : shape of ONE 3D volume (z, y, x)

        Returns
        -------
        bbox : dict with keys z0, z1, y0, y1, x0, x1, slices
        """
        coords = np.array(shape_data)
        nz, ny, nx = volume_shape

        if coords.shape[1] == 4:
            # 4D stack — columns are (frame, z, y, x)
            z_coords = coords[:, 1]
            y_coords = coords[:, 2]
            x_coords = coords[:, 3]
        else:
            # 3D volume — columns are (z, y, x)
            z_coords = coords[:, 0]
            y_coords = coords[:, 1]
            x_coords = coords[:, 2]

        # Use actual min/max from the drawn rectangle — true 3D bbox
        z0 = int(np.floor(z_coords.min())) - self.margin
        z1 = int(np.ceil( z_coords.max())) + self.margin + 1

        y0 = int(np.floor(y_coords.min())) - self.margin
        y1 = int(np.ceil( y_coords.max())) + self.margin + 1

        x0 = int(np.floor(x_coords.min())) - self.margin
        x1 = int(np.ceil( x_coords.max())) + self.margin + 1

        # Clip to volume bounds so we never go out of range
        z0 = max(0,  z0);  z1 = min(nz, z1)
        y0 = max(0,  y0);  y1 = min(ny, y1)
        x0 = max(0,  x0);  x1 = min(nx, x1)

        bbox = dict(
            z0=z0, z1=z1,
            y0=y0, y1=y1,
            x0=x0, x1=x1,
            slices=(slice(z0, z1), slice(y0, y1), slice(x0, x1)),
        )

        self.last_bbox = bbox   # save for merge step later

        print(f"  3D bbox: z[{z0}:{z1}] y[{y0}:{y1}] x[{x0}:{x1}]")
        return bbox

    def crop_volume(
        self,
        volume: np.ndarray,
        bbox: dict,
    ) -> np.ndarray:
        """
        Crop a 3D volume (z, y, x) using the bounding box.

        Parameters
        ----------
        volume : ndarray shape (z, y, x)
        bbox   : dict returned by shapes_to_bbox

        Returns
        -------
        cropped : ndarray shape (z1-z0, y1-y0, x1-x0)
        """
        s = bbox["slices"]
        crop = volume[s]
        print(f"  Cropped shape: {crop.shape}")
        return crop