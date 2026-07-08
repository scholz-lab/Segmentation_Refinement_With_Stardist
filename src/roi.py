"""
roi.py
------
Combines two 2D rectangle selections into one 3D bounding box:
  - XY view rectangle → gives Y and X extent
  - YZ view rectangle → gives Z and Y extent (Y used as cross-check)

Nothing here knows about StarDist or the GUI.
"""
import numpy as np


class ROIManager:
    def __init__(self):
        self.last_bbox = None  # saved after each crop for merge step

    def bbox_from_two_views(
        self,
        xy_shapes_layer,    # rectangle drawn in XY view
        yz_shapes_layer,    # rectangle drawn in YZ view
        volume_shape: tuple,  # (z, y, x) of one 3D volume
    ) -> dict:
        """
        Build a 3D bounding box from two 2D rectangle selections.

        XY view rectangle vertices: columns are (y, x)
        YZ view rectangle vertices: columns are (y, z)
            → we read Z from this view

        Parameters
        ----------
        xy_shapes_layer : napari Shapes layer drawn in XY view
        yz_shapes_layer : napari Shapes layer drawn in YZ view
        volume_shape    : (z, y, x)

        Returns
        -------
        bbox : dict with z0,z1,y0,y1,x0,x1 and slices
        """
        if len(xy_shapes_layer.data) == 0:
            raise RuntimeError("No rectangle drawn in the XY view yet.")
        if len(yz_shapes_layer.data) == 0:
            raise RuntimeError("No rectangle drawn in the YZ view yet.")

        nz, ny, nx = volume_shape

        # ── XY view: get Y and X ──────────────────────────────────────────
        # XY viewer is 2D with axes (y, x)
        # vertices shape: (4, 2) → columns: (y, x)
        xy_verts = np.array(xy_shapes_layer.data[-1])
        y0 = max(0,  int(np.floor(xy_verts[:, 0].min())))
        y1 = min(ny, int(np.ceil( xy_verts[:, 0].max())))
        x0 = max(0,  int(np.floor(xy_verts[:, 1].min())))
        x1 = min(nx, int(np.ceil( xy_verts[:, 1].max())))

        # ── YZ view: get Z ────────────────────────────────────────────────
        # YZ viewer is 2D with axes (z, y)
        # vertices shape: (4, 2) → columns: (z, y)
        yz_verts = np.array(yz_shapes_layer.data[-1])
        z0 = max(0,  int(np.floor(yz_verts[:, 0].min())))
        z1 = min(nz, int(np.ceil( yz_verts[:, 0].max())))

        bbox = dict(
            z0=z0, z1=z1,
            y0=y0, y1=y1,
            x0=x0, x1=x1,
            slices=(slice(z0, z1), slice(y0, y1), slice(x0, x1)),
        )
        self.last_bbox = bbox

        print(f"  3D bbox from two views:")
        print(f"    XY view → y[{y0}:{y1}] x[{x0}:{x1}]")
        print(f"    YZ view → z[{z0}:{z1}]")
        print(f"    Full 3D → z[{z0}:{z1}] y[{y0}:{y1}] x[{x0}:{x1}]")
        return bbox

    def crop_volume(self, volume: np.ndarray, bbox: dict) -> np.ndarray:
        """
        Crop a 3D volume (z, y, x) using bbox from bbox_from_two_views.
        """
        crop = volume[bbox["slices"]]
        print(f"  Cropped shape: {crop.shape}")
        return crop