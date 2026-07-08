"""
OrthoViewer Package - Synchronized multi-planar viewer for 3D volumes.

This package provides an easy-to-use interface for visualizing 3D image volumes
with synchronized XY, YZ, XZ, and 3D viewers.

Example
-------
    from orthoviewer import OrthoViewer
    
    ortho = OrthoViewer(stack, names=names, model=model)
    v3d, v_xy, v_yz, v_xz = ortho.get_viewers()
"""

from .viewer import OrthoViewer

__version__ = "0.1.0"
__all__ = ["OrthoViewer"]
