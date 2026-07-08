# OrthoViewer Package

Synchronized multi-planar viewer for 3D volumetric image analysis.

## Features

- **Synchronized 4-Viewer Display**: XY (top), YZ (side), XZ (front), and 3D volume views
- **Real-time Cursor Sync**: Moving cursor in one viewer updates all others
- **Crosshair Visualization**: Optional slicing planes to show intersection geometry
- **Auto Window Arrangement**: Viewers positioned side-by-side based on screen resolution
- **Easy Integration**: Works seamlessly with napari and existing segmentation workflows

## Quick Start

```python
from orthoviewer import OrthoViewer
import numpy as np

# Load your 3D/4D image stack
stack = np.load("mydata.npy")  # Shape: (Z, Y, X) or (T, Z, Y, X)

# Create orthoviewer
ortho = OrthoViewer(stack, names=["Frame 0", "Frame 1", ...])

# Get individual viewers
v3d, v_xy, v_yz, v_xz = ortho.get_viewers()

# Add custom layers
v3d.add_labels(segmentation_result, name="Segmentation")
```

## API Reference

### OrthoViewer

Main class for managing synchronized viewers.

#### Methods

- `get_viewers()` → (v3d, v_xy, v_yz, v_xz)
  - Returns tuple of all four napari.Viewer instances

- `get_current_position()` → (z, y, x)
  - Returns current cursor position in pixel coordinates

- `set_current_position(z, y, x)`
  - Set cursor position in all viewers

- `toggle_slicing(show: bool)`
  - Toggle visibility of slicing plane overlays

- `add_layer(data, name, layer_type="image", **kwargs)`
  - Add a layer to all viewers simultaneously

## Integration with GUI

The package integrates smoothly with your GUI framework:

```python
from orthoviewer import OrthoViewer
from gui import _build_panel

ortho = OrthoViewer(stack, names=names, model=model)
v3d, _, _, _ = ortho.get_viewers()

panel = _build_panel(ortho, stack, names, model, roi_manager, ...)
v3d.window.add_dock_widget(panel, area="right", name="Controls")
```

## Architecture

```
src/
├── orthoviewer/
│   ├── __init__.py      # Package initialization and exports
│   └── viewer.py        # OrthoViewer class implementation
├── gui.py               # GUI integration
└── main.py              # Entry point
```

## Requirements

- napari
- numpy
- qtpy
- screeninfo (optional - for auto window arrangement)

See `requirements.txt` for specific versions.

**Optional**: If `screeninfo` is not installed, the viewer will still work but windows won't be auto-arranged. Install it with:
```bash
pip install screeninfo
```
