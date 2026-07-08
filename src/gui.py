import numpy as np
import napari
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QDoubleSpinBox, QHBoxLayout, QGroupBox
from stardist_inference import run_inference
from roi import ROIManager
from orthoviewer import OrthoViewer


def open_viewer(stack: np.ndarray, names: list, model) -> napari.Viewer:
    """
    Open orthogonal viewer with 4 synchronized napari windows in 2x2 grid.

    Parameters
    ----------
    stack : np.ndarray
        Image stack, shape (T, Z, Y, X) or (Z, Y, X)
    names : list
        Volume names/labels
    model
        StarDist model for inference

    Returns
    -------
    napari.Viewer
        3D viewer instance (control panel will be docked here)
    """
    # Initialize OrthoViewer with 4 synchronized separate windows
    ortho = OrthoViewer(stack, names=names, model=model)
    v3d, v_xy, v_yz, v_xz = ortho.get_viewers()

    # Add labels layer to 3D viewer for segmentation results
    v3d.add_labels(np.zeros(stack.shape, dtype=np.int32), name="Segmentation")

    # Build control panel and dock to 3D viewer
    roi_manager = ROIManager()
    panel = _build_panel(ortho, stack, names, model, roi_manager)
    v3d.window.add_dock_widget(panel, area="right", name="StarDist Controls")

    return v3d


def _build_panel(ortho: OrthoViewer, stack, names, model, roi_manager):
    """Build control panel with StarDist and viewer options."""
    v3d, v_xy, v_yz, v_xz = ortho.get_viewers()

    panel = QWidget()
    layout = QVBoxLayout()
    panel.setLayout(layout)

    # Title
    title = QLabel("🔬 StarDist Controls")
    layout.addWidget(title)

    # Info label
    info_label = QLabel("Double-click to center\non any viewer")
    info_label.setStyleSheet("color: #666; font-size: 10px;")
    layout.addWidget(info_label)

    # Slicing visualization toggle
    slicing_btn = QPushButton("Toggle Slicing Planes")
    slicing_btn.setCheckable(True)
    slicing_btn.setChecked(False)

    def toggle_slicing(checked):
        ortho.toggle_slicing(checked)

    slicing_btn.clicked.connect(toggle_slicing)
    layout.addWidget(slicing_btn)

    layout.addSpacing(10)
    layout.addWidget(QLabel("Segmentation:"))

    # ROI button
    roi_btn = QPushButton("Run StarDist on ROI (Current zoom)")
    layout.addWidget(roi_btn)

    def run_roi():
        """Run segmentation on current view region."""
        try:
            vol_idx = ortho.t_idx
            volume = stack[vol_idx]
            z, y, x = ortho.get_current_position()
            
            # Define ROI around current center (200x200x200 pixels)
            roi_size = 100
            z0 = max(0, z - roi_size)
            z1 = min(volume.shape[0], z + roi_size)
            y0 = max(0, y - roi_size)
            y1 = min(volume.shape[1], y + roi_size)
            x0 = max(0, x - roi_size)
            x1 = min(volume.shape[2], x + roi_size)
            
            roi = volume[z0:z1, y0:y1, x0:x1]
            result = run_inference(model, roi)

            v3d.add_labels(
                result,
                name=f"StarDist_ROI_{vol_idx}",
                translate=[z0, y0, x0],
                opacity=0.5
            )
            print(f"✓ Segmented ROI: Z[{z0}:{z1}], Y[{y0}:{y1}], X[{x0}:{x1}]")
        except Exception as e:
            print(f"Error during segmentation: {e}")

    roi_btn.clicked.connect(run_roi)

    # Full volume segmentation
    full_btn = QPushButton("Run StarDist on Full Volume")
    layout.addWidget(full_btn)

    def run_full():
        """Run segmentation on entire volume."""
        try:
            vol_idx = ortho.t_idx
            volume = stack[vol_idx]
            result = run_inference(model, volume)
            v3d.add_labels(result, name=f"StarDist_Full_{vol_idx}", opacity=0.5)
            print(f"✓ Segmented full volume at frame {vol_idx}")
        except Exception as e:
            print(f"Error during segmentation: {e}")

    full_btn.clicked.connect(run_full)

    # Position display
    layout.addSpacing(10)
    layout.addWidget(QLabel("Position Info:"))
    pos_label = QLabel("Position: (Z, Y, X) = (-, -, -)")
    pos_label.setStyleSheet("font-family: monospace; font-size: 11px;")
    layout.addWidget(pos_label)

    def update_pos(*args):
        z, y, x = ortho.get_current_position()
        frame = ortho.t_idx
        pos_label.setText(f"Frame: {frame} | Z={z:3d}, Y={y:3d}, X={x:3d}")

    for viewer in ortho.get_viewers():
        viewer.dims.events.current_step.connect(update_pos)

    # Initial position update
    update_pos()

    layout.addStretch()
    return panel