"""
gui.py
------
Napari viewer setup and dock widgets:
  - Full volume StarDist segmentation
  - ROI-only StarDist segmentation
"""
import numpy as np
import napari
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QLabel,
    QPushButton, QDoubleSpinBox,
    QHBoxLayout, QGroupBox,
)
from qtpy.QtCore import Qt

from stardist_inference import run_inference
from roi import ROIManager


def open_viewer(stack: np.ndarray, names: list, model) -> napari.Viewer:
    viewer = napari.Viewer(title="Neuron Segmentation")

    image_layer = viewer.add_image(stack, name="Image_Stack")

    global_labels = np.zeros(stack.shape, dtype=np.int32)
    viewer.add_labels(global_labels, name="Global_Labels")

    viewer.add_shapes(
        name="ROI",
        ndim=stack.ndim,
        edge_color="yellow",
        face_color="transparent",
    )

    viewer.dims.ndisplay = 3
    viewer.window.resize(1600, 950)

    roi_manager = ROIManager(margin=8)

    widget = _build_panel(viewer, image_layer, stack, names, model, roi_manager)
    viewer.window.add_dock_widget(widget, area="right", name="StarDist Controls")

    print("Napari open.")
    print("  Top slider  -> switch volume")
    print("  Drag        -> rotate 3D view")
    print("  ROI layer   -> draw rectangle, then click Run StarDist on ROI")

    return viewer


def _build_panel(viewer, image_layer, stack, names, model, roi_manager):

    panel = QWidget()
    layout = QVBoxLayout()
    layout.setAlignment(Qt.AlignTop)
    panel.setLayout(layout)

    # ── Full volume ───────────────────────────────────────────────────────
    full_box = QGroupBox("Full Volume")
    full_layout = QVBoxLayout()
    full_box.setLayout(full_layout)

    full_status = QLabel("Not run yet.")
    full_status.setWordWrap(True)
    full_layout.addWidget(full_status)

    full_btn = QPushButton("Run StarDist on current volume")
    full_btn.setStyleSheet("padding: 6px;")
    full_layout.addWidget(full_btn)

    layout.addWidget(full_box)

    # ── ROI ───────────────────────────────────────────────────────────────
    roi_box = QGroupBox("ROI Segmentation")
    roi_layout = QVBoxLayout()
    roi_box.setLayout(roi_layout)

    roi_layout.addWidget(QLabel(
        "1. Select ROI layer\n"
        "2. Draw a rectangle around neurons\n"
        "3. Adjust params and click Run"
    ))

    row1 = QHBoxLayout()
    row1.addWidget(QLabel("Z margin:"))
    margin_spin = QDoubleSpinBox()
    margin_spin.setRange(0, 50)
    margin_spin.setValue(8)
    margin_spin.setDecimals(0)
    row1.addWidget(margin_spin)
    roi_layout.addLayout(row1)

    row2 = QHBoxLayout()
    row2.addWidget(QLabel("Prob threshold:"))
    prob_spin = QDoubleSpinBox()
    prob_spin.setRange(0.01, 0.99)
    prob_spin.setSingleStep(0.05)
    prob_spin.setValue(0.5)
    prob_spin.setDecimals(2)
    row2.addWidget(prob_spin)
    roi_layout.addLayout(row2)

    row3 = QHBoxLayout()
    row3.addWidget(QLabel("NMS threshold:"))
    nms_spin = QDoubleSpinBox()
    nms_spin.setRange(0.01, 0.99)
    nms_spin.setSingleStep(0.05)
    nms_spin.setValue(0.3)
    nms_spin.setDecimals(2)
    row3.addWidget(nms_spin)
    roi_layout.addLayout(row3)

    roi_status = QLabel("No ROI run yet.")
    roi_status.setWordWrap(True)
    roi_layout.addWidget(roi_status)

    roi_btn = QPushButton("Run StarDist on ROI only")
    roi_btn.setStyleSheet("padding: 6px;")
    roi_layout.addWidget(roi_btn)

    layout.addWidget(roi_box)

    # ── Callbacks ─────────────────────────────────────────────────────────

    def run_full():
        vol_idx = viewer.dims.current_step[0]
        name = names[vol_idx] if vol_idx < len(names) else str(vol_idx)
        full_status.setText(f"Running on [{vol_idx}] {name} ...")
        full_btn.setEnabled(False)

        volume = stack[vol_idx]
        result = run_inference(model, volume)

        data = viewer.layers["Global_Labels"].data.copy()
        data[vol_idx] = result
        viewer.layers["Global_Labels"].data = data

        full_status.setText(
            f"Done [{vol_idx}] {name}\n"
            f"Found {int(result.max())} objects."
        )
        full_btn.setEnabled(True)

    def run_roi():
        shapes_layer = viewer.layers["ROI"]

        # Check a rectangle has been drawn
        if len(shapes_layer.data) == 0:
            roi_status.setText(
                "No ROI drawn yet.\n"
                "Select the ROI layer and draw a rectangle first."
            )
            return

        try:
            roi_manager.margin = int(margin_spin.value())

            # current frame index
            vol_idx = viewer.dims.current_step[0]

            # single 3D volume for this frame (z, y, x)
            volume = image_layer.data[vol_idx]

            # shapes_to_bbox takes the last drawn shape's vertex array
            # and the 3D volume shape (z, y, x)
            shape_data = shapes_layer.data[-1]   # (4, 4) array: last rectangle
            bbox = roi_manager.shapes_to_bbox(shape_data, volume.shape)

            # crop using the slices stored inside bbox
            roi = roi_manager.crop_volume(volume, bbox)

        except Exception as e:
            roi_status.setText(f"ROI error: {e}")
            return

        roi_status.setText(
            f"ROI: z[{bbox['z0']}:{bbox['z1']}] "
            f"y[{bbox['y0']}:{bbox['y1']}] "
            f"x[{bbox['x0']}:{bbox['x1']}]\n"
            f"Shape: {roi.shape}\n"
            f"Running StarDist..."
        )
        roi_btn.setEnabled(False)

        result = run_inference(
            model,
            roi,
            prob_thresh=prob_spin.value(),
            nms_thresh=nms_spin.value(),
        )

        # unique layer name per run so you can compare multiple parameter sets
        layer_name = (
            f"ROI_Frame{vol_idx}"
            f"_p{prob_spin.value():.2f}"
            f"_nms{nms_spin.value():.2f}"
        )

        # translate places the labels at the correct position in the volume
        # result is 3D (z,y,x) so translate needs 3 values, not 4
        translate = [bbox["z0"], bbox["y0"], bbox["x0"]]
        viewer.add_labels(result, name=layer_name, translate=translate)

        roi_status.setText(
            f"Done. Found {int(result.max())} objects.\n"
            f"Layer: {layer_name}\n"
            "Run again with different params to compare."
        )
        roi_btn.setEnabled(True)

    full_btn.clicked.connect(run_full)
    roi_btn.clicked.connect(run_roi)

    return panel