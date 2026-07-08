"""
OrthoViewer - 4 synchronized napari viewers arranged in 2x2 grid.

Creates 4 separate napari windows for XY, YZ, XZ, and 3D projections arranged in 2x2 grid:
- Top-Left: XY View
- Top-Right: YZ View  
- Bottom-Left: XZ View
- Bottom-Right: 3D Volume View
"""

import numpy as np
import napari
from qtpy import QtCore
import time as _time

try:
    from screeninfo import get_monitors
except ImportError:
    get_monitors = None


class OrthoViewer:
    """Manages 4 synchronized orthogonal viewers arranged in 2x2 grid."""

    def __init__(self, stack: np.ndarray, names: list = None, model=None):
        """
        Initialize OrthoViewer with 4 synchronized viewers in 2x2 grid arrangement.

        Parameters
        ----------
        stack : np.ndarray
            4D image stack, shape (T, Z, Y, X) or (Z, Y, X)
        names : list, optional
            Names for each time frame
        model : object, optional
            StarDist model or other inference model
        """
        self.stack = stack
        self.names = names or []
        self.model = model

        # Handle 3D or 4D stack
        if stack.ndim == 3:
            self.stack = stack[np.newaxis, ...]  # Add time dimension
        
        self.t_idx = 0
        self.z_idx = self.stack.shape[1] // 2
        self.y_idx = self.stack.shape[2] // 2
        self.x_idx = self.stack.shape[3] // 2
        
        # Flag to prevent recursion during sync updates (like in reference code)
        self._updating_display = False

        # Create 4 separate napari viewers (shown=True by default)
        self.v3d = napari.Viewer(title="OrthoViewer | 3D View")
        self.v_xy = napari.Viewer(title="OrthoViewer | XY View (top)")
        self.v_yz = napari.Viewer(title="OrthoViewer | YZ View (side)")
        self.v_xz = napari.Viewer(title="OrthoViewer | XZ View (front)")

        self._add_image_layers()
        self._setup_viewer_configs()
        self._connect_viewers()
        
        # Process Qt events to ensure viewers are ready
        QtCore.QCoreApplication.processEvents()
        
        # Arrange windows in 2x2 grid
        self._arrange_windows()

    def update_global_points(self, points):
        """Updates the 'target' layer in all viewers to match the points."""
        for v in [self.v3d, self.v_xy, self.v_yz, self.v_xz]:
            if 'target' in v.layers:
                # Update the data
                v.layers['target'].data = points

                # Force the camera to center on the new point
                # We map the point [z, y, x] to the viewer's camera center
                # In 2D viewers, we use the relevant two coordinates
                v.camera.center = points[0][-2:]

    def _add_image_layers(self):
        """Add image data and target layer to all viewers."""
        for v in [self.v3d, self.v_xy, self.v_yz, self.v_xz]:
            v.add_image(self.stack, name="volume", colormap="gray")

            # Use the helper to add the point layer
            self._create_target_layer(v)

        self.v3d.dims.ndisplay = 3
        self._add_slicing_planes()

    def _add_slicing_planes(self):
        """Add visualization for slicing planes (crosshairs)."""
        vol_shape = self.stack[self.t_idx].shape
        
        red_plane = np.zeros(vol_shape, dtype=np.uint8)
        blue_plane = np.zeros(vol_shape, dtype=np.uint8)
        green_plane = np.zeros(vol_shape, dtype=np.uint8)

        # Add to 3D viewer
        self.v3d.add_image(red_plane, name="slice_z", colormap="red", opacity=0.5, blending="translucent", visible=False)
        self.v3d.add_image(blue_plane, name="slice_y", colormap="blue", opacity=0.5, blending="translucent", visible=False)
        self.v3d.add_image(green_plane, name="slice_x", colormap="green", opacity=0.5, blending="translucent", visible=False)

        # Add to 2D viewers
        self.v_xy.add_image(red_plane, name="slice_z", colormap="red", opacity=0.3, blending="additive", visible=False)
        self.v_yz.add_image(green_plane, name="slice_x", colormap="green", opacity=0.3, blending="additive", visible=False)
        self.v_xz.add_image(blue_plane, name="slice_y", colormap="blue", opacity=0.3, blending="additive", visible=False)


    def _setup_viewer_configs(self):
        """Configure viewer display settings."""
        self.v_xy.dims.ndisplay = 2
        self.v_yz.dims.ndisplay = 2
        self.v_xz.dims.ndisplay = 2

        # Set axis orders for correct projections
        self.v_yz.dims.order = (0, 3, 2, 1)  # (t, x, z, y)
        self.v_xz.dims.order = (0, 2, 1, 3)  # (t, z, y, x)

        # Initial cursor positions
        for viewer in [self.v3d, self.v_xy, self.v_yz, self.v_xz]:
            viewer.dims.current_step = (self.t_idx, self.z_idx, self.y_idx, self.x_idx)

    def _create_target_layer(self, viewer):
        """Creates a standardized, interactive target layer."""
        # 1. Create the layer with only basic arguments
        target_layer = viewer.add_points(
            np.array([[self.z_idx, self.y_idx, self.x_idx]]),
            name='target',
            size=15,
            symbol='cross'
        )

        # 2. Set properties after creation to avoid keyword argument errors
        target_layer.face_color = 'transparent'
        target_layer.edge_color = 'yellow'

        # Enable interactivity
        target_layer.interactive = True

        # Optional: Safely attempt to set width
        try:
            target_layer.edge_width = 2
        except AttributeError:
            pass  # Older versions might not support edge_width

        return target_layer


    def _force_sync_all(self, pos):
        """The single 'Master' method to update every viewer."""
        if self._updating_display: return
        self._updating_display = True
        try:
            # We must use .data = ... sparingly
            # Only update if the data is actually different
            for v in [self.v3d, self.v_xy, self.v_yz, self.v_xz]:
                if not np.allclose(v.layers['target'].data[0], pos):
                    v.layers['target'].data = np.array([pos])

                # Sync Slices and Camera
                v.dims.current_step = (self.t_idx, int(pos[0]), int(pos[1]), int(pos[2]))
                if v == self.v3d:
                    v.camera.center = pos
                else:
                    v.camera.center = pos[-2:]
        finally:
            self._updating_display = False

    def _connect_viewers(self):
        """Connect viewer step changes and synchronization callbacks."""
        # 1. Sync Slices (Dimension changes)
        self.v3d.dims.events.current_step.connect(self._update_from_v3d)
        self.v_xy.dims.events.current_step.connect(self._update_from_xy)
        self.v_yz.dims.events.current_step.connect(self._update_from_yz)
        self.v_xz.dims.events.current_step.connect(self._update_from_xz)

        # 2. Sync Point Movement and Camera
        for v in [self.v3d, self.v_xy, self.v_yz, self.v_xz]:


            # Use mouse drag to trigger the manual sync
            v.layers['target'].events.data.connect(self._on_target_moved)

    def _on_target_moved(self, event):
        """Triggered when point is dragged."""
        if self._updating_display: return

        # Grab position
        new_pos = event.source.data[0]  # [z, y, x]

        self._updating_display = True
        try:
            for v in [self.v3d, self.v_xy, self.v_yz, self.v_xz]:
                # 1. Update point if different
                if not np.allclose(v.layers['target'].data[0], new_pos):
                    v.layers['target'].data = np.array([new_pos])

                # 2. Update Slices (without re-triggering the Point Update!)
                # We use a direct set instead of calling _update_all
                v.dims.current_step = (self.t_idx, int(new_pos[0]), int(new_pos[1]), int(new_pos[2]))

                # 3. Camera Sync
                v.camera.center = new_pos if v == self.v3d else new_pos[-2:]
        finally:
            self._updating_display = False

    
    def _update_from_v3d(self, event):
        if not self._updating_display:
            self._update_all(self.v3d)
    
    def _update_from_xy(self, event):
        if not self._updating_display:
            self._update_all(self.v_xy)
    
    def _update_from_yz(self, event):
        if not self._updating_display:
            self._update_all(self.v_yz)
    
    def _update_from_xz(self, event):
        if not self._updating_display:
            self._update_all(self.v_xz)


    def _sync_camera(self, event):
        """Synchronizes camera center and zoom across all views."""
        if self._updating_display:
            return

        self._updating_display = True
        try:
            source = event.source
            for v in [self.v3d, self.v_xy, self.v_yz, self.v_xz]:
                if v.camera != source:
                    # Sync the view parameters
                    v.camera.center = source.center
                    v.camera.zoom = source.zoom

        finally:
            self._updating_display = False

    def update_target_crosshair(self, z, y, x):
        """Update the crosshair position in all viewers."""
        point = np.array([[z, y, x]])
        for v in [self.v3d, self.v_xy, self.v_yz, self.v_xz]:
            if 'target' in v.layers:
                v.layers['target'].data = point

    def _update_all(self, source_viewer):
        """Sync ONLY sliders and slices, NOT the point."""
        if self._updating_display: return

        self._updating_display = True
        try:
            step = source_viewer.dims.current_step
            self.t_idx, self.z_idx, self.y_idx, self.x_idx = step[-4:]

            # Sync other viewers' sliders
            new_step = (self.t_idx, self.z_idx, self.y_idx, self.x_idx)
            for v in [self.v3d, self.v_xy, self.v_yz, self.v_xz]:
                if v != source_viewer:
                    v.dims.current_step = new_step

            # Only update slice visuals
            self._update_slicing()
            # REMOVED: self.update_target_crosshair(...)
        finally:
            self._updating_display = False


    def _update_slicing(self):
        """Update slicing plane visualizations based on current position."""
        vol_shape = self.stack[self.t_idx].shape

        z_plane = np.zeros(vol_shape, dtype=np.uint8)
        y_plane = np.zeros(vol_shape, dtype=np.uint8)
        x_plane = np.zeros(vol_shape, dtype=np.uint8)

        z_plane[self.z_idx, :, :] = 255
        y_plane[:, self.y_idx, :] = 255
        x_plane[:, :, self.x_idx] = 255

        # Update 3D viewer slices
        self.v3d.layers["slice_z"].data = z_plane
        self.v3d.layers["slice_y"].data = y_plane
        self.v3d.layers["slice_x"].data = x_plane

        # Update 2D viewer slices
        self.v_xy.layers["slice_z"].data = z_plane
        self.v_yz.layers["slice_x"].data = x_plane
        self.v_xz.layers["slice_y"].data = y_plane

    def _on_double_click(self, viewer, event):
        """On double-click, sync all viewers to that position."""
        try:
            # Extract coordinates from click
            layer = viewer.layers[0]
            coords = layer.world_to_data(event.position)
            coords = np.round(coords).astype(int)
            
            # Set viewer's step (this will trigger _update_all via the event)
            viewer.dims.current_step = tuple(np.clip(coords, 0, 
                [self.stack.shape[i]-1 for i in range(len(coords))]))
        except Exception as e:
            print(f"Double-click error: {e}")

    def _arrange_windows(self):
        """Arrange 4 viewer windows in 2x2 grid on screen."""
        try:
            if get_monitors is None:
                print("Warning: screeninfo not installed. Windows will not be auto-arranged.")
                print("Install with: pip install screeninfo")
                return

            monitors = get_monitors()
            if not monitors:
                print("Warning: No monitors detected.")
                return

            monitor = max(monitors, key=lambda m: m.width * m.height)
            
            width_monitor = monitor.width
            height_monitor = monitor.height
            
            print(f"Screen resolution: {width_monitor}x{height_monitor}")
            
            # Reduced margins for better fit
            margin_x = 2
            margin_y = 40  # taskbar space
            spacing = 0    # no gap between windows for tight fit
            
            # Calculate window size for 2x2 grid
            window_width = (width_monitor - 2 * margin_x) // 2
            window_height = (height_monitor - margin_y - margin_x) // 2
            
            # Ensure minimum window size
            window_width = max(window_width, 300)
            window_height = max(window_height, 250)
            
            print(f"Window size: {window_width}x{window_height}")

            # Define 2x2 grid positions: [row, col]
            # [0, 0]: XY (top-left)
            # [0, 1]: YZ (top-right)
            # [1, 0]: XZ (bottom-left)
            # [1, 1]: 3D (bottom-right)
            
            positions = [
                (monitor.x + margin_x, monitor.y + margin_y),                              # XY: top-left
                (monitor.x + margin_x + window_width + spacing, monitor.y + margin_y),       # YZ: top-right
                (monitor.x + margin_x, monitor.y + margin_y + window_height + spacing),       # XZ: bottom-left
                (monitor.x + margin_x + window_width + spacing, monitor.y + margin_y + window_height + spacing),  # 3D: bottom-right
            ]
            
            viewers = [self.v_xy, self.v_yz, self.v_xz, self.v3d]
            titles = ["XY (top)", "YZ (side)", "XZ (front)", "3D View"]
            
            print("\nArranging windows...")
            for viewer, (x, y), title in zip(viewers, positions, titles):
                # Ensure Qt event loop is running
                QtCore.QCoreApplication.processEvents()
                _time.sleep(0.1)
                
                w = viewer.window._qt_window
                
                # Move and resize window
                w.setGeometry(x, y, window_width, window_height)
                
                # Ensure window is shown
                w.show()
                w.raise_()
                w.activateWindow()
                
                # Process events again
                QtCore.QCoreApplication.processEvents()
                _time.sleep(0.05)
                
                print(f"  ✓ {title}: ({x}, {y}) {window_width}x{window_height}")

        except Exception as e:
            print(f"Warning: Could not arrange windows: {e}")
            import traceback
            traceback.print_exc()

    def get_viewers(self):
        """Return tuple of (v3d, v_xy, v_yz, v_xz) viewers."""
        return self.v3d, self.v_xy, self.v_yz, self.v_xz

    def get_main_viewer(self):
        """Return the 3D viewer (for docking widgets)."""
        return self.v3d

    def get_current_position(self):
        """Return current cursor position as (z, y, x)."""
        return np.array([self.z_idx, self.y_idx, self.x_idx])

    def set_current_position(self, z, y, x):
        """Set cursor position in all viewers."""
        self.z_idx = np.clip(z, 0, self.stack.shape[1] - 1)
        self.y_idx = np.clip(y, 0, self.stack.shape[2] - 1)
        self.x_idx = np.clip(x, 0, self.stack.shape[3] - 1)

        for viewer in [self.v3d, self.v_xy, self.v_yz, self.v_xz]:
            viewer.dims.current_step = (self.t_idx, self.z_idx, self.y_idx, self.x_idx)

    def toggle_slicing(self, show: bool = True):
        """Toggle visibility of slicing plane visualizations."""
        for viewer in [self.v3d, self.v_xy, self.v_yz, self.v_xz]:
            for layer_name in ["slice_z", "slice_y", "slice_x"]:
                if layer_name in viewer.layers:
                    viewer.layers[layer_name].visible = show

    def add_layer(self, data: np.ndarray, name: str, layer_type: str = "image", **kwargs):
        """Add a layer to all viewers."""
        for viewer in [self.v3d, self.v_xy, self.v_yz, self.v_xz]:
            if layer_type == "image":
                viewer.add_image(data, name=name, **kwargs)
            elif layer_type == "labels":
                viewer.add_labels(data, name=name, **kwargs)
