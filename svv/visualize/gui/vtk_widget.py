"""
VTK/PyVista widget for 3D domain visualization.
"""
import os
import platform
import sys

# macOS: force layer-backed views to avoid Qt/VTK view initialization deadlocks.
if sys.platform == 'darwin':
    os.environ.setdefault('QT_MAC_WANTS_LAYER', '1')

# Configure software rendering only on Linux to ensure in-window rendering
# without imposing Mesa settings on Windows/macOS.
if sys.platform.startswith('linux'):
    def _find_dri_path():
        override_dri = os.environ.get('SVV_LIBGL_DRIVERS_PATH')
        if override_dri and os.path.isdir(override_dri):
            return override_dri
        conda_prefix = os.environ.get('CONDA_PREFIX', '')
        candidates = []
        if conda_prefix:
            candidates.extend([
                os.path.join(conda_prefix, 'lib', 'dri'),
                os.path.join(conda_prefix, 'x86_64-conda-linux-gnu', 'sysroot', 'usr', 'lib64', 'dri'),
            ])
        candidates.extend([
            '/usr/lib/x86_64-linux-gnu/dri',
            '/usr/lib64/dri',
            '/usr/lib/dri'
        ])
        for dri_path in candidates:
            if not os.path.isdir(dri_path):
                continue
            if (os.path.isfile(os.path.join(dri_path, 'swrast_dri.so')) or
                    os.path.isfile(os.path.join(dri_path, 'llvmpipe_dri.so'))):
                return dri_path
        return None

    # Default to system GL unless explicitly opting into software
    gl_mode = os.environ.get('SVV_GUI_GL_MODE', 'system').strip().lower()  # 'software' or 'system'
    dri_path = _find_dri_path() if gl_mode != 'system' else None
    if gl_mode != 'system' and not dri_path:
        # No software driver found; fall back to system GL
        os.environ['SVV_GUI_GL_MODE'] = 'system'
        gl_mode = 'system'

    if gl_mode != 'system':
        # Prefer software rendering to avoid GPU/driver issues on varied Linux setups
        os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
        os.environ.setdefault('GALLIUM_DRIVER', 'llvmpipe')
        os.environ.setdefault('MESA_GL_VERSION_OVERRIDE', '3.3')
        sw_driver = os.environ.get('SVV_GUI_SOFTWARE_DRIVER', 'llvmpipe')
        os.environ.setdefault('MESA_LOADER_DRIVER_OVERRIDE', sw_driver)

        # Qt side: force software OpenGL and use XCB when on Wayland
        os.environ.setdefault('QT_OPENGL', 'software')
        if 'WAYLAND_DISPLAY' in os.environ:
            os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

        # Set DRI drivers path when present
        if dri_path:
            os.environ.setdefault('LIBGL_DRIVERS_PATH', dri_path)
            os.environ.setdefault('SVV_LIBGL_DRIVERS_PATH', dri_path)
    else:
        for var in ("LIBGL_ALWAYS_SOFTWARE", "GALLIUM_DRIVER", "MESA_GL_VERSION_OVERRIDE",
                    "MESA_LOADER_DRIVER_OVERRIDE", "LIBGL_DRIVERS_PATH", "SVV_LIBGL_DRIVERS_PATH",
                    "QT_OPENGL"):
            os.environ.pop(var, None)
        os.environ.setdefault('QT_OPENGL', 'desktop')

    # Optional debug output for GL setup
    if os.environ.get('SVV_GUI_DEBUG_GL') == '1':
        debug_state = {
            'gl_mode': gl_mode,
            'SVV_GUI_SOFTWARE_DRIVER': os.environ.get('SVV_GUI_SOFTWARE_DRIVER'),
            'MESA_LOADER_DRIVER_OVERRIDE': os.environ.get('MESA_LOADER_DRIVER_OVERRIDE'),
            'LIBGL_DRIVERS_PATH': os.environ.get('LIBGL_DRIVERS_PATH'),
            'QT_QPA_PLATFORM': os.environ.get('QT_QPA_PLATFORM'),
            'QT_OPENGL': os.environ.get('QT_OPENGL'),
            'WAYLAND_DISPLAY': os.environ.get('WAYLAND_DISPLAY'),
        }
        print('[svv.gui] GL debug config:', debug_state)

import numpy as np
from PySide6.QtWidgets import QVBoxLayout, QWidget, QLabel, QApplication, QFrame
from PySide6.QtCore import Signal, QTimer, Qt, QRectF
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QFontMetrics, QPalette
from svv.visualize.gui.theme import CADTheme


class ScaleBarWidget(QLabel):
    """
    A minimal scale bar widget using QLabel for reliable rendering over VTK.

    Displays a scale bar with measurement value and unit using HTML/CSS
    which renders reliably over OpenGL surfaces.
    """

    def __init__(self, parent=None, unit_label="cm"):
        """
        Initialize the scale bar widget.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget (typically VTKWidget)
        unit_label : str
            The unit label to display (e.g., "mm", "cm", "m")
        """
        super().__init__(parent)
        self._unit_label = unit_label
        self._pixels_per_unit = 100.0
        self._min_bar_width = 40
        self._max_bar_width = 100
        self._current_bar_width = 60

        # Allow mouse events to pass through
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        # Style the label with a visible background
        self.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 220);
                border: 1px solid rgba(100, 100, 100, 150);
                border-radius: 4px;
                padding: 4px 8px;
                color: #1a1a1a;
                font-family: Arial, sans-serif;
                font-size: 9pt;
            }
        """)

        self._update_display()

    def set_unit_label(self, label: str):
        """Set the unit label (e.g., 'mm', 'cm')."""
        self._unit_label = label
        self._update_display()

    def set_pixels_per_unit(self, pixels: float):
        """Set the number of pixels per world unit."""
        self._pixels_per_unit = max(0.1, pixels)
        self._update_display()

    def _calculate_nice_length(self) -> tuple:
        """Calculate a nice rounded reference length and its pixel width."""
        target_pixels = 60
        raw_length = target_pixels / self._pixels_per_unit

        # Nice values: 1, 2, 5, 10, 20, 50, 100, etc.
        nice_values = []
        for exp in range(-4, 5):
            base = 10 ** exp
            nice_values.extend([base, 2 * base, 5 * base])
        nice_values.sort()

        # Find closest nice value
        best_length = 1.0
        best_diff = float('inf')
        for nv in nice_values:
            diff = abs(nv - raw_length)
            if diff < best_diff:
                best_diff = diff
                best_length = nv

        pixel_width = best_length * self._pixels_per_unit

        # Clamp to reasonable range
        if pixel_width < self._min_bar_width:
            for nv in nice_values:
                if nv > best_length:
                    test_pixels = nv * self._pixels_per_unit
                    if test_pixels >= self._min_bar_width:
                        best_length = nv
                        pixel_width = test_pixels
                        break
        elif pixel_width > self._max_bar_width:
            for nv in reversed(nice_values):
                if nv < best_length:
                    test_pixels = nv * self._pixels_per_unit
                    if test_pixels <= self._max_bar_width:
                        best_length = nv
                        pixel_width = test_pixels
                        break

        # Format label: "10 mm" style
        if best_length >= 1:
            if best_length == int(best_length):
                label = f"{int(best_length)} {self._unit_label}"
            else:
                label = f"{best_length:.1f} {self._unit_label}"
        elif best_length >= 0.01:
            label = f"{best_length:.2f} {self._unit_label}"
        else:
            label = f"{best_length:.3f} {self._unit_label}"

        return best_length, pixel_width, label

    def _update_display(self):
        """Update the scale bar display with current values."""
        ref_length, bar_width, label_text = self._calculate_nice_length()
        bar_width = int(max(self._min_bar_width, min(self._max_bar_width, bar_width)))

        # Only update if something actually changed
        new_text = label_text
        if hasattr(self, '_last_label') and self._last_label == new_text:
            return
        self._last_label = new_text
        self._current_bar_width = bar_width

        # Use a fixed-width bar representation for consistent display
        # This avoids visual artifacts from changing character counts
        bar_chars = "─" * 8  # Fixed 8 characters
        display_text = f"├{bar_chars}┤\n{label_text}"

        # Clear and update
        self.clear()
        self.setText(display_text)
        self.setAlignment(Qt.AlignCenter)
        self.adjustSize()
        self.repaint()  # Force immediate repaint


class _OffscreenBlitWidget(QWidget):
    """
    Fallback 3D viewport that renders VTK offscreen and blits frames to a QLabel.

    This is used on macOS environments where creating an embedded QtInteractor
    can hang during OpenGL context initialization.
    """

    _CLICK_TOLERANCE_PX = 5

    def __init__(self, plotter, parent=None):
        super().__init__(parent)
        from PySide6.QtGui import QImage, QPixmap

        self._plotter = plotter
        self._QImage = QImage
        self._QPixmap = QPixmap
        self.pick_callback = None
        self._pickable_meshes = []

        self._display = QLabel(self)
        self._display.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._display)

        self.setMouseTracking(True)
        self._display.setMouseTracking(True)
        self._press_pos = None

        self._vtk_iren = None
        try:
            iren = plotter.iren
            if iren is not None:
                self._vtk_iren = getattr(iren, 'interactor', iren)
                try:
                    self._vtk_iren.Initialize()
                except Exception:
                    pass
        except Exception:
            pass

        self._blit_timer = QTimer(self)
        self._blit_timer.setSingleShot(True)
        self._blit_timer.timeout.connect(self.blit)

        try:
            import vtk as _vtk

            plotter.ren_win.AddObserver(
                _vtk.vtkCommand.RenderEvent,
                lambda _obj, _ev: self.schedule_blit(),
            )
        except Exception:
            pass

    def blit(self):
        """Capture the offscreen framebuffer and display it in the label."""
        try:
            img = self._plotter.screenshot(return_img=True)
            if img is None:
                return
            h, w, ch = img.shape
            qimg = self._QImage(
                img.tobytes(),
                w,
                h,
                w * ch,
                self._QImage.Format.Format_RGB888,
            )
            self._display.setPixmap(self._QPixmap.fromImage(qimg))
        except Exception:
            pass

    def schedule_blit(self, delay_ms=16):
        """Request a debounced framebuffer copy."""
        if not self._blit_timer.isActive():
            self._blit_timer.start(delay_ms)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        size = event.size()
        width, height = size.width(), size.height()
        if width > 0 and height > 0:
            try:
                self._plotter.window_size = (width, height)
            except Exception:
                pass
            self.schedule_blit()

    def _ray_pick(self, display_x, display_y):
        """Return the closest ray-hit point on a registered pickable mesh."""
        try:
            import vtk as _vtk

            renderer = self._plotter.renderer
            coord = _vtk.vtkCoordinate()
            coord.SetCoordinateSystemToDisplay()
            coord.SetValue(float(display_x), float(self._plotter.window_size[1] - display_y), 0.0)
            world = np.array(coord.GetComputedWorldValue(renderer))

            cam_pos = np.array(self._plotter.camera.position)
            direction = world - cam_pos
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                return None
            direction /= norm
            ray_end = cam_pos + direction * norm * 1000.0

            best_point = None
            best_dist = float('inf')
            for mesh in self._pickable_meshes:
                try:
                    points, _ = mesh.ray_trace(cam_pos, ray_end)
                    if points is None or len(points) == 0:
                        continue
                    dists = np.linalg.norm(points - cam_pos, axis=1)
                    idx = int(np.argmin(dists))
                    if dists[idx] < best_dist:
                        best_dist = dists[idx]
                        best_point = points[idx]
                except Exception:
                    continue
            return best_point
        except Exception:
            return None

    def _get_pos(self, event):
        pt = event.position().toPoint() if hasattr(event, 'position') else event.pos()
        return int(pt.x()), int(pt.y())

    def _forward_pos(self, event):
        if self._vtk_iren is None:
            return
        x, y = self._get_pos(event)
        ctrl = int(bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier))
        shift = int(bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier))
        self._vtk_iren.SetEventInformationFlipY(x, y, ctrl, shift)

    def mousePressEvent(self, event):
        self._press_pos = self._get_pos(event)
        if self._vtk_iren is None:
            return
        self._forward_pos(event)
        button = event.button()
        if button == Qt.MouseButton.LeftButton:
            self._vtk_iren.LeftButtonPressEvent()
        elif button == Qt.MouseButton.RightButton:
            self._vtk_iren.RightButtonPressEvent()
        elif button == Qt.MouseButton.MiddleButton:
            self._vtk_iren.MiddleButtonPressEvent()
        self.schedule_blit()

    def mouseReleaseEvent(self, event):
        release_pos = self._get_pos(event)

        if self._vtk_iren is not None:
            self._forward_pos(event)
            button = event.button()
            if button == Qt.MouseButton.LeftButton:
                self._vtk_iren.LeftButtonReleaseEvent()
            elif button == Qt.MouseButton.RightButton:
                self._vtk_iren.RightButtonReleaseEvent()
            elif button == Qt.MouseButton.MiddleButton:
                self._vtk_iren.MiddleButtonReleaseEvent()

        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._press_pos is not None
            and self.pick_callback is not None
        ):
            dx = release_pos[0] - self._press_pos[0]
            dy = release_pos[1] - self._press_pos[1]
            if dx * dx + dy * dy <= self._CLICK_TOLERANCE_PX ** 2:
                hit = self._ray_pick(release_pos[0], release_pos[1])
                if hit is not None:
                    try:
                        self.pick_callback(hit)
                    except Exception:
                        pass

        self._press_pos = None
        self.schedule_blit()

    def mouseMoveEvent(self, event):
        if self._vtk_iren is None:
            return
        self._forward_pos(event)
        self._vtk_iren.MouseMoveEvent()
        if event.buttons():
            self.schedule_blit()

    def wheelEvent(self, event):
        if self._vtk_iren is None:
            return
        self._forward_pos(event)
        if event.angleDelta().y() > 0:
            self._vtk_iren.MouseWheelForwardEvent()
        else:
            self._vtk_iren.MouseWheelBackwardEvent()
        self.schedule_blit()

    def stop(self):
        """Stop the blit timer before shutting down."""
        try:
            self._blit_timer.stop()
        except Exception:
            pass


class VTKWidget(QWidget):
    """
    Widget for 3D visualization of Domain objects using PyVista.
    """

    # Signals
    point_picked = Signal(object)  # Emitted when a point is picked (numpy array)

    def __init__(self, parent=None):
        """
        Initialize the VTK widget.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self.domain = None
        self.domain_actor = None
        self.points_actors = []
        self.direction_actors = []
        self.tree_actors = []
        self.connection_actors = []
        # Grouped actor mappings for visibility toggles
        self.tree_actor_groups = {}
        self.connection_actor_groups = {}
        # Per-group color tracking (group_id -> color used at render time)
        self.tree_group_colors = {}
        self.connection_group_colors = {}
        # Scale bar actor
        self._scale_bar_actor = None
        self._scale_bar_visible = True
        # Domain edge visibility state
        self._domain_edges_visible = True
        # Grid visibility state
        self._grid_visible = False
        self._grid_actor = None

        # Create layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self._layout = layout

        # Lazily initialize the VTK/PyVista interactor after the widget is shown.
        # On some platforms (notably macOS CI), creating the interactor during
        # __init__ can hang before the window is realized.
        self._pv = None
        self._QtInteractor = None
        self.plotter = None
        self._plotter_init_in_progress = False
        self._plotter_init_done = False
        self._plotter_init_failed = False
        self._offscreen_mode = False
        self._blit_widget = None

        self._vtk_placeholder = QLabel("Initializing 3D viewport…", self)
        self._vtk_placeholder.setAlignment(Qt.AlignCenter)
        self._vtk_placeholder.setWordWrap(True)
        self._vtk_placeholder.setStyleSheet(
            f"padding: 24px; color: {CADTheme.get_color('text', 'secondary')};"
        )
        layout.addWidget(self._vtk_placeholder)

        # Lightweight HUD overlay (created eagerly; rendering is driven by plotter availability)
        self._hud = QLabel(self)
        self._hud.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._hud.setStyleSheet(
            "color: #E0E0E0; background-color: rgba(30,30,30,180);"
            "border: 1px solid rgba(122,155,192,180); padding: 6px 10px; border-radius: 6px;"
        )
        self._hud.hide()

    def showEvent(self, event):
        super().showEvent(event)
        self._ensure_plotter_initialized()

    def _ensure_plotter_initialized(self):
        disable_vtk = os.environ.get("SVV_GUI_DISABLE_VTK", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if disable_vtk:
            if getattr(self, "_vtk_placeholder", None) is not None:
                self._vtk_placeholder.setText(
                    "3D visualization disabled (SVV_GUI_DISABLE_VTK=1).\n\n"
                    "The GUI will run with limited 3D viewport features."
                )
                self._vtk_placeholder.setStyleSheet(
                    "padding: 20px; background-color: #FFF3CD; border: 1px solid #FFC107;"
                )
            self._plotter_init_failed = True
            self._plotter_init_in_progress = False
            return

        if self._plotter_init_done or self._plotter_init_in_progress or self._plotter_init_failed:
            return
        self._plotter_init_in_progress = True
        # Schedule after the event loop starts to avoid platform-specific hangs.
        delay_ms = 200 if sys.platform == 'darwin' else 0
        QTimer.singleShot(delay_ms, self._init_plotter)

    @staticmethod
    def _should_use_offscreen():
        """Return True only for the known-bad macOS ARM + conda environment."""
        def _env_true(name):
            return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}

        if _env_true("SVV_GUI_FORCE_EMBEDDED_VTK"):
            return False
        if _env_true("SVV_GUI_FORCE_OFFSCREEN_VTK"):
            return True
        return (
            sys.platform == "darwin"
            and platform.machine().lower() == "arm64"
            and bool(os.environ.get("CONDA_PREFIX"))
        )

    def _init_plotter(self):
        if self._plotter_init_done or self._plotter_init_failed:
            self._plotter_init_in_progress = False
            return

        try:
            import pyvista as _pv
        except Exception as e:
            msg = (
                "Failed to import PyVista/PyVistaQt. On Linux, ensure libffi is installed "
                "(conda: `conda install -c conda-forge libffi>=3.4,<3.5`; system: `sudo apt install libffi8` on Ubuntu 22.04 "
                "or `libffi7` on Ubuntu 20.04). Also install dependencies: PySide6, pyvista, pyvistaqt."
            )
            self._show_vtk_error(msg)
            return

        self._pv = _pv

        try:
            import vtk
            vtk.vtkObject.GlobalWarningDisplayOff()
        except Exception:
            pass

        if self._should_use_offscreen():
            self._init_plotter_offscreen(_pv)
        else:
            self._init_plotter_embedded(_pv)

        if self.plotter is None or self._plotter_init_failed:
            return

        self._setup_plotter_common()

    def _init_plotter_embedded(self, _pv):
        """Create the standard embedded PyVista Qt interactor."""
        try:
            from pyvistaqt import QtInteractor as _QtInteractor
        except Exception as e:
            self._show_vtk_error(
                f"Failed to import PyVistaQt:\n{e}\n\n"
                "Install it with: pip install pyvistaqt"
            )
            return

        self._QtInteractor = _QtInteractor

        try:
            self.plotter = _QtInteractor(self)
            self._layout.addWidget(self.plotter.interactor)
            self._remove_placeholder()
        except Exception as e:
            self._show_vtk_error(
                f"3D Visualization unavailable:\n{e}\n\n"
                "This may be due to missing OpenGL libraries.\n"
                "The GUI will function with limited 3D visualization.\n\n"
                "To fix, install Mesa libraries:\n"
                "conda install -c conda-forge mesalib"
            )

    def _init_plotter_offscreen(self, _pv):
        """Create an offscreen plotter for the known-bad macOS environment."""
        try:
            self.plotter = _pv.Plotter(off_screen=True, window_size=(800, 600))
        except Exception as e:
            self._show_vtk_error(f"Failed to create offscreen plotter:\n{e}")
            return

        self._offscreen_mode = True
        self._blit_widget = _OffscreenBlitWidget(self.plotter, parent=self)
        self._layout.addWidget(self._blit_widget)
        self._remove_placeholder()

    def _setup_plotter_common(self):
        """Apply shared plotter setup after either init path succeeds."""
        if self._offscreen_mode:
            if self._blit_widget is not None:
                self._blit_widget.pick_callback = self._on_point_picked
        else:
            selection_color = CADTheme.get_color('viewport', 'selection')
            try:
                self.plotter.enable_surface_point_picking(
                    callback=self._on_point_picked,
                    show_message=False,
                    color=selection_color,
                    point_size=14,
                    tolerance=0.025,
                    pickable_window=False,
                )
            except (AttributeError, TypeError):
                try:
                    self.plotter.enable_surface_picking(
                        callback=self._on_surface_picked,
                        show_message=False,
                        color=selection_color,
                        point_size=14,
                    )
                except (AttributeError, TypeError):
                    try:
                        self.plotter.enable_point_picking(
                            callback=self._on_point_picked,
                            show_message=False,
                            color=selection_color,
                            point_size=14
                        )
                    except Exception:
                        pass

        bg_bottom = CADTheme.get_color('viewport', 'background-bottom')
        bg_top = CADTheme.get_color('viewport', 'background-top')
        self.plotter.set_background(bg_bottom, top=bg_top)
        try:
            self.plotter.enable_eye_dome_lighting()
        except Exception:
            pass
        try:
            self.plotter.enable_anti_aliasing()
        except Exception:
            pass
        self.plotter.show_axes()
        self.plotter.camera_position = 'iso'
        self._init_scale_bar()
        if self.domain is not None:
            try:
                self.set_domain(self.domain)
            except Exception:
                pass
        if self._blit_widget is not None:
            self._blit_widget.schedule_blit(delay_ms=50)

        self._plotter_init_done = True
        self._plotter_init_in_progress = False

    def _remove_placeholder(self):
        """Remove the initial viewport placeholder."""
        if getattr(self, "_vtk_placeholder", None) is not None:
            try:
                self._layout.removeWidget(self._vtk_placeholder)
            except Exception:
                pass
            try:
                self._vtk_placeholder.deleteLater()
            except Exception:
                pass
            self._vtk_placeholder = None

    def _show_vtk_error(self, message: str):
        """Replace the placeholder with a static error label."""
        error_label = QLabel(message)
        error_label.setWordWrap(True)
        error_label.setStyleSheet(
            "padding: 20px; background-color: #FFF3CD; border: 1px solid #FFC107;"
        )
        self._remove_placeholder()
        self._layout.addWidget(error_label)
        self.plotter = None
        self._plotter_init_failed = True
        self._plotter_init_in_progress = False

    def shutdown(self):
        """
        Release VTK/PyVista resources to help the application exit cleanly.

        This clears all actors and closes the underlying QtInteractor so that
        GPU/CPU memory is returned promptly.
        """
        if getattr(self, '_blit_widget', None) is not None:
            self._blit_widget.stop()

        # Stop scale bar update timer
        if hasattr(self, '_scale_bar_timer') and self._scale_bar_timer:
            try:
                self._scale_bar_timer.stop()
            except Exception:
                pass
            self._scale_bar_timer = None

        if getattr(self, "plotter", None) is not None:
            # Fast shutdown path: avoid per-actor removals which can be slow
            # when many vessel actors are present. Instead, drop references to
            # actors and let the plotter clear everything in one call.
            try:
                self.points_actors.clear()
                self.direction_actors.clear()
                self.tree_actors.clear()
                self.connection_actors.clear()
                self.tree_actor_groups.clear()
                self.connection_actor_groups.clear()
            except Exception:
                pass

            try:
                # Clear all meshes and close the interactor window
                self.plotter.clear()
            except Exception:
                pass
            try:
                self.plotter.close()
            except Exception:
                pass
            self.plotter = None

    def _init_scale_bar(self):
        """
        Initialize the scale bar/ruler widget in the viewport.

        Uses a custom Qt widget positioned in the lower-right corner
        that displays a reference length based on the current zoom level.
        """
        if not self.plotter:
            return

        try:
            # Get unit label from domain if available
            # Default to 'cm' to match UnitSystem default (cm, g, s)
            unit_label = "cm"
            if self.domain is not None and hasattr(self.domain, 'unit'):
                unit_label = self.domain.unit or "cm"

            # Create the custom scale bar widget as a child of this widget
            self._scale_bar_widget = ScaleBarWidget(self, unit_label=unit_label)
            self._scale_bar_widget.show()

            # Position the scale bar in the lower-right corner
            self._position_scale_bar()

            # Connect to plotter's render callback to update scale bar
            self._setup_camera_callback()

            self._scale_bar_visible = True
            # Keep reference for toggle methods (backward compatibility)
            self._scale_bar_actor = self._scale_bar_widget

        except Exception as e:
            # Scale bar is non-critical - don't fail if it can't be created
            self._scale_bar_widget = None
            self._scale_bar_actor = None
            print(f"[VTKWidget] Scale bar initialization failed: {e}")

    def _position_scale_bar(self):
        """Position the scale bar widget in the lower-right corner."""
        if not hasattr(self, '_scale_bar_widget') or not self._scale_bar_widget:
            return

        # Ensure the widget is sized correctly first
        self._scale_bar_widget.adjustSize()

        margin = 12
        widget_width = self._scale_bar_widget.width()
        widget_height = self._scale_bar_widget.height()

        x = self.width() - widget_width - margin
        y = self.height() - widget_height - margin

        self._scale_bar_widget.move(max(0, x), max(0, y))
        self._scale_bar_widget.raise_()  # Ensure it's on top

    def _setup_camera_callback(self):
        """Set up callback to update scale bar when camera changes."""
        if not self.plotter:
            return

        try:
            # Add an observer to the renderer's camera
            renderer = self.plotter.renderer
            if renderer and renderer.GetActiveCamera():
                # Use a timer to periodically update (VTK camera callbacks can be tricky)
                # 250ms is enough for smooth updates without causing visual artifacts
                self._scale_bar_timer = QTimer(self)
                self._scale_bar_timer.timeout.connect(self._update_scale_bar)
                self._scale_bar_timer.start(250)
        except Exception:
            pass

    def _update_scale_bar(self):
        """Update the scale bar based on current camera/view settings."""
        if not hasattr(self, '_scale_bar_widget') or not self._scale_bar_widget:
            return
        if not self._scale_bar_visible:
            return
        if not self.plotter:
            return

        try:
            renderer = self.plotter.renderer
            if not renderer:
                return

            camera = renderer.GetActiveCamera()
            if not camera:
                return

            # Calculate pixels per world unit based on camera settings
            # Get the view size in pixels
            view_width, view_height = renderer.GetSize()
            if view_width <= 0 or view_height <= 0:
                return

            # Get the parallel scale (for parallel projection) or calculate from perspective
            if camera.GetParallelProjection():
                # Parallel projection: parallel scale is half the height in world units
                parallel_scale = camera.GetParallelScale()
                if parallel_scale > 0:
                    pixels_per_unit = view_height / (2.0 * parallel_scale)
                else:
                    pixels_per_unit = 100.0
            else:
                # Perspective projection: estimate from camera distance and view angle
                distance = camera.GetDistance()
                view_angle = camera.GetViewAngle()

                if distance > 0 and view_angle > 0:
                    import math
                    # Height in world units at the focal point
                    world_height = 2.0 * distance * math.tan(math.radians(view_angle / 2.0))
                    if world_height > 0:
                        pixels_per_unit = view_height / world_height
                    else:
                        pixels_per_unit = 100.0
                else:
                    pixels_per_unit = 100.0

            self._scale_bar_widget.set_pixels_per_unit(pixels_per_unit)
            # Reposition after size may have changed
            self._position_scale_bar()

        except Exception:
            pass

    def resizeEvent(self, event):
        """Handle resize to reposition scale bar."""
        super().resizeEvent(event)
        self._position_scale_bar()

    def toggle_scale_bar(self):
        """Toggle visibility of the scale bar."""
        if not hasattr(self, '_scale_bar_widget') or not self._scale_bar_widget:
            return

        self._scale_bar_visible = not self._scale_bar_visible
        if self._scale_bar_visible:
            self._scale_bar_widget.show()
        else:
            self._scale_bar_widget.hide()

    def set_scale_bar_visible(self, visible: bool):
        """
        Set scale bar visibility.

        Parameters
        ----------
        visible : bool
            True to show, False to hide
        """
        if not hasattr(self, '_scale_bar_widget') or not self._scale_bar_widget:
            return

        self._scale_bar_visible = visible
        if visible:
            self._scale_bar_widget.show()
        else:
            self._scale_bar_widget.hide()

    def is_scale_bar_visible(self) -> bool:
        """
        Check if scale bar is currently visible.

        Returns
        -------
        bool
            True if visible, False otherwise
        """
        return self._scale_bar_visible

    def set_scale_bar_unit(self, unit_label: str):
        """
        Set the unit label for the scale bar.

        Parameters
        ----------
        unit_label : str
            Unit label (e.g., 'mm', 'cm', 'm')
        """
        if hasattr(self, '_scale_bar_widget') and self._scale_bar_widget:
            self._scale_bar_widget.set_unit_label(unit_label)

    def set_domain(self, domain):
        """
        Set and visualize the domain.

        Parameters
        ----------
        domain : svv.domain.Domain
            Domain object to visualize
        """
        self.domain = domain
        if not self.plotter:
            return

        self.clear()
        if self._offscreen_mode and self._blit_widget is not None:
            self._blit_widget._pickable_meshes = []

        # Add domain boundary if available
        if hasattr(domain, 'boundary') and domain.boundary is not None:
            surface_color = CADTheme.get_color('viewport', 'domain-surface')
            edge_color = CADTheme.get_color('viewport', 'domain-edge')
            self.domain_actor = self.plotter.add_mesh(
                domain.boundary,
                color=surface_color,
                opacity=0.35,
                show_edges=self._domain_edges_visible,
                edge_color=edge_color,
                line_width=1,
                specular=0.15,
                smooth_shading=True,
                name='domain'
            )
            if self._offscreen_mode and self._blit_widget is not None:
                self._blit_widget._pickable_meshes = [domain.boundary]

        # Update scale bar unit if domain has unit info
        if hasattr(domain, 'unit') and domain.unit:
            self.set_scale_bar_unit(domain.unit)

        def _reset_and_render():
            if not self.plotter:
                return
            try:
                self.plotter.reset_camera()
            except Exception:
                pass
            try:
                self.plotter.render()
            except Exception:
                pass

        # Reset camera to show full domain. Some platform/runner combinations can hang
        # when rendering before the widget is realized, so defer a tick when needed.
        if self.isVisible():
            _reset_and_render()
        else:
            QTimer.singleShot(0, _reset_and_render)

    def _show_hud(self, message: str, duration_ms: int = 1400):
        """Show a transient HUD message in the viewport."""
        if not self.plotter:
            return
        self._hud.setText(message)
        self._hud.adjustSize()
        self._hud.move(16, 16)
        self._hud.show()
        QTimer.singleShot(duration_ms, self._hud.hide)

    def view_iso(self):
        if self.plotter:
            self.plotter.view_isometric()
            self.plotter.render()
            self._show_hud("Isometric")

    def view_top(self):
        if self.plotter:
            self.plotter.view_xy()
            self.plotter.render()
            self._show_hud("Top")

    def view_front(self):
        if self.plotter:
            self.plotter.view_yz()
            self.plotter.render()
            self._show_hud("Front")

    def view_right(self):
        if self.plotter:
            self.plotter.view_xz()
            self.plotter.render()
            self._show_hud("Right")

    def add_start_point(self, point, index=None, color='red'):
        """
        Add a start point marker to the visualization.

        Parameters
        ----------
        point : array-like
            3D coordinates of the point [x, y, z]
        index : int, optional
            Index/label for the point
        color : str, optional
            Color of the point marker

        Returns
        -------
        actor
            PyVista actor for the point
        """
        if not self.plotter:
            return None

        point = np.asarray(point).flatten()

        # Create sphere marker
        sphere = self._pv.Sphere(radius=self._get_marker_size() * 1.2, center=point)
        actor = self.plotter.add_mesh(
            sphere,
            color=color,
            specular=0.2,
            smooth_shading=True,
            name=f'start_point_{len(self.points_actors)}'
        )

        # Add label if index provided
        if index is not None:
            self.plotter.add_point_labels(
                [point],
                [f'P{index}'],
                point_size=10,
                font_size=12,
                text_color='black',
                name=f'label_{len(self.points_actors)}'
            )

        self.points_actors.append(actor)
        self.plotter.render()
        return actor

    def add_direction(self, point, direction, length=None, color='blue'):
        """
        Add a direction vector visualization.

        Parameters
        ----------
        point : array-like
            Starting point [x, y, z]
        direction : array-like
            Direction vector [dx, dy, dz]
        length : float, optional
            Length of the arrow. If None, uses domain characteristic length
        color : str, optional
            Color of the arrow

        Returns
        -------
        actor
            PyVista actor for the direction arrow
        """
        if not self.plotter:
            return None

        point = np.asarray(point).flatten()
        direction = np.asarray(direction).flatten()
        direction = direction / np.linalg.norm(direction)

        if length is None:
            length = self._get_arrow_length()

        # Create arrow
        end_point = point + direction * length
        arrow = self._pv.Line(point, end_point)

        actor = self.plotter.add_mesh(
            arrow,
            color=color,
            line_width=3,
            name=f'direction_{len(self.direction_actors)}'
        )

        # Add arrow head
        cone_height = length * 0.15
        cone = self._pv.Cone(
            center=end_point,
            direction=direction,
            height=cone_height,
            radius=cone_height * 0.5
        )
        cone_actor = self.plotter.add_mesh(
            cone,
            color=color,
            name=f'direction_cone_{len(self.direction_actors)}'
        )

        self.direction_actors.append((actor, cone_actor))
        self.plotter.render()
        return actor

    def add_tree(self, tree, color='red', label=None, group_id=None):
        """
        Add a Tree visualization.

        Parameters
        ----------
        tree : svv.tree.Tree
            Tree object to visualize
        color : str, optional
            Color for the tree vessels
        label : str, optional
            Base label for actor naming; ensures unique actor names.

        Returns
        -------
        list
            List of actors for the tree vessels
        """
        if not self.plotter:
            return []

        actors = []
        base = label or f"tree_{len(self.tree_actors)}"
        for i in range(tree.data.shape[0]):
            center = (tree.data[i, 0:3] + tree.data[i, 3:6]) / 2
            direction = tree.data.get('w_basis', i)
            radius = tree.data.get('radius', i)
            length = tree.data.get('length', i)

            vessel = self._pv.Cylinder(
                center=center,
                direction=direction,
                radius=radius,
                height=length
            )
            actor = self.plotter.add_mesh(
                vessel,
                color=color,
                name=f'{base}_vessel_{i}'
            )
            actors.append(actor)
            # Periodically process Qt events to keep the GUI responsive
            if i % 100 == 0:
                try:
                    QApplication.processEvents()
                except Exception:
                    pass

        self.tree_actors.extend(actors)
        if group_id is not None:
            self.tree_actor_groups[group_id] = actors
            self.tree_group_colors[group_id] = color
        self.plotter.render()
        return actors

    def add_connection_vessels(self, vessels, color='red', label=None, group_id=None):
        """
        Add connecting vessels (array of segments with radius).

        Parameters
        ----------
        vessels : ndarray (n, 7)
            Each row: [x0,y0,z0,x1,y1,z1,radius]
        color : str
            Color of the vessels
        """
        if vessels is None or vessels.size == 0:
            return []
        if not self.plotter:
            return []
        actors = []
        base = label or f"connection_{len(self.connection_actors)}"
        for idx, seg in enumerate(vessels):
            p0 = seg[0:3]
            p1 = seg[3:6]
            radius = seg[6]
            direction = p1 - p0
            length = np.linalg.norm(direction)
            if length <= 0:
                continue
            direction = direction / length
            center = (p0 + p1) / 2
            cyl = self._pv.Cylinder(center=center, direction=direction, radius=radius, height=length)
            actor = self.plotter.add_mesh(cyl, color=color, name=f"{base}_seg_{idx}")
            actors.append(actor)
            if idx % 200 == 0:
                try:
                    QApplication.processEvents()
                except Exception:
                    pass
        self.connection_actors.extend(actors)
        if group_id is not None:
            self.connection_actor_groups[group_id] = actors
            self.connection_group_colors[group_id] = color
        self.plotter.render()
        return actors

    def clear_points(self):
        """Clear all start point markers."""
        if not self.plotter:
            return
        for actor in self.points_actors:
            self.plotter.remove_actor(actor)
        self.points_actors.clear()
        self.plotter.render()

    def clear_directions(self):
        """Clear all direction arrows."""
        if not self.plotter:
            return
        for actor_tuple in self.direction_actors:
            for actor in actor_tuple:
                self.plotter.remove_actor(actor)
        self.direction_actors.clear()
        self.plotter.render()

    def clear_trees(self):
        """Clear all tree visualizations."""
        if not self.plotter:
            return
        for actor in self.tree_actors:
            self.plotter.remove_actor(actor)
        self.tree_actors.clear()
        self.tree_actor_groups.clear()
        self.tree_group_colors.clear()
        self.plotter.render()

    def clear_connections(self):
        """Clear all connection visualizations."""
        if not self.plotter:
            return
        for actor in list(self.connection_actors):
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self.connection_actors.clear()
        self.connection_actor_groups.clear()
        self.connection_group_colors.clear()
        self.plotter.render()

    def clear(self):
        """Clear all visualizations except the domain."""
        self.clear_points()
        self.clear_directions()
        self.clear_trees()
        self.clear_connections()

    def reset_camera(self):
        """Reset the camera to show the full domain."""
        if not self.plotter:
            return
        self.plotter.reset_camera()
        self.plotter.render()

    def toggle_domain_visibility(self):
        """Toggle the visibility of the domain mesh."""
        if not self.plotter or self.domain_actor is None:
            return
        self.domain_actor.SetVisibility(not self.domain_actor.GetVisibility())
        self.plotter.render()

    def set_domain_edges_visible(self, visible: bool):
        """
        Show or hide mesh edges on the domain surface.

        Parameters
        ----------
        visible : bool
            True to show edges, False to hide them.
        """
        self._domain_edges_visible = bool(visible)
        if not self.plotter or self.domain_actor is None:
            return
        try:
            prop = self.domain_actor.GetProperty()
            if self._domain_edges_visible:
                prop.EdgeVisibilityOn()
            else:
                prop.EdgeVisibilityOff()
        except Exception:
            # Fallback for any backend that lacks EdgeVisibility helpers
            try:
                self.domain_actor.GetProperty().SetEdgeVisibility(1 if self._domain_edges_visible else 0)
            except Exception:
                pass
        self.plotter.render()

    def toggle_grid(self):
        """Toggle the visibility of the 3D grid."""
        self._grid_visible = not self._grid_visible
        self.set_grid_visible(self._grid_visible)

    def set_grid_visible(self, visible: bool):
        """
        Show or hide the 3D grid in the viewport.

        Parameters
        ----------
        visible : bool
            True to show grid, False to hide it.
        """
        self._grid_visible = bool(visible)
        if not self.plotter:
            return

        try:
            if self._grid_visible:
                # Show grid with theme colors
                grid_color = CADTheme.get_color('viewport', 'grid')
                self.plotter.show_grid(
                    color=grid_color,
                    location='back',
                    grid='back',
                    show_xaxis=True,
                    show_yaxis=True,
                    show_zaxis=True,
                    font_size=8
                )
            else:
                # Remove the grid
                self.plotter.remove_bounds_axes()
        except Exception:
            pass

        self.plotter.render()

    def is_grid_visible(self) -> bool:
        """
        Check if the grid is currently visible.

        Returns
        -------
        bool
            True if visible, False otherwise
        """
        return self._grid_visible

    def draw_forest(self, forest):
        """
        Render a forest with optional connections.

        Parameters
        ----------
        forest : svv.forest.forest.Forest
            Forest to render
        """
        if not self.plotter:
            return

        self.clear_trees()
        self.clear_connections()

        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']

        has_conn = getattr(forest, 'connections', None) is not None and \
            getattr(forest.connections, 'tree_connections', None)

        if has_conn:
            for net_idx, tree_conn in enumerate(forest.connections.tree_connections):
                # Trees within this network
                for tree_idx, tree in enumerate(tree_conn.connected_network):
                    color = colors[(net_idx + tree_idx) % len(colors)]
                    self.add_tree(
                        tree,
                        color=color,
                        label=f"forest_{net_idx}_{tree_idx}",
                        group_id=("forest", net_idx, tree_idx),
                    )
                # Connection vessels between trees in this network
                for tree_idx, vessel_list in enumerate(tree_conn.vessels):
                    color = colors[tree_idx % len(colors)]
                    for conn_idx, vessel in enumerate(vessel_list):
                        self.add_connection_vessels(
                            vessel,
                            color=color,
                            label=f"conn_{net_idx}_{tree_idx}_{conn_idx}",
                            group_id=("conn", net_idx, tree_idx, conn_idx),
                        )
        else:
            for net_idx, network in enumerate(forest.networks):
                for tree_idx, tree in enumerate(network):
                    color = colors[(net_idx + tree_idx) % len(colors)]
                    self.add_tree(
                        tree,
                        color=color,
                        label=f"forest_{net_idx}_{tree_idx}",
                        group_id=("forest", net_idx, tree_idx),
                    )

    def _on_point_picked(self, point):
        """
        Callback for point picking.

        Parameters
        ----------
        point : array-like
            Picked point coordinates
        """
        if point is not None:
            arr = np.asarray(point)
            # Ensure we have a single 1D point of 3 coordinates
            if arr.ndim == 2:
                # Multiple points or (1, 3) shaped - take first point
                arr = arr[0]
            self.point_picked.emit(arr.flatten())

    def _on_surface_picked(self, *args):
        """
        Callback for surface picking (handles various PyVista versions).

        Different PyVista versions pass different arguments:
        - Some pass just the point coordinates
        - Some pass (mesh, point_id)
        - Some pass (point, mesh)
        """
        point = None

        if len(args) == 1:
            # Single argument - likely the point coordinates or a mesh
            arg = args[0]
            if hasattr(arg, 'points'):
                # It's a mesh - get the picked point (usually first point of selection)
                if arg.n_points > 0:
                    point = arg.points[0]
            else:
                # Assume it's point coordinates
                point = np.asarray(arg)
        elif len(args) >= 2:
            # Multiple arguments - check each one
            for arg in args:
                if isinstance(arg, np.ndarray) and arg.shape == (3,):
                    point = arg
                    break
                elif hasattr(arg, '__len__') and len(arg) == 3:
                    point = np.asarray(arg)
                    break
                elif hasattr(arg, 'points') and arg.n_points > 0:
                    # It's a mesh
                    point = arg.points[0]
                    break

        if point is not None:
            self.point_picked.emit(np.asarray(point).flatten())

    def _get_marker_size(self):
        """
        Calculate appropriate marker size based on domain size.

        Returns
        -------
        float
            Marker radius
        """
        if self.domain is not None and hasattr(self.domain, 'characteristic_length'):
            return self.domain.characteristic_length * 0.05
        return 0.1

    def _get_arrow_length(self):
        """
        Calculate appropriate arrow length based on domain size.

        Returns
        -------
        float
            Arrow length
        """
        if self.domain is not None and hasattr(self.domain, 'characteristic_length'):
            return self.domain.characteristic_length * 0.2
        return 1.0

    # ---- Visibility helpers for Model Tree ----
    def set_tree_visibility(self, mode, net_idx, tree_idx, visible: bool):
        """
        Toggle visibility of a single tree.

        Parameters
        ----------
        mode : str
            'single' for standalone tree, 'forest' for forest trees.
        net_idx : int
            Network index (0 for single tree).
        tree_idx : int
            Tree index within the network.
        visible : bool
            True to show, False to hide.
        """
        if not self.plotter:
            return
        key = ("single", tree_idx) if mode == "single" else ("forest", net_idx, tree_idx)
        actors = self.tree_actor_groups.get(key)
        if not actors:
            return
        for actor in actors:
            try:
                actor.SetVisibility(bool(visible))
            except Exception:
                pass
        self.plotter.render()

    def set_network_visibility(self, net_idx: int, visible: bool):
        """
        Toggle visibility for all trees and connection vessels within a network.

        Parameters
        ----------
        net_idx : int
            Network index
        visible : bool
            True to show, False to hide.
        """
        if not self.plotter:
            return
        # Trees in this network
        for key, actors in list(self.tree_actor_groups.items()):
            if isinstance(key, tuple) and len(key) >= 3 and key[0] == "forest" and key[1] == net_idx:
                for actor in actors:
                    try:
                        actor.SetVisibility(bool(visible))
                    except Exception:
                        pass
        # Connection vessels in this network
        for key, actors in list(self.connection_actor_groups.items()):
            if isinstance(key, tuple) and len(key) >= 2 and key[0] == "conn" and key[1] == net_idx:
                for actor in actors:
                    try:
                        actor.SetVisibility(bool(visible))
                    except Exception:
                        pass
        self.plotter.render()

    # ---- Color helpers for Model Tree ----
    def set_tree_color(self, group_id, rgb):
        """
        Change the color of all actors belonging to a tree group.

        Parameters
        ----------
        group_id : tuple
            Group identifier, e.g. ("forest", net_idx, tree_idx).
        rgb : tuple of float
            (r, g, b) with values in 0-1.
        """
        if not self.plotter:
            return
        actors = self.tree_actor_groups.get(group_id)
        if not actors:
            return
        for actor in actors:
            try:
                actor.GetProperty().SetColor(rgb)
            except Exception:
                pass
        self.tree_group_colors[group_id] = rgb
        self.plotter.render()

    def set_connection_color(self, group_id, rgb):
        """
        Change the color of all actors belonging to a connection group.

        Parameters
        ----------
        group_id : tuple
            Group identifier, e.g. ("conn", net_idx, tree_idx, conn_idx).
        rgb : tuple of float
            (r, g, b) with values in 0-1.
        """
        if not self.plotter:
            return
        actors = self.connection_actor_groups.get(group_id)
        if not actors:
            return
        for actor in actors:
            try:
                actor.GetProperty().SetColor(rgb)
            except Exception:
                pass
        self.connection_group_colors[group_id] = rgb
        self.plotter.render()

    def set_network_color(self, net_idx, rgb):
        """
        Change the color of all trees and connections in a network.

        Parameters
        ----------
        net_idx : int
            Network index.
        rgb : tuple of float
            (r, g, b) with values in 0-1.
        """
        for key in list(self.tree_actor_groups):
            if isinstance(key, tuple) and len(key) >= 3 and key[0] == "forest" and key[1] == net_idx:
                self.set_tree_color(key, rgb)
        for key in list(self.connection_actor_groups):
            if isinstance(key, tuple) and len(key) >= 2 and key[0] == "conn" and key[1] == net_idx:
                self.set_connection_color(key, rgb)
