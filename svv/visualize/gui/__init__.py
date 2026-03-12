import os
import sys

# CRITICAL: Set GL environment BEFORE any Qt/VTK imports
# This must happen at module import time
if sys.platform == 'darwin':
    os.environ.setdefault('QT_MAC_WANTS_LAYER', '1')

if sys.platform.startswith('linux'):
    # If running on Wayland but lack the plugin, fall back to xcb early
    if 'WAYLAND_DISPLAY' in os.environ and not os.environ.get('QT_QPA_PLATFORM'):
        os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
    def _find_dri_path():
        """Return a DRI path containing swrast/llvmpipe if available."""
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
            if '/envs/' in conda_prefix:
                base_prefix = conda_prefix.split('/envs/')[0]
                candidates.append(os.path.join(base_prefix, 'x86_64-conda-linux-gnu', 'sysroot', 'usr', 'lib64', 'dri'))
        candidates.extend([
            '/usr/lib/x86_64-linux-gnu/dri',
            '/usr/lib64/dri',
            '/usr/lib/dri'
        ])
        for path in candidates:
            if not os.path.isdir(path):
                continue
            if (os.path.isfile(os.path.join(path, 'swrast_dri.so')) or
                    os.path.isfile(os.path.join(path, 'llvmpipe_dri.so'))):
                return path
        return None

    # Default to system GL; allow opt-in software via SVV_GUI_GL_MODE=software
    gl_mode = os.environ.get('SVV_GUI_GL_MODE', 'system').strip().lower()
    dri_path = None
    if gl_mode != 'system':
        dri_path = _find_dri_path()
        if not dri_path:
            # Fall back to system GL if no software driver is present
            os.environ['SVV_GUI_GL_MODE'] = 'system'
            gl_mode = 'system'

    if gl_mode != 'system':
        os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
        os.environ.setdefault('GALLIUM_DRIVER', 'llvmpipe')
        os.environ.setdefault('MESA_GL_VERSION_OVERRIDE', '3.3')
        os.environ.setdefault('MESA_LOADER_DRIVER_OVERRIDE', 'llvmpipe')
        os.environ.setdefault('QT_OPENGL', 'software')
        if dri_path:
            os.environ.setdefault('LIBGL_DRIVERS_PATH', dri_path)
            os.environ.setdefault('SVV_LIBGL_DRIVERS_PATH', dri_path)
        if 'WAYLAND_DISPLAY' in os.environ:
            os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
    else:
        # Ensure software-specific overrides don't pollute system GL
        for var in ("LIBGL_ALWAYS_SOFTWARE", "GALLIUM_DRIVER", "MESA_GL_VERSION_OVERRIDE",
                    "MESA_LOADER_DRIVER_OVERRIDE", "LIBGL_DRIVERS_PATH", "SVV_LIBGL_DRIVERS_PATH",
                    "QT_OPENGL"):
            os.environ.pop(var, None)
        os.environ.setdefault('QT_OPENGL', 'desktop')

# Unified CAD-styled GUI
from svv.visualize.gui.main_window import VascularizeGUI
VascularizeCADGUI = VascularizeGUI  # Backwards compatibility alias


def launch_gui(domain=None, block=True, style=None):
    """
    Launch the Vascularize GUI from a Python interpreter or script.

    Parameters
    ----------
    domain : optional
        Optional domain object to preload into the GUI.
    block : bool
        If True, start the Qt event loop and block until exit.
        If False, return ``(app, gui)`` without starting the loop.
    style : str, optional
        Deprecated. Previously used to choose alternate themes; now ignored.

    Returns
    -------
    tuple | None
        ``(app, gui)`` when ``block=False``, otherwise ``None``.
    """
    import sys
    from pathlib import Path
    from PySide6.QtWidgets import QApplication, QMessageBox
    from PySide6.QtCore import Qt, QSettings
    from PySide6.QtGui import QIcon
    import warnings
    from svv import __version__ as _svv_version
    from svv.telemetry import init_telemetry

    # On Linux, prefer Qt's software OpenGL to avoid GPU/driver issues
    # unless explicitly opting into system GL.
    if sys.platform.startswith('linux') and os.environ.get('SVV_GUI_GL_MODE', 'software') != 'system':
        try:
            QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL)
        except Exception:
            pass

    if style not in (None, 'cad'):
        warnings.warn(
            "The 'style' argument is deprecated; the GUI now uses a single CAD theme.",
            DeprecationWarning,
            stacklevel=2
        )

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("svVascularize")

    # Set application icon if available
    icon_path = Path(__file__).with_name("svIcon.png")
    if icon_path.is_file():
        icon = QIcon(str(icon_path))
        app.setWindowIcon(icon)

    # Initialize telemetry (optional; user opt-in + DSN-driven)
    settings = QSettings("svVascularize", "GUI")
    telemetry_enabled = False
    if os.environ.get("SVV_TELEMETRY_DISABLED", "").strip() != "1":
        key = "telemetry/enabled"
        if settings.contains(key):
            telemetry_enabled = settings.value(key, False, type=bool)
        else:
            # Ask the user once whether they want to send crash reports
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle("Crash Reporting")
            msg.setText(
                "Allow svVascularize to send anonymous crash reports?\n\n"
                "This helps improve stability. No model geometry or patient data "
                "is sent, but stack traces and basic environment info are included."
            )
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            choice = msg.exec()
            telemetry_enabled = choice == QMessageBox.Yes
            settings.setValue(key, telemetry_enabled)

    if telemetry_enabled:
        # Use DSN from environment when provided; otherwise allow the
        # telemetry module to fall back to its built-in default DSN.
        # Prefer the svVascularize-specific variable, but also honour the
        # standard SENTRY_DSN used by many deployments.
        dsn_env = os.environ.get("SVV_SENTRY_DSN", "").strip()
        if not dsn_env:
            dsn_env = os.environ.get("SENTRY_DSN", "").strip()
        dsn = dsn_env or None
        init_telemetry(dsn=dsn, release=_svv_version, environment="gui")

        # If the previous session did not exit cleanly, emit a telemetry
        # message so crashes that kill the process can be detected on the
        # next launch.
        previous_running = settings.value("session/running", False, type=bool)
        if previous_running:
            try:
                from svv.telemetry import capture_message

                last_action = settings.value("session/last_action", "", type=str)
                capture_message(
                    "Previous GUI session did not exit cleanly",
                    level="error",
                    last_action=last_action or "unknown",
                )
            except Exception:
                pass

    # Mark this session as running; a clean shutdown will clear this flag.
    settings.setValue("session/running", True)

    # Single unified GUI style
    gui = VascularizeGUI(domain=domain)
    if icon_path.is_file():
        gui.setWindowIcon(icon)

    gui.show()

    if not block:
        return app, gui

    try:
        app.exec()
    except AttributeError:
        app.exec_()
    return None


__all__ = ['VascularizeGUI', 'launch_gui', 'VascularizeCADGUI']
