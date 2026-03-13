import pytest

pytest.importorskip("PySide6")
pv = pytest.importorskip("pyvista")

import numpy as np

from PySide6.QtWidgets import QApplication

import svv.visualize.gui.vtk_widget as vtk_widget_mod
from svv.visualize.gui.vtk_widget import VTKWidget


def test_vtk_widget_disables_vtk_for_offscreen_qt(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.delenv("SVV_GUI_DISABLE_VTK", raising=False)

    app = QApplication.instance() or QApplication([])
    widget = VTKWidget()

    widget._ensure_plotter_initialized()

    assert widget.plotter is None
    assert widget._plotter_init_failed is True
    assert widget._plotter_init_done is False
    assert widget._vtk_placeholder is not None
    assert "offscreen" in widget._vtk_placeholder.text().lower()

    widget.deleteLater()
    app.processEvents()


def test_vtk_widget_prefers_offscreen_for_macos_arm64_conda(monkeypatch):
    monkeypatch.setattr(vtk_widget_mod.sys, "platform", "darwin")
    monkeypatch.setattr(vtk_widget_mod.platform, "machine", lambda: "arm64")
    monkeypatch.setenv("CONDA_PREFIX", "/tmp/svv-conda")
    monkeypatch.delenv("SVV_GUI_FORCE_OFFSCREEN_VTK", raising=False)
    monkeypatch.delenv("SVV_GUI_FORCE_EMBEDDED_VTK", raising=False)

    assert VTKWidget._should_use_offscreen() is True


def test_vtk_widget_force_offscreen_env(monkeypatch):
    monkeypatch.setenv("SVV_GUI_FORCE_OFFSCREEN_VTK", "1")
    monkeypatch.delenv("SVV_GUI_FORCE_EMBEDDED_VTK", raising=False)

    assert VTKWidget._should_use_offscreen() is True


def test_vtk_widget_force_embedded_overrides_offscreen(monkeypatch):
    monkeypatch.setenv("SVV_GUI_FORCE_OFFSCREEN_VTK", "1")
    monkeypatch.setenv("SVV_GUI_FORCE_EMBEDDED_VTK", "1")

    assert VTKWidget._should_use_offscreen() is False


def test_build_vessel_tube_mesh_batches_segments():
    proximal = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    distal = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    radii = np.array([0.1, 0.2])

    mesh = VTKWidget._build_vessel_tube_mesh(pv, proximal, distal, radii, n_sides=8)

    assert mesh is not None
    assert mesh.n_points > 0
    assert mesh.n_cells > 0
