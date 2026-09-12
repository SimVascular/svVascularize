import types

import numpy as np
import pyvista as pv
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox

import svv.simulation.simulation as simulation_mod
import svv.visualize.gui.main_window as main_window_mod
from svv.forest.forest import Forest
from svv.tree.tree import Tree


@pytest.fixture
def mesh_dialog(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("SVV_GUI_DISABLE_VTK", "1")
    monkeypatch.setenv("SVV_TELEMETRY_DISABLED", "1")
    monkeypatch.setattr(
        main_window_mod, "QSettings",
        lambda *args: QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat),
    )
    app = QApplication.instance() or QApplication([])
    gui = main_window_mod.VascularizeGUI()
    state = types.SimpleNamespace(events=[], errors=[], statuses=[], meshes=[], path=tmp_path / "tissue")
    state.obj = Tree()
    state.obj.domain = types.SimpleNamespace(boundary=object())

    class Simulation:
        def __init__(self, obj):
            self.tissue_domain_volume_meshes = state.meshes
            self.tissue_constraint_metadata = []

        def build_meshes(self, **kwargs):
            state.events.append("build")

    def choose_file(dialog):
        state.events.append("choose_file")
        dialog.selectFile(str(state.path))
        return QDialog.Accepted

    monkeypatch.setattr(simulation_mod, "Simulation", Simulation)
    monkeypatch.setattr(gui, "_require_synthetic_object", lambda: state.obj)
    monkeypatch.setattr(gui, "update_status", state.statuses.append)
    monkeypatch.setattr(gui, "_show_constrained_tissue_mesh_preview", lambda surfaces: None)
    monkeypatch.setattr(QDialog, "exec", lambda self: QDialog.Accepted)
    monkeypatch.setattr(QFileDialog, "exec", choose_file)
    monkeypatch.setattr(QMessageBox, "critical", lambda parent, title, message: state.errors.append(message))
    yield gui, state
    gui.close()
    app.processEvents()


def _mesh(offset):
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]) + offset
    mesh = pv.UnstructuredGrid({pv.CellType.TETRA: np.array([[0, 1, 2, 3]])}, points)
    mesh.point_data["constrained_point"] = np.array([1, 0, 0, 1], dtype=np.uint8)
    mesh.point_data["centerline_spline_id"] = np.array([4, -1, -1, 4])
    mesh.field_data["centerline_constraint_count"] = np.array([2])
    return mesh


@pytest.mark.parametrize("is_forest", [False, True])
def test_constrained_tissue_dialog_saves_all_meshes_and_constraint_data(mesh_dialog, is_forest):
    gui, state = mesh_dialog
    expected = [_mesh(0)]
    suffix = ".vtu"
    if is_forest:
        state.obj = Forest(domain=state.obj.domain)
        expected.append(_mesh(2))
        suffix = ".vtm"
    state.meshes = [[mesh] for mesh in expected]

    gui.build_constrained_tissue_mesh_dialog()

    assert state.events == ["choose_file", "build"]
    assert not state.errors
    output = state.path.with_suffix(suffix)
    loaded = pv.read(output)
    actual = list(loaded) if is_forest else [loaded]
    assert len(actual) == len(expected)
    for saved, original in zip(actual, expected):
        np.testing.assert_array_equal(saved.points, original.points)
        np.testing.assert_array_equal(saved.cells, original.cells)
        for name in original.point_data:
            np.testing.assert_array_equal(saved.point_data[name], original.point_data[name])
        for name in original.field_data:
            np.testing.assert_array_equal(saved.field_data[name], original.field_data[name])
    assert str(output) in state.statuses[-1]


@pytest.mark.parametrize("dialog_type", [QDialog, QFileDialog])
def test_constrained_tissue_dialog_cancel_does_not_build_or_save(mesh_dialog, monkeypatch, dialog_type):
    gui, state = mesh_dialog
    monkeypatch.setattr(dialog_type, "exec", lambda self: QDialog.Rejected)

    gui.build_constrained_tissue_mesh_dialog()

    assert "build" not in state.events
    assert not list(state.path.parent.glob("tissue*"))
    assert not state.errors


def test_constrained_tissue_dialog_reports_write_failure(mesh_dialog):
    gui, state = mesh_dialog
    state.meshes = [_mesh(0)]
    state.path = state.path.parent / "missing_directory" / "tissue.vtu"

    gui.build_constrained_tissue_mesh_dialog()

    assert state.events == ["choose_file", "build"]
    assert len(state.errors) == 1
    assert "save" in state.errors[0].lower()
    assert "saved to" not in state.statuses[-1]
    assert gui._last_constrained_tissue_simulation is not None
