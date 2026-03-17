import pytest
import sys

pytest.importorskip("PySide6")

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QApplication

import svv.visualize.gui.main_window as main_window_mod
from svv.visualize.gui.main_window import VascularizeGUI, _ZeroDPipelineStep


def _make_gui(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("SVV_GUI_DISABLE_VTK", "1")
    monkeypatch.setenv("SVV_TELEMETRY_DISABLED", "1")
    app = QApplication.instance() or QApplication([])
    gui = VascularizeGUI()
    return app, gui


def _close_gui(app, gui):
    gui.close()
    app.processEvents()


def test_build_0d_pipeline_steps_disable_gui_unfriendly_outputs(monkeypatch, tmp_path):
    app, gui = _make_gui(monkeypatch)

    steps = gui._build_0d_pipeline_steps(str(tmp_path))

    assert [step.log_prefix for step in steps] == ["solver", "plot", "collate"]
    assert steps[0].args[-1].endswith("run.py")
    assert steps[1].env["SVV_0D_RENDER_SCREENSHOTS"] == "0"
    assert steps[1].env["SVV_0D_DISABLE_TQDM"] == "1"

    _close_gui(app, gui)


def test_cancel_0d_pipeline_terminates_active_process(monkeypatch, tmp_path):
    app, gui = _make_gui(monkeypatch)

    class _FakeProcess:
        def __init__(self):
            self.terminate_calls = 0
            self.kill_calls = 0
            self._state = QProcess.Running

        def terminate(self):
            self.terminate_calls += 1

        def kill(self):
            self.kill_calls += 1

        def state(self):
            return self._state

    fake_process = _FakeProcess()
    gui._zerod_process = fake_process
    gui._zerod_pipeline_steps = [
        _ZeroDPipelineStep("Run 0D solver", "solver", "python", ["run.py"])
    ]
    gui._zerod_pipeline_step_index = 0
    gui._zerod_pipeline_export_path = str(tmp_path)
    monkeypatch.setattr(main_window_mod.QTimer, "singleShot", lambda _ms, _fn: None)

    gui._cancel_0d_pipeline()

    assert gui._zerod_pipeline_cancel_requested is True
    assert fake_process.terminate_calls == 1
    assert fake_process.kill_calls == 0

    gui._cleanup_0d_pipeline()
    _close_gui(app, gui)


def test_auto_open_0d_results_loads_generated_pvd(monkeypatch, tmp_path):
    app, gui = _make_gui(monkeypatch)
    pvd_path = tmp_path / "timeseries" / "timeseries.pvd"
    pvd_path.parent.mkdir(parents=True)
    pvd_path.write_text(
        "<?xml version='1.0' encoding='UTF-8'?><VTKFile type='Collection'></VTKFile>",
        encoding="utf-8",
    )

    shown = []
    opened = []
    monkeypatch.setattr(gui, "_show_solution_inspector", lambda: shown.append(True))
    monkeypatch.setattr(gui.solution_inspector, "open_solution", lambda path: opened.append(path) or True)

    opened_path = gui._auto_open_0d_results(str(tmp_path))

    assert opened_path == str(pvd_path)
    assert shown == [True]
    assert opened == [str(pvd_path)]

    _close_gui(app, gui)


def test_solution_inspector_open_solution_sets_path_and_calls_loader(monkeypatch):
    app, gui = _make_gui(monkeypatch)
    inspector = gui.solution_inspector
    loaded = []

    monkeypatch.setattr(inspector, "_load_solution", lambda path: loaded.append(path) or True)

    assert inspector.open_solution("/tmp/example.pvd") is True
    assert inspector.file_edit.text() == "/tmp/example.pvd"
    assert loaded == ["/tmp/example.pvd"]

    _close_gui(app, gui)


def test_solution_inspector_field_change_updates_auto_colorbar_label(monkeypatch):
    app, gui = _make_gui(monkeypatch)
    inspector = gui.solution_inspector
    render_calls = []

    monkeypatch.setattr(
        inspector,
        "_render_current_mesh",
        lambda *, reset_camera=False: render_calls.append(reset_camera),
    )

    inspector.scalar_label_edit.setText("pressure")
    inspector._scalar_label_auto_text = "pressure"

    inspector._on_scalar_changed("velocity")

    assert inspector.scalar_label_edit.text() == "velocity"
    assert render_calls == [False]

    _close_gui(app, gui)


def test_solution_inspector_field_change_keeps_custom_colorbar_label(monkeypatch):
    app, gui = _make_gui(monkeypatch)
    inspector = gui.solution_inspector
    render_calls = []

    monkeypatch.setattr(
        inspector,
        "_render_current_mesh",
        lambda *, reset_camera=False: render_calls.append(reset_camera),
    )

    inspector.scalar_label_edit.setText("Custom Label")
    inspector._scalar_label_auto_text = "pressure"

    inspector._on_scalar_changed("velocity")

    assert inspector.scalar_label_edit.text() == "Custom Label"
    assert render_calls == [False]

    _close_gui(app, gui)


def test_solution_inspector_scalar_label_edit_rerenders_live(monkeypatch):
    app, gui = _make_gui(monkeypatch)
    inspector = gui.solution_inspector
    render_calls = []

    monkeypatch.setattr(
        inspector,
        "_render_current_mesh",
        lambda *, reset_camera=False: render_calls.append(reset_camera),
    )

    inspector._on_scalar_label_edited("Flow")

    assert render_calls == [False]

    _close_gui(app, gui)


def test_solution_inspector_plotter_ready_replays_pending_camera_reset(monkeypatch):
    app, gui = _make_gui(monkeypatch)
    inspector = gui.solution_inspector
    render_calls = []

    monkeypatch.setattr(
        inspector,
        "_render_current_mesh",
        lambda *, reset_camera=False: render_calls.append(reset_camera),
    )

    inspector._current_mesh = object()
    inspector._pending_camera_reset = True

    inspector._on_vtk_plotter_ready()

    assert render_calls == [True]
    assert inspector._pending_camera_reset is False

    _close_gui(app, gui)


def test_solution_inspector_time_change_uses_pvd_time_point_api(monkeypatch):
    app, gui = _make_gui(monkeypatch)
    inspector = gui.solution_inspector

    class _FakeMesh:
        def __init__(self, idx):
            self.idx = idx

    class _FakeReader:
        def __init__(self):
            self.active_idx = 0
            self.time_values = [0.0, 0.1, 0.2]
            self.calls = []

        def set_active_time_point(self, idx):
            self.calls.append(idx)
            self.active_idx = idx

        def read(self):
            return _FakeMesh(self.active_idx)

    class _FakePV:
        class MultiBlock:
            pass

    renders = []
    monkeypatch.setattr(inspector, "_update_scalar_range", lambda: None)
    monkeypatch.setattr(inspector, "_render_current_mesh", lambda *, reset_camera=False: renders.append((inspector._current_mesh.idx, reset_camera)))
    monkeypatch.setattr(inspector, "_update_calculator_mesh", lambda: None)

    inspector._pv = _FakePV()
    inspector._reader = _FakeReader()
    inspector._time_values = [0.0, 0.1, 0.2]

    inspector._on_time_index_changed(2)

    assert inspector._reader.calls == [2]
    assert inspector._current_time_index == 2
    assert inspector._current_mesh.idx == 2
    assert renders == [(2, False)]

    _close_gui(app, gui)


def test_solution_inspector_load_solution_enables_global_range_for_time_series(monkeypatch, tmp_path):
    app, gui = _make_gui(monkeypatch)
    inspector = gui.solution_inspector

    class _FakeReader:
        def __init__(self):
            self.time_values = [0.0, 0.1, 0.2]

        def set_active_time_point(self, _idx):
            pass

        def read(self):
            return object()

    class _FakePV:
        class MultiBlock:
            pass

        @staticmethod
        def get_reader(_path):
            return _FakeReader()

    solution_path = tmp_path / "timeseries.pvd"
    solution_path.write_text("<VTKFile/>", encoding="utf-8")

    monkeypatch.setattr(inspector, "_update_mesh_from_reader", lambda *args, **kwargs: None)
    monkeypatch.setattr(inspector, "_populate_scalar_fields", lambda: None)
    monkeypatch.setattr(inspector, "_populate_vector_fields", lambda: None)
    monkeypatch.setattr(inspector, "_update_statistics", lambda: None)

    inspector._pv = _FakePV()
    inspector.auto_range_cb.setChecked(True)
    inspector.global_range_cb.setChecked(False)

    assert inspector._load_solution(str(solution_path)) is True
    assert inspector.global_range_cb.isChecked() is True

    _close_gui(app, gui)


def test_solution_inspector_manual_scalar_range_persists_across_time_changes(monkeypatch):
    app, gui = _make_gui(monkeypatch)
    inspector = gui.solution_inspector

    class _FakeReader:
        def __init__(self):
            self.active_idx = 0

        def set_active_time_point(self, idx):
            self.active_idx = idx

        def read(self):
            return object()

    class _FakePV:
        class MultiBlock:
            pass

    monkeypatch.setattr(inspector, "_compute_local_range", lambda _name: (0.0, 100.0))
    monkeypatch.setattr(inspector, "_render_current_mesh", lambda *, reset_camera=False: None)
    monkeypatch.setattr(inspector, "_update_calculator_mesh", lambda: None)

    inspector._pv = _FakePV()
    inspector._reader = _FakeReader()
    inspector._time_values = [0.0, 0.1, 0.2]
    inspector._current_scalar_name = "Pressure [mmHg]"
    inspector._current_mesh = object()
    inspector.global_range_cb.setChecked(False)
    inspector.auto_range_cb.setChecked(False)
    inspector.scalar_min_spin.setValue(1.2)
    inspector.scalar_max_spin.setValue(9.8)

    inspector._on_time_index_changed(2)

    assert inspector._current_time_index == 2
    assert inspector.scalar_min_spin.value() == 1.2
    assert inspector.scalar_max_spin.value() == 9.8

    _close_gui(app, gui)


def test_time_series_dialog_derivative_accepts_colored_plot_tuples(monkeypatch):
    app, gui = _make_gui(monkeypatch)
    module = sys.modules[type(gui.solution_inspector).__module__]
    dialog = module.TimeSeriesPlotDialog(gui)

    class _FakeLine:
        pass

    class _FakeAxes:
        def plot(self, *_args, **_kwargs):
            return [_FakeLine()]

        def legend(self, *_args, **_kwargs):
            return None

    class _FakeFigure:
        def tight_layout(self):
            return None

    class _FakeCanvas:
        def draw(self):
            return None

    monkeypatch.setattr(dialog, "_clear_derivative", lambda: None)
    dialog._mpl_available = True
    dialog._ax = _FakeAxes()
    dialog._fig = _FakeFigure()
    dialog._canvas = _FakeCanvas()
    dialog.deriv_separate_cb.setChecked(False)
    dialog._plot_data = [([0.0, 1.0, 2.0], [1.0, 3.0, 5.0], "Series A", "blue")]

    dialog._compute_derivative()

    assert dialog._derivative_line is not None

    dialog.close()
    _close_gui(app, gui)
