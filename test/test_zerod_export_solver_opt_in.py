import json
from types import SimpleNamespace

import numpy as np

from svv.simulation.fluid.rom.zero_d import zerod_forest, zerod_tree


def _base_tree_data() -> np.ndarray:
    data = np.zeros((1, 23), dtype=float)
    data[0, 0:3] = [0.0, 0.0, 0.0]
    data[0, 3:6] = [1.0, 0.0, 0.0]
    data[0, 15] = np.nan
    data[0, 16] = -1.0
    data[0, 20] = 1.0
    data[0, 21] = 0.1
    data[0, 22] = 1.0
    return data


def _raise_if_called():
    raise AssertionError("get_solver_0d_exe() should not be called")


def _load_inflow_bc(case_dir):
    data = json.loads((case_dir / "solver_0d.in").read_text(encoding="utf-8"))
    return next(bc for bc in data["boundary_conditions"] if bc["bc_name"] == "INFLOW")


def test_tree_0d_export_does_not_probe_solver_when_not_requested(monkeypatch, tmp_path):
    tree = SimpleNamespace(
        data=_base_tree_data(),
        parameters=SimpleNamespace(terminal_pressure=100.0, kinematic_viscosity=0.04),
    )

    monkeypatch.setattr(zerod_tree, "get_solver_0d_exe", _raise_if_called)

    zerod_tree.export_0d_simulation(
        tree,
        outdir=str(tmp_path),
        folder="case",
        get_0d_solver=False,
    )

    outdir = tmp_path / "case"
    assert (outdir / "solver_0d.in").is_file()
    assert (outdir / "run.py").is_file()
    assert not (outdir / "cmd_run.bash").exists()
    run_text = (outdir / "run.py").read_text(encoding="utf-8")
    plot_text = (outdir / "plot_0d_results_to_3d.py").read_text(encoding="utf-8")
    assert "CHILD_PID_FILE" in run_text
    assert "signal.signal" in run_text
    assert "OUTPUT_FILENAME" in run_text
    assert "SVV_0D_DISABLE_TQDM" in plot_text


def test_tree_0d_export_command_file_runs_from_export_dir(monkeypatch, tmp_path):
    tree = SimpleNamespace(
        data=_base_tree_data(),
        parameters=SimpleNamespace(terminal_pressure=100.0, kinematic_viscosity=0.04),
    )
    solver = tmp_path / "svzerodsolver.exe"
    solver.write_text("", encoding="utf-8")
    monkeypatch.setattr(zerod_tree.platform, "system", lambda: "Windows")

    zerod_tree.export_0d_simulation(
        tree,
        outdir=str(tmp_path),
        folder="case",
        path_to_0d_solver=str(solver),
    )

    cmd_text = (tmp_path / "case" / "cmd_run.bat").read_text(encoding="utf-8")
    assert 'cd /d "%~dp0"' in cmd_text
    assert '"solver_0d.in" "output.csv"' in cmd_text


def test_forest_0d_export_does_not_probe_solver_when_not_requested(monkeypatch, tmp_path):
    tree = SimpleNamespace(
        data=_base_tree_data(),
        parameters=SimpleNamespace(kinematic_viscosity=0.04, fluid_density=1.06),
    )
    connection = SimpleNamespace(connected_network=[tree], vessels=[[]])
    forest = SimpleNamespace(
        n_trees_per_network=[1],
        networks=[[tree]],
        connections=SimpleNamespace(tree_connections=[connection]),
    )

    monkeypatch.setattr(zerod_forest, "get_solver_0d_exe", _raise_if_called)

    zerod_forest.export_0d_simulation(
        forest,
        network_id=0,
        inlets=[0],
        outdir=str(tmp_path),
        folder="case",
        get_0d_solver=False,
    )

    outdir = tmp_path / "case"
    assert (outdir / "solver_0d.in").is_file()
    assert (outdir / "run.py").is_file()
    assert not (outdir / "cmd_run.bash").exists()
    run_text = (outdir / "run.py").read_text(encoding="utf-8")
    assert "CHILD_PID_FILE" in run_text
    assert "OUTPUT_FILENAME" in run_text


def test_tree_0d_export_uses_waveform_for_unsteady_custom_flow(monkeypatch, tmp_path):
    tree = SimpleNamespace(
        data=_base_tree_data(),
        parameters=SimpleNamespace(terminal_pressure=100.0, kinematic_viscosity=0.04),
    )
    calls = []

    def _fake_waveform(flow_value, diameter, *args, **kwargs):
        calls.append((flow_value, diameter))
        return np.array([0.0, 0.5, 1.0]), np.array([2.0, 3.0, 4.0])

    monkeypatch.setattr(zerod_tree, "generate_physiologic_wave", _fake_waveform)

    zerod_tree.export_0d_simulation(
        tree,
        outdir=str(tmp_path),
        folder="case",
        steady=False,
        flow=2.5,
    )

    inflow = _load_inflow_bc(tmp_path / "case")

    assert calls == [(2.5, 0.2)]
    assert inflow["bc_values"]["t"] == [0.0, 0.5, 1.0]
    assert inflow["bc_values"]["Q"] == [2.0, 3.0, 2.0]
    assert (tmp_path / "case" / "inflow.flow").read_text(encoding="utf-8").splitlines() == [
        "0.0  2.0",
        "0.5  3.0",
        "1.0  2.0",
    ]


def test_forest_0d_export_uses_waveform_for_unsteady_custom_flow(monkeypatch, tmp_path):
    tree = SimpleNamespace(
        data=_base_tree_data(),
        parameters=SimpleNamespace(kinematic_viscosity=0.04, fluid_density=1.06),
    )
    connection = SimpleNamespace(connected_network=[tree], vessels=[[]])
    forest = SimpleNamespace(
        n_trees_per_network=[1],
        networks=[[tree]],
        connections=SimpleNamespace(tree_connections=[connection]),
    )
    calls = []

    def _fake_waveform(flow_value, diameter, *args, **kwargs):
        calls.append((flow_value, diameter))
        return np.array([0.0, 0.25, 0.5]), np.array([5.0, 6.0, 7.0])

    monkeypatch.setattr(zerod_forest, "generate_physiologic_wave", _fake_waveform)

    zerod_forest.export_0d_simulation(
        forest,
        network_id=0,
        inlets=[0],
        outdir=str(tmp_path),
        folder="case",
        steady=False,
        flow=3.5,
        get_0d_solver=False,
    )

    inflow = _load_inflow_bc(tmp_path / "case")

    assert calls == [(3.5, 0.2)]
    assert inflow["bc_values"]["t"] == [0.0, 0.25, 0.5]
    assert inflow["bc_values"]["Q"] == [5.0, 6.0, 5.0]
    assert (tmp_path / "case" / "inflow.flow").read_text(encoding="utf-8").splitlines() == [
        "0.0  5.0",
        "0.25  6.0",
        "0.5  5.0",
    ]
