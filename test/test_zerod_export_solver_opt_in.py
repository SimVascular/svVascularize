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
