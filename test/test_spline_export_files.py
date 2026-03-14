from types import SimpleNamespace

import numpy as np
import pytest

from svv.forest.export.export_spline import write_splines as write_forest_splines
from svv.visualize.spline_export import export_spline_files


def _make_tree(x_offset: float, y_offset: float = 0.0, radius: float = 0.1):
    data = np.full((2, 31), np.nan)

    data[0, 0:3] = [x_offset + 0.0, y_offset, 0.0]
    data[0, 3:6] = [x_offset + 1.0, y_offset, 0.0]
    data[1, 0:3] = [x_offset + 1.0, y_offset, 0.0]
    data[1, 3:6] = [x_offset + 2.0, y_offset, 0.0]

    data[:, 12:15] = [1.0, 0.0, 0.0]
    data[:, 20] = 1.0
    data[:, 21] = radius

    data[0, 15] = 1
    data[1, 17] = 0
    data[0, 26] = 0
    data[1, 26] = 1

    return SimpleNamespace(data=data)


def _make_connected_tree_connection(network_id: int, x_offset: float, y_offset: float = 0.0):
    upstream = _make_tree(x_offset=x_offset, y_offset=y_offset, radius=0.1)
    downstream = _make_tree(x_offset=x_offset + 3.0, y_offset=y_offset, radius=0.1)

    vessels = [
        np.array([[[x_offset + 2.0, y_offset, 0.0, x_offset + 2.5, y_offset, 0.0, 0.1]]]),
        np.array([[[x_offset + 2.5, y_offset, 0.0, x_offset + 3.0, y_offset, 0.0, 0.1]]]),
    ]

    return SimpleNamespace(
        network_id=network_id,
        assignments=[np.array([1]), np.array([1])],
        vessels=vessels,
        connected_network=[upstream, downstream],
        forest=None,
    )


def _make_three_tree_connection(network_id: int, x_offset: float, y_offset: float = 0.0):
    tree_0 = _make_tree(x_offset=x_offset, y_offset=y_offset, radius=0.1)
    tree_1 = _make_tree(x_offset=x_offset + 3.0, y_offset=y_offset, radius=0.1)
    tree_2 = _make_tree(x_offset=x_offset + 6.0, y_offset=y_offset, radius=0.1)

    vessels = [
        np.array([[[x_offset + 2.0, y_offset, 0.0, x_offset + 2.5, y_offset, 0.0, 0.1]]]),
        np.array([[[x_offset + 2.5, y_offset, 0.0, x_offset + 3.0, y_offset, 0.0, 0.1]]]),
        np.array([[[x_offset + 5.5, y_offset, 0.0, x_offset + 6.0, y_offset, 0.0, 0.1]]]),
    ]

    return SimpleNamespace(
        network_id=network_id,
        assignments=[np.array([1]), np.array([1]), np.array([1])],
        vessels=vessels,
        connected_network=[tree_0, tree_1, tree_2],
        forest=None,
    )


def _make_unconnected_forest():
    return SimpleNamespace(
        networks=[
            [_make_tree(0.0, 0.0), _make_tree(10.0, 0.0)],
            [_make_tree(0.0, 10.0), _make_tree(10.0, 10.0)],
        ],
        connections=None,
    )


def _make_connected_forest():
    tree_connections = [
        _make_connected_tree_connection(network_id=0, x_offset=0.0, y_offset=0.0),
        _make_connected_tree_connection(network_id=1, x_offset=0.0, y_offset=5.0),
    ]
    forest = SimpleNamespace(networks=[[], []], connections=SimpleNamespace(tree_connections=tree_connections))
    for tree_connection in tree_connections:
        tree_connection.forest = forest
    return forest


def _make_three_tree_connected_forest():
    tree_connection = _make_three_tree_connection(network_id=0, x_offset=0.0, y_offset=0.0)
    forest = SimpleNamespace(
        networks=[[]],
        n_trees_per_network=[3],
        connections=SimpleNamespace(tree_connections=[tree_connection]),
    )
    tree_connection.forest = forest
    return forest


def _data_lines(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line and not line.startswith("Vessel:")]


def _read_sidecar(path):
    sections = {"inlet": [], "outlet": []}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line in sections:
            current = line
            continue
        if line and current is not None:
            sections[current].append(tuple(float(value) for value in line.split(", ")))
    return sections


def test_tree_export_writes_requested_txt_file(tmp_path):
    tree = _make_tree(0.0, 0.0)

    written = export_spline_files(tree, tmp_path / "tree_export.txt", spline_sample_points=7)

    assert written == [tmp_path / "tree_export.txt"]
    assert written[0].is_file()
    assert sorted(path.name for path in tmp_path.glob("*.txt")) == ["tree_export.txt"]

    first_row = _data_lines(written[0])[0].split(", ")
    assert len(first_row) == 4


def test_unconnected_forest_export_writes_one_txt_per_tree(tmp_path):
    forest = _make_unconnected_forest()

    written = export_spline_files(forest, tmp_path / "forest_export.txt", spline_sample_points=5)

    expected_names = [
        "forest_export_network0_tree0.txt",
        "forest_export_network0_tree1.txt",
        "forest_export_network1_tree0.txt",
        "forest_export_network1_tree1.txt",
    ]
    assert [path.name for path in written] == expected_names
    assert not (tmp_path / "forest_export.txt").exists()
    assert sorted(path.name for path in tmp_path.glob("*.txt")) == expected_names
    assert all(path.read_text(encoding="utf-8").startswith("Vessel: 0, Number of Points: 5") for path in written)


def test_connected_forest_export_writes_one_txt_per_connected_network(tmp_path):
    forest = _make_connected_forest()

    written = export_spline_files(
        forest,
        tmp_path / "connected_export.txt",
        spline_sample_points=6,
        separate=True,
    )

    expected_names = ["connected_export_network0.txt", "connected_export_network1.txt"]
    assert [path.name for path in written] == expected_names
    assert not (tmp_path / "connected_export.txt").exists()
    assert sorted(path.name for path in tmp_path.glob("*.txt")) == expected_names

    first_row = _data_lines(written[0])[0].split(", ")
    assert len(first_row) == 5
    assert first_row[-1] in {"0", "1"}


def test_tree_export_writes_inlet_root_sidecar(tmp_path):
    tree = _make_tree(0.0, 0.0)

    written = export_spline_files(
        tree,
        tmp_path / "tree_export.txt",
        spline_sample_points=7,
        export_inlet_outlet_roots=True,
        tree_root_role="inlet",
    )

    assert written == [tmp_path / "tree_export.txt"]
    sidecar = tmp_path / "tree_export_inlet_outlet.txt"
    assert sidecar.is_file()
    assert _read_sidecar(sidecar) == {"inlet": [(0.0, 0.0, 0.0)], "outlet": []}


def test_tree_export_writes_outlet_root_sidecar(tmp_path):
    tree = _make_tree(4.0, 1.5)

    written = export_spline_files(
        tree,
        tmp_path / "tree_export.txt",
        spline_sample_points=7,
        export_inlet_outlet_roots=True,
        tree_root_role="outlet",
    )

    assert written == [tmp_path / "tree_export.txt"]
    sidecar = tmp_path / "tree_export_inlet_outlet.txt"
    assert sidecar.is_file()
    assert _read_sidecar(sidecar) == {"inlet": [], "outlet": [(4.0, 1.5, 0.0)]}


def test_tree_export_requires_root_role_for_sidecar(tmp_path):
    tree = _make_tree(0.0, 0.0)

    with pytest.raises(ValueError, match="tree_root_role"):
        export_spline_files(
            tree,
            tmp_path / "tree_export.txt",
            spline_sample_points=7,
            export_inlet_outlet_roots=True,
        )


def test_unconnected_forest_export_writes_one_sidecar_per_tree_file(tmp_path):
    forest = _make_unconnected_forest()

    written = export_spline_files(
        forest,
        tmp_path / "forest_export.txt",
        spline_sample_points=5,
        export_inlet_outlet_roots=True,
        inlet_tree_indices_by_network={0: {0}, 1: {1}},
    )

    expected = {
        "forest_export_network0_tree0_inlet_outlet.txt": {"inlet": [(0.0, 0.0, 0.0)], "outlet": []},
        "forest_export_network0_tree1_inlet_outlet.txt": {"inlet": [], "outlet": [(10.0, 0.0, 0.0)]},
        "forest_export_network1_tree0_inlet_outlet.txt": {"inlet": [], "outlet": [(0.0, 10.0, 0.0)]},
        "forest_export_network1_tree1_inlet_outlet.txt": {"inlet": [(10.0, 10.0, 0.0)], "outlet": []},
    }
    assert len(written) == 4
    for sidecar_name, expected_sections in expected.items():
        sidecar = tmp_path / sidecar_name
        assert sidecar.is_file()
        assert _read_sidecar(sidecar) == expected_sections


def test_connected_two_tree_forest_sidecar_contains_both_root_points(tmp_path):
    forest = _make_connected_forest()

    written = export_spline_files(
        forest,
        tmp_path / "connected_export.txt",
        spline_sample_points=6,
        export_inlet_outlet_roots=True,
        inlet_tree_indices_by_network={0: {0}, 1: {1}},
    )

    assert [path.name for path in written] == ["connected_export_network0.txt", "connected_export_network1.txt"]
    assert _read_sidecar(tmp_path / "connected_export_network0_inlet_outlet.txt") == {
        "inlet": [(0.0, 0.0, 0.0)],
        "outlet": [(3.0, 0.0, 0.0)],
    }
    assert _read_sidecar(tmp_path / "connected_export_network1_inlet_outlet.txt") == {
        "inlet": [(3.0, 5.0, 0.0)],
        "outlet": [(0.0, 5.0, 0.0)],
    }


def test_connected_three_tree_forest_sidecars_follow_component_tree_mapping(tmp_path):
    forest = _make_three_tree_connected_forest()

    written = export_spline_files(
        forest,
        tmp_path / "three_tree.txt",
        spline_sample_points=6,
        export_inlet_outlet_roots=True,
        inlet_tree_indices_by_network={0: {0, 2}},
    )

    assert [path.name for path in written] == ["three_tree_network0_0.txt", "three_tree_network0_1.txt"]
    assert _read_sidecar(tmp_path / "three_tree_network0_0_inlet_outlet.txt") == {
        "inlet": [(0.0, 0.0, 0.0)],
        "outlet": [(3.0, 0.0, 0.0)],
    }
    assert _read_sidecar(tmp_path / "three_tree_network0_1_inlet_outlet.txt") == {
        "inlet": [(6.0, 0.0, 0.0)],
        "outlet": [],
    }


def test_export_spline_files_rejects_invalid_inlet_tree_index(tmp_path):
    forest = _make_unconnected_forest()

    with pytest.raises(ValueError, match="Invalid inlet tree index"):
        export_spline_files(
            forest,
            tmp_path / "forest_export.txt",
            spline_sample_points=5,
            export_inlet_outlet_roots=True,
            inlet_tree_indices_by_network={0: {9}},
        )


def test_forest_spline_writer_returns_all_network_splines_and_files(tmp_path):
    all_points = [
        [
            [[0.0, 0.0, 0.0], [0.33, 0.0, 0.0], [0.66, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        [
            [[0.0, 1.0, 0.0], [0.33, 1.0, 0.0], [0.66, 1.0, 0.0], [1.0, 1.0, 0.0]],
        ],
    ]
    all_radii = [
        [[0.2, 0.2, 0.2, 0.2]],
        [[0.3, 0.3, 0.3, 0.3]],
    ]

    spline_functions = write_forest_splines(
        all_points,
        all_radii,
        spline_sample_points=4,
        outdir=str(tmp_path),
        write_splines=True,
    )

    assert len(spline_functions) == 2
    assert all(len(network_splines) == 1 for network_splines in spline_functions)
    assert sorted(path.name for path in tmp_path.glob("*.txt")) == [
        "network_0_b_splines.txt",
        "network_1_b_splines.txt",
    ]
