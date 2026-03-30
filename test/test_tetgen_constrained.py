import numpy as np
import pyvista as pv
import pytest

from svv.domain.routines import tetgen_constrained as constrained_mod


def _surface():
    return pv.Sphere(radius=1.0, theta_resolution=24, phi_resolution=24).triangulate()


def test_filter_prescribed_points_to_surface_drops_outside_points_and_remaps_lines():
    surface = _surface()
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 0.0, 1.25],
        ],
        dtype=float,
    )
    lines = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    point_metadata = {
        "spline_id": np.array([4, 4, 4, 9], dtype=np.int32),
        "spline_order": np.array([0, 1, 2, 0], dtype=np.int32),
        "radius": np.array([0.1, 0.2, 0.3, 0.4], dtype=float),
    }

    filtered = constrained_mod.filter_prescribed_points_to_surface(
        surface,
        points,
        lines,
        point_metadata=point_metadata,
        verify_tol=1e-6,
    )

    assert np.allclose(filtered["points"], points[[0, 1]])
    assert filtered["lines"].tolist() == [[0, 1]]
    assert filtered["kept_point_ids"].tolist() == [0, 1]
    assert filtered["dropped_point_ids"].tolist() == [2, 3]
    assert filtered["point_metadata"]["spline_id"].tolist() == [4, 4]
    assert filtered["point_metadata"]["spline_order"].tolist() == [0, 1]
    assert np.allclose(filtered["point_metadata"]["radius"], [0.1, 0.2])


def test_verify_prescribed_points_requires_exact_coordinate_matches():
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    prescribed_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.99, 0.0, 0.0],
        ],
        dtype=float,
    )

    with pytest.raises(RuntimeError, match="does not contain all prescribed spline points"):
        constrained_mod.verify_prescribed_points(nodes, prescribed_points, tol=1e-6)


def test_tetrahedralize_with_prescribed_points_uses_exact_insertion_switches(monkeypatch):
    surface = _surface()
    prescribed_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        ],
        dtype=float,
    )
    prescribed_lines = np.array([[0, 1], [1, 2]], dtype=np.int64)
    point_metadata = {
        "spline_id": np.array([3, 3, 3], dtype=np.int32),
        "spline_order": np.array([0, 1, 2], dtype=np.int32),
        "radius": np.array([0.1, 0.1, 0.1], dtype=float),
    }
    captured = {"switches": []}

    monkeypatch.setattr(constrained_mod, "resolve_tetgen_exe", lambda tetgen_exe=None: "/tmp/tetgen")
    monkeypatch.setattr(constrained_mod, "write_poly", lambda surface_mesh, path: None)

    def fake_run_tetgen(exe, switches, stem, cwd):
        captured["switches"].append(switches)

    def fake_write_a_node(points, path):
        captured["written_points"] = np.asarray(points, dtype=float).copy()

    def fake_read_node(path):
        nodes = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        return nodes, {1: 0, 2: 1, 3: 2, 4: 3}

    monkeypatch.setattr(constrained_mod, "run_tetgen", fake_run_tetgen)
    monkeypatch.setattr(constrained_mod, "write_a_node", fake_write_a_node)
    monkeypatch.setattr(constrained_mod, "read_node", fake_read_node)
    monkeypatch.setattr(constrained_mod, "read_ele", lambda path, index_map: np.array([[0, 1, 2, 3]], dtype=np.int64))

    grid, nodes, elems, meta = constrained_mod.tetrahedralize_with_prescribed_points(
        surface,
        prescribed_points,
        prescribed_lines=prescribed_lines,
        point_metadata=point_metadata,
        verify_tol=1e-6,
    )

    assert captured["switches"] == ["pq1.1/10.0Q", "riJMQ"]
    assert np.allclose(captured["written_points"], prescribed_points[:2])
    assert meta["line_count"] == 1
    assert meta["original_point_count"] == 3
    assert meta["retained_point_count"] == 2
    assert meta["dropped_point_count"] == 1
    assert meta["dropped_point_ids"] == [2]
    assert meta["unique_node_count"] == 2
    assert np.isclose(meta["max_assignment_distance"], 0.0)
    assert meta["insertion_switches"] == "riJMQ"
    assert int(grid.field_data["centerline_constraint_count"][0]) == 2
    assert int(grid.field_data["centerline_constraint_line_count"][0]) == 1
    assert int(grid.field_data["centerline_constraint_original_count"][0]) == 3
    assert int(grid.field_data["centerline_constraint_dropped_count"][0]) == 1
    assert int(grid.field_data["centerline_constraint_unique_node_count"][0]) == 2
    assert np.isclose(float(grid.field_data["centerline_constraint_assignment_max_distance"][0]), 0.0)
    assert int(grid.field_data["centerline_constraint_tolerance_exceeded_count"][0]) == 0
    assert np.array_equal(grid.point_data["constrained_point"], np.array([1, 1, 0, 0], dtype=np.uint8))
    assert np.array_equal(grid.point_data["centerline_node"], np.array([1, 1, 0, 0], dtype=np.uint8))
    assert np.array_equal(grid.point_data["centerline_spline_id"][:2], np.array([3, 3], dtype=np.int32))
    assert np.array_equal(grid.point_data["centerline_spline_order"][:2], np.array([0, 1], dtype=np.int32))
    assert np.allclose(grid.point_data["centerline_radius"][:2], [0.1, 0.1])
    assert nodes.shape == (4, 3)
    assert elems.shape == (1, 4)
