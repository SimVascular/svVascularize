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

    assert captured["switches"] == ["pq1.1/10.0YQ", "riJMQ"]
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


@pytest.mark.parametrize("scale", [1.0, 100.0])
def test_tetrahedralize_with_prescribed_points_preserves_surface_boundary(scale):
    try:
        tetgen_exe = constrained_mod.resolve_tetgen_exe()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    surface = pv.Box(bounds=(-scale, scale) * 3).triangulate().clean()
    prescribed_points = np.array([[0.125, 0.25, 0.375]]) * scale
    grid, nodes, elems, meta = constrained_mod.tetrahedralize_with_prescribed_points(
        surface, prescribed_points, tetgen_exe=tetgen_exe,
    )

    boundary = grid.extract_surface()
    assert boundary.n_points == surface.n_points
    surface_ids = np.array([surface.find_closest_point(point) for point in boundary.points])
    np.testing.assert_array_equal(boundary.points, surface.points[surface_ids])
    actual_faces = surface_ids[boundary.faces.reshape(-1, 4)[:, 1:]]
    expected_faces = surface.faces.reshape(-1, 4)[:, 1:]
    assert {tuple(sorted(face)) for face in actual_faces} == {
        tuple(sorted(face)) for face in expected_faces
    }
    np.testing.assert_array_equal(nodes[meta["node_ids"]], prescribed_points)
    assert np.isin(meta["node_ids"], elems).all()


@pytest.mark.parametrize("distribution", ["across_concavity", "insertion_flips"])
def test_tetrahedralize_with_prescribed_points_connects_points_in_nonconvex_domain(distribution):
    try:
        tetgen_exe = constrained_mod.resolve_tetgen_exe()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    # A U-shaped domain requires retrying point searches across its concavity.
    voxels = pv.ImageData(dimensions=(7, 7, 2), spacing=(0.1, 0.1, 0.1))
    centers = voxels.cell_centers().points
    keep = (centers[:, 0] < 0.1) | (centers[:, 0] > 0.5) | (centers[:, 1] < 0.1)
    domain = voxels.extract_cells(np.flatnonzero(keep))
    surface = domain.extract_surface().triangulate().clean()
    centers = domain.cell_centers().points
    if distribution == "insertion_flips":
        # These insertions delete a tetrahedron retained by the next point search.
        rng = np.random.default_rng(29)
        ids = rng.integers(0, len(centers), 1000)
        points = centers[ids] + rng.uniform(-0.045, 0.045, (1000, 3))
    else:
        rng = np.random.default_rng(32)
        points = (centers[:, None, :] + rng.uniform(-0.04, 0.04, (len(centers), 8, 3))).reshape(-1, 3)

    _, nodes, elems, meta = constrained_mod.tetrahedralize_with_prescribed_points(
        surface, points, tetgen_exe=tetgen_exe,
    )

    assert meta["retained_point_count"] == len(points)
    np.testing.assert_array_equal(nodes[meta["node_ids"]], points)
    assert np.isin(meta["node_ids"], elems).all()


def test_tetrahedralize_with_prescribed_points_recovers_unused_interior_node(monkeypatch):
    nodes = np.array([
        [0.0, 0.0, 0.0],  # Present in .node, but unused by every tetrahedron.
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.0, 0.5, -0.5],
        [0.0, 0.0, 0.5],
    ])
    monkeypatch.setattr(constrained_mod, "resolve_tetgen_exe", lambda tetgen_exe=None: "/tmp/tetgen")
    monkeypatch.setattr(constrained_mod, "run_tetgen", lambda *args: None)
    monkeypatch.setattr(constrained_mod, "read_node", lambda path: (nodes, {}))
    monkeypatch.setattr(constrained_mod, "read_ele", lambda path, index_map: np.array([[1, 2, 3, 4]]))

    grid, output, elems, meta = constrained_mod.tetrahedralize_with_prescribed_points(_surface(), nodes[:1])

    np.testing.assert_array_equal(output, nodes)
    assert elems.shape == (4, 4)
    assert (elems == 0).sum() == 4
    assert meta["node_ids"].tolist() == [0]
    assert grid.extract_surface().n_open_edges == 0
    x = output[elems]
    volumes = np.linalg.det(x[:, 1:] - x[:, :1]) / 6
    assert (volumes > 0).all()
    assert volumes.sum() == pytest.approx(1 / 6)


@pytest.mark.parametrize("location", ["interior", "face", "edge", "edge_midpoint", "multiple", "absent"])
@pytest.mark.parametrize("order", [1, 2])
def test_missing_constraints_are_inserted_conformingly(monkeypatch, location, order):
    if location.startswith("edge"):
        vertices = np.array([[0, 0, -1], [0, 0, 1], [1, 0, 0],
                             [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=float)
        original = np.array([[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 5], [0, 1, 5, 2]])
        points = np.array([[0.0, 0.0, 0.125]])
        expected_count = 8
        if location == "edge_midpoint":
            points = np.vstack([[[0.0, 0.0, 0.0]], points])
            expected_count = 12
    elif location == "face":
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
        original = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])
        points = np.array([[0.25, 0.25, 0.0]])
        expected_count = 6
    else:
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        original = np.array([[0, 1, 2, 3]])
        points = np.array([[0.125, 0.25, 0.375]])
        expected_count = 4
        if location == "multiple":
            points = np.vstack([points, [[0.375, 0.125, 0.125]]])
            expected_count = 7

    before = pv.UnstructuredGrid(np.column_stack([np.full(len(original), 4), original]).ravel(),
                                np.full(len(original), pv.CellType.TETRA), vertices)
    surface = before.extract_surface()
    # TetGen's six additional node slots use this edge order.
    edges = np.array([[2, 3], [0, 3], [0, 1], [1, 2], [1, 3], [0, 2]])
    native_elems = original.copy()
    if order == 2:
        edge_ids = {}
        extra = []
        rows = []
        for cell in original:
            row = []
            for edge in cell[edges]:
                key = tuple(sorted(edge))
                if key not in edge_ids:
                    edge_ids[key] = len(vertices) + len(extra)
                    extra.append(vertices[edge].mean(axis=0))
                row.append(edge_ids[key])
            rows.append([*cell, *row])
        native_elems = np.asarray(rows)
        vertices = np.vstack([vertices, extra])
    nodes = vertices if location == "absent" else np.vstack([vertices, points])
    monkeypatch.setattr(constrained_mod, "resolve_tetgen_exe", lambda tetgen_exe=None: "/tmp/tetgen")
    monkeypatch.setattr(constrained_mod, "run_tetgen", lambda *args: None)
    monkeypatch.setattr(constrained_mod, "read_node", lambda path: (nodes, {}))
    monkeypatch.setattr(constrained_mod, "read_ele", lambda path, index_map: native_elems.copy())

    _, output, elems, meta = constrained_mod.tetrahedralize_with_prescribed_points(surface, points, order=order)

    np.testing.assert_array_equal(output[:len(nodes)], nodes)
    np.testing.assert_array_equal(output[meta["node_ids"]], points)
    assert np.isin(meta["node_ids"], elems[:, :4]).all()
    assert len(elems) == expected_count
    after = pv.UnstructuredGrid(np.column_stack([np.full(len(elems), 4), elems[:, :4]]).ravel(),
                               np.full(len(elems), pv.CellType.TETRA), output)
    assert after.extract_surface().n_open_edges == 0
    x = output[elems[:, :4]]
    volumes = np.linalg.det(x[:, 1:] - x[:, :1]) / 6
    assert (volumes > 0).all()
    assert volumes.sum() == pytest.approx(before.volume, rel=1e-12)
    def boundary_faces(mesh):
        boundary = mesh.extract_surface()
        return {tuple(sorted(map(tuple, boundary.points[face])))
                for face in boundary.faces.reshape(-1, 4)[:, 1:]}
    assert boundary_faces(after) == boundary_faces(before)
    if order == 2:
        np.testing.assert_array_equal(output[elems[:, 4:]], output[elems[:, :4][:, edges]].mean(axis=2))
        shared = {}
        for cell in elems:
            for edge, midpoint in zip(cell[:4][edges], cell[4:]):
                key = tuple(sorted(edge))
                assert shared.setdefault(key, midpoint) == midpoint


def test_unused_point_outside_output_volume_is_still_rejected(monkeypatch):
    nodes = np.array([[0.0, 0.0, 0.75], [-0.5, -0.5, -0.5],
                      [0.5, -0.5, -0.5], [0.0, 0.5, -0.5], [0.0, 0.0, 0.5]])
    monkeypatch.setattr(constrained_mod, "resolve_tetgen_exe", lambda tetgen_exe=None: "/tmp/tetgen")
    monkeypatch.setattr(constrained_mod, "run_tetgen", lambda *args: None)
    monkeypatch.setattr(constrained_mod, "read_node", lambda path: (nodes, {}))
    monkeypatch.setattr(constrained_mod, "read_ele", lambda path, index_map: np.array([[1, 2, 3, 4]]))
    with pytest.raises(RuntimeError, match="does not contain all prescribed spline points"):
        constrained_mod.tetrahedralize_with_prescribed_points(_surface(), nodes[:1])


@pytest.mark.parametrize("point", [
    [[0.1, 0.2, 0.7]], [[0.1, 0.1, 0.8]],
    [[0.1, 0.2, 0.7 + 1e-13]], [[0.1, 0.2, 0.7 + 1e-12]],
])
def test_point_recovery_checks_other_cells_when_locator_rounds_across_face(point):
    from svv.domain.routines.tetrahedral_repair import recover_prescribed_points

    nodes = np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.], [0, 0, 0], [1., 1., 1.]])
    elems = np.array([[0, 1, 2, 4], [0, 2, 1, 3]])
    # The exact sum of these stored coordinates is slightly less than one.
    # A tolerant locator can select the upper cell across x + y + z = 1.
    point = np.asarray(point)
    output, repaired = recover_prescribed_points(nodes, elems, point, 1e-6)
    assert len(repaired) == 6
    np.testing.assert_array_equal(output[-1:], point)
    assert np.isin(len(nodes), repaired)
    x = output[repaired]
    assert (np.linalg.det(x[:, 1:] - x[:, :1]) > 0).all()


def test_point_recovery_preserves_existing_within_tolerance_assignments():
    from svv.domain.routines.tetrahedral_repair import recover_prescribed_points

    nodes = np.array([[0., 0, 0], [1., 0, 0], [0, 1., 0], [0, 0, 1.]])
    elems = np.array([[0, 1, 2, 3]])
    output, repaired = recover_prescribed_points(nodes, elems, np.array([[1e-7, 1e-7, 1e-7]]), 1e-6)
    np.testing.assert_array_equal(output, nodes)
    np.testing.assert_array_equal(repaired, elems)


def test_point_recovery_does_not_accept_numerically_singular_children():
    from svv.domain.routines.tetrahedral_repair import recover_prescribed_points

    nodes = np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.], [1., 1., 1.]])
    elems = np.array([[0, 1, 2, 3]])
    output, repaired = recover_prescribed_points(nodes, elems, np.array([[0.1, 0.1, 0.8]]), 1e-6)
    np.testing.assert_array_equal(output, nodes)
    np.testing.assert_array_equal(repaired, elems)


@pytest.mark.parametrize("points", [
    [[0.125, 0.25, -1.0 + 1e-14], [0.0, 0.0, 0.0]],
    [[0.0, 0.0, -1.0]],  # Shared edge of two coplanar triangles.
    [[1.0, 0.0, -1.0]],  # Edge between different surface facets.
    [[1.0, 1.0, -1.0]],  # Existing surface vertex.
    [[0.125, 0.25, -1.0], [0.2, 0.4, -1.0], [0.125, 0.25, -1.0]],
])
def test_tetrahedralize_connects_boundary_constraints_without_moving_them(points):
    try:
        tetgen_exe = constrained_mod.resolve_tetgen_exe()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    surface = pv.Box().triangulate().clean()
    points = np.asarray(points, dtype=float)
    grid, nodes, elems, meta = constrained_mod.tetrahedralize_with_prescribed_points(
        surface, points, tetgen_exe=tetgen_exe,
    )

    assert meta["retained_point_count"] == len(points)
    np.testing.assert_array_equal(nodes[meta["node_ids"]], points)
    assert np.isin(meta["node_ids"], elems).all()
    boundary = grid.extract_surface()
    assert boundary.n_open_edges == 0
    boundary_points = points[np.isclose(points[:, 2], -1.0)]
    constrained_mod.verify_prescribed_points(boundary.points, boundary_points, 1e-12)
    x = nodes[elems]
    volumes = np.linalg.det(x[:, 1:] - x[:, :1]) / 6.0
    assert (volumes > 0).all()
    assert np.isclose(volumes.sum(), 8.0, rtol=1e-12, atol=1e-12)
