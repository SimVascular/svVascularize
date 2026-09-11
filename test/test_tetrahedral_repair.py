"""Regression neighborhoods from the 25-terminal pediatric-heart benchmark."""
from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

from svv.domain.routines import tetrahedral_repair as repair
from svv.domain.routines import tetgen_constrained as constrained

_QUADRATIC_EDGES = np.array([[2, 3], [0, 3], [0, 1], [1, 2], [1, 3], [0, 2]])


def _fixture(seed):
    with np.load(Path(__file__).parent / 'data' / 'tetrahedral_repair' / f'near_collinear_{seed}.npz') as data:
        return data['nodes'], data['elems'], data['constrained'].astype(bool)


def _grid(nodes, elems):
    return pv.UnstructuredGrid(np.column_stack([np.full(len(elems), 4), elems[:, :4]]).ravel(),
                               np.full(len(elems), pv.CellType.TETRA), nodes)


def _boundary_faces(elems):
    faces = elems[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]].reshape(-1, 3)
    faces, counts = np.unique(np.sort(faces, axis=1), return_counts=True, axis=0)
    assert counts.max() <= 2
    return {tuple(face) for face in faces[counts == 1]}


def _quadratic(nodes, elems):
    midpoint_ids, added, rows = {}, [], []
    for cell in elems:
        row = []
        for edge in cell[_QUADRATIC_EDGES]:
            key = tuple(sorted(edge))
            if key not in midpoint_ids:
                midpoint_ids[key] = len(nodes) + len(added)
                added.append(nodes[edge].mean(axis=0))
            row.append(midpoint_ids[key])
        rows.append([*cell, *row])
    return np.vstack([nodes, added]), np.asarray(rows)


@pytest.mark.parametrize('seed', [1385, 1883, 1536])
@pytest.mark.parametrize('scale', [0.001, 1.0, 1000.0])
def test_repair_removes_nearly_degenerate_cells_without_moving_nodes(seed, scale):
    nodes, elems, fixed = _fixture(seed)
    nodes *= scale
    original_nodes, original_elems = nodes.copy(), elems.copy()
    assert _grid(nodes, elems).cell_quality('scaled_jacobian')['scaled_jacobian'].min() < 1e-12

    output, repaired = repair.repair_degenerate_tetrahedra(nodes, elems)

    np.testing.assert_array_equal(nodes, original_nodes)
    np.testing.assert_array_equal(elems, original_elems)
    np.testing.assert_array_equal(output, original_nodes)
    np.testing.assert_array_equal(np.unique(repaired), np.unique(elems))
    assert np.isin(np.flatnonzero(fixed), repaired).all()
    assert _boundary_faces(repaired) == _boundary_faces(elems)
    grid = _grid(output, repaired)
    assert grid.extract_surface().n_open_edges == 0
    assert grid.cell_quality('scaled_jacobian')['scaled_jacobian'].min() >= 1e-12
    x = output[repaired]
    volumes = np.linalg.det(x[:, 1:] - x[:, :1]) / 6
    assert (volumes > 0).all()
    assert volumes.sum() == pytest.approx(_grid(nodes, elems).volume, rel=1e-12)


@pytest.mark.parametrize('seed', [1385, 1883, 1536])
def test_constrained_mesh_checks_and_repairs_quality_after_insertion(monkeypatch, seed):
    nodes, elems, fixed = _fixture(seed)
    surface = _grid(nodes, elems).extract_surface()
    monkeypatch.setattr(constrained, 'resolve_tetgen_exe', lambda tetgen_exe=None: '/tmp/tetgen')
    monkeypatch.setattr(constrained, 'run_tetgen', lambda *args: None)
    monkeypatch.setattr(constrained, 'read_node', lambda path: (nodes, {}))
    monkeypatch.setattr(constrained, 'read_ele', lambda path, index_map: elems)

    grid, output, repaired, meta = constrained.tetrahedralize_with_prescribed_points(surface, nodes[fixed])

    np.testing.assert_array_equal(output[meta['node_ids']], nodes[fixed])
    assert np.isin(meta['node_ids'], repaired).all()
    assert grid.cell_quality('scaled_jacobian')['scaled_jacobian'].min() >= 1e-12
    assert _boundary_faces(repaired) == _boundary_faces(elems)


def test_quality_check_leaves_healthy_mesh_unchanged():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    elems = np.array([[0, 1, 2, 3]])
    output, repaired = repair.repair_degenerate_tetrahedra(nodes, elems)
    np.testing.assert_array_equal(output, nodes)
    np.testing.assert_array_equal(repaired, elems)


def test_unrepairable_degenerate_mesh_is_rejected():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    with pytest.raises(RuntimeError, match='near-degenerate tetrahedra'):
        repair.repair_degenerate_tetrahedra(nodes, np.array([[0, 1, 2, 3]]))


def test_quality_repair_preserves_prescribed_quadratic_midpoints(monkeypatch):
    nodes, elems, fixed = _fixture(1385)
    # Move the middle sample along its nearly collinear edge, so the old
    # quadratic midpoint is distinct from it and must remain connected too.
    nodes[24] = 0.75 * nodes[23] + 0.25 * nodes[25]
    surface = _grid(nodes, elems).extract_surface()
    points = np.vstack([nodes[fixed], nodes[[23, 25]].mean(axis=0)])
    original_nodes, original_elems = _quadratic(nodes, elems)
    monkeypatch.setattr(constrained, 'resolve_tetgen_exe', lambda tetgen_exe=None: '/tmp/tetgen')
    monkeypatch.setattr(constrained, 'run_tetgen', lambda *args: None)
    monkeypatch.setattr(constrained, 'read_node', lambda path: (original_nodes, {}))
    monkeypatch.setattr(constrained, 'read_ele', lambda path, index_map: original_elems)

    _, output, repaired, meta = constrained.tetrahedralize_with_prescribed_points(surface, points, order=2)

    np.testing.assert_array_equal(output[:len(original_nodes)], original_nodes)
    np.testing.assert_array_equal(output[meta['node_ids']], points)
    assert np.isin(meta['node_ids'], repaired).all()
    assert _grid(output, repaired).cell_quality('scaled_jacobian')['scaled_jacobian'].min() >= 1e-12
    assert _boundary_faces(repaired[:, :4]) == _boundary_faces(elems)
    np.testing.assert_array_equal(output[repaired[:, 4:]], output[repaired[:, :4][:, _QUADRATIC_EDGES]].mean(axis=2))


@pytest.mark.parametrize('seed, point', [
    (1385, [-0.49988080181612954, 5.736128140242343, -25.165556345356073]),
    (1883, [4.948457289217935, 8.500866330326478, -9.895426472082915]),
])
def test_point_recovery_retries_after_neighboring_bad_cells_are_repaired(monkeypatch, seed, point):
    nodes, elems, fixed = _fixture(seed)
    points = np.vstack([nodes[fixed], point])
    surface = _grid(nodes, elems).extract_surface()
    monkeypatch.setattr(constrained, 'resolve_tetgen_exe', lambda tetgen_exe=None: '/tmp/tetgen')
    monkeypatch.setattr(constrained, 'run_tetgen', lambda *args: None)
    monkeypatch.setattr(constrained, 'read_node', lambda path: (nodes, {}))
    monkeypatch.setattr(constrained, 'read_ele', lambda path, index_map: elems)

    grid, output, repaired, meta = constrained.tetrahedralize_with_prescribed_points(surface, points)

    np.testing.assert_array_equal(output[meta['node_ids']], points)
    np.testing.assert_array_equal(output[:len(nodes)], nodes)
    assert np.isin(meta['node_ids'], repaired).all()
    assert grid.cell_quality('scaled_jacobian')['scaled_jacobian'].min() >= 1e-12
    assert _boundary_faces(repaired) == _boundary_faces(elems)


def test_independent_quality_repairs_can_progress_with_missing_quadratic_constraints(monkeypatch):
    first_nodes, first_elems, first_fixed = _fixture(1385)
    second_nodes, second_elems, second_fixed = _fixture(1883)
    nodes = np.vstack([first_nodes, second_nodes])
    elems = np.vstack([first_elems, second_elems + len(first_nodes)])
    fixed = np.concatenate([first_fixed, second_fixed])
    points = np.vstack([nodes[fixed],
                        [-0.49988080181612954, 5.736128140242343, -25.165556345356073],
                        [4.948457289217935, 8.500866330326478, -9.895426472082915]])
    surface = _grid(nodes, elems).extract_surface()
    native_nodes, native_elems = _quadratic(nodes, elems)
    monkeypatch.setattr(constrained, 'resolve_tetgen_exe', lambda tetgen_exe=None: '/tmp/tetgen')
    monkeypatch.setattr(constrained, 'run_tetgen', lambda *args: None)
    monkeypatch.setattr(constrained, 'read_node', lambda path: (native_nodes, {}))
    monkeypatch.setattr(constrained, 'read_ele', lambda path, index_map: native_elems)

    _, output, repaired, meta = constrained.tetrahedralize_with_prescribed_points(surface, points, order=2)

    np.testing.assert_array_equal(output[meta['node_ids']], points)
    np.testing.assert_array_equal(output[:len(native_nodes)], native_nodes)
    assert np.isin(meta['node_ids'], repaired).all()
    assert _grid(output, repaired).cell_quality('scaled_jacobian')['scaled_jacobian'].min() >= 1e-12
    assert _boundary_faces(repaired[:, :4]) == _boundary_faces(elems)
