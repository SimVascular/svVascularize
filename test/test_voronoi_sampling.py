import numpy as np
import pyvista as pv

from svv.domain.domain import Domain


def _make_tet_mesh(points: np.ndarray, tets: np.ndarray) -> pv.UnstructuredGrid:
    points = np.asarray(points, dtype=float)
    tets = np.asarray(tets, dtype=np.int64)
    n_cells = int(tets.shape[0])
    cells = np.hstack([np.full((n_cells, 1), 4, dtype=np.int64), tets]).ravel()
    celltypes = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, celltypes, points)


def _dummy_domain_with_mesh(mesh: pv.UnstructuredGrid) -> Domain:
    dom = Domain(np.zeros((1, 3), dtype=float))
    dom.mesh = mesh
    # Provide a cheap implicit evaluator so voronoi sampling can filter by implicit_range.
    dom.evaluate_fast = lambda pts, **_: -0.5 * np.ones((np.asarray(pts).shape[0], 1), dtype=float)
    return dom


def test_voronoi_sampling_is_deterministic_and_uses_cell_centers():
    # Five tetrahedra with distinct cell centers.
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [3, 0, 0],
            [4, 0, 0],
            [3, 1, 0],
            [3, 0, 1],
            [0, 3, 0],
            [0, 4, 0],
            [1, 3, 0],
            [0, 3, 1],
            [0, 0, 3],
            [1, 0, 3],
            [0, 1, 3],
            [0, 0, 4],
            [2, 2, 2],
            [3, 2, 2],
            [2, 3, 2],
            [2, 2, 3],
        ],
        dtype=float,
    )
    tets = np.array(
        [
            [0, 1, 2, 3],      # near origin
            [4, 5, 6, 7],      # +x
            [8, 9, 10, 11],    # +y
            [12, 13, 14, 15],  # +z
            [16, 17, 18, 19],  # diagonal
        ],
        dtype=np.int64,
    )
    mesh = _make_tet_mesh(points, tets)
    dom = _dummy_domain_with_mesh(mesh)

    seeds = np.array([[0.0, 0.0, 0.0]])

    pts1, cells1 = dom.get_interior_points(3, tree=seeds, threshold=0.0, method="voronoi", implicit_range=(-1.0, 0.0))
    pts2, cells2 = dom.get_interior_points(3, tree=seeds, threshold=0.0, method="voronoi", implicit_range=(-1.0, 0.0))

    assert np.allclose(pts1, pts2, equal_nan=True)
    assert np.array_equal(cells1, cells2)

    centers = mesh.cell_centers().points
    for p, cid in zip(pts1, cells1):
        assert cid >= 0
        assert np.allclose(p, centers[cid])

    # Farthest-first order should pick the diagonal cell, then +x, then +y given the stable tie-breaks.
    assert cells1.tolist() == [4, 1, 2]


def test_voronoi_sampling_respects_volume_threshold():
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [3, 0, 0],
            [4, 0, 0],
            [3, 1, 0],
            [3, 0, 1],
            [2, 2, 2],
            [3, 2, 2],
            [2, 3, 2],
            [2, 2, 3],
        ],
        dtype=float,
    )
    tets = np.array(
        [
            [0, 1, 2, 3],     # near origin
            [4, 5, 6, 7],     # +x (dist ~3.268)
            [8, 9, 10, 11],   # diagonal (dist ~3.897)
        ],
        dtype=np.int64,
    )
    mesh = _make_tet_mesh(points, tets)
    dom = _dummy_domain_with_mesh(mesh)

    seeds = np.array([[0.0, 0.0, 0.0]])

    # Exclude the diagonal cell by limiting the allowed distance to the existing tree.
    pts, cells = dom.get_interior_points(
        2,
        tree=seeds,
        threshold=0.0,
        volume_threshold=3.3,
        method="voronoi",
        implicit_range=(-1.0, 0.0),
    )

    assert np.array_equal(cells, np.array([1, 0], dtype=np.int64))
    centers = mesh.cell_centers().points
    assert np.allclose(pts[0], centers[1])
    assert np.allclose(pts[1], centers[0])
