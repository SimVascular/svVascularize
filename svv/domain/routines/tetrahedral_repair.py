"""Local tetrahedron repairs that preserve prescribed coordinates."""

from fractions import Fraction
from itertools import combinations

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


# The additional node slots in TetGen's quadratic .ele output.
_TETGEN_EDGES = np.array([[2, 3], [0, 3], [0, 1], [1, 2], [1, 3], [0, 2]])
_FACES = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
# Numerical degeneracy guard, separate from TetGen's requested quality targets.
_MIN_SCALED_JACOBIAN = 1e-12


def _orientation(vertices):
    """Exact determinant of the stored coordinates, used only in local repairs."""
    a, b, c, d = [[Fraction(float(value)) for value in point] for point in vertices]
    u, v, w = [[point[i] - a[i] for i in range(3)] for point in (b, c, d)]
    return (u[0] * (v[1] * w[2] - v[2] * w[1])
            - u[1] * (v[0] * w[2] - v[2] * w[0])
            + u[2] * (v[0] * w[1] - v[1] * w[0]))


def _linear_grid(nodes, elems):
    cells = np.column_stack([np.full(len(elems), 4), elems[:, :4]]).ravel()
    return pv.UnstructuredGrid(cells, np.full(len(elems), pv.CellType.TETRA), nodes)


def _scaled_jacobians(nodes, elems):
    grid = _linear_grid(nodes, elems)
    if hasattr(grid, "cell_quality"):
        return np.asarray(grid.cell_quality("scaled_jacobian")["scaled_jacobian"])
    return np.asarray(grid.compute_cell_quality(quality_measure="scaled_jacobian")["CellQuality"])


def _cavity_boundary(elems):
    faces = elems[:, _FACES].reshape(-1, 3)
    _, indices, counts = np.unique(np.sort(faces, axis=1), axis=0,
                                  return_index=True, return_counts=True)
    if np.any(counts > 2):
        raise RuntimeError("Cannot repair a nonmanifold tetrahedron neighborhood.")
    return faces[indices[counts == 1]]


def _containing_supports(nodes, elems, point):
    grid = _linear_grid(nodes, elems)
    first = int(grid.find_containing_cell(point))
    # A tolerant locator can choose the wrong side of a nearly coplanar face.
    # Fall back to the other bounding boxes only if exact containment rejects it.
    def candidates():
        if first >= 0:
            yield first
        yield from grid.find_cells_within_bounds(np.repeat(point, 2))

    for cell_id in candidates():
        cell = elems[cell_id, :4]
        vertices = nodes[cell]
        volume = _orientation(vertices)
        if volume <= 0:
            continue
        weights = []
        for i in range(4):
            child = vertices.copy()
            child[i] = point
            weights.append(_orientation(child))
        if any(weight < 0 for weight in weights):
            continue
        # If a direct split is too thin, try the cavity across the closest
        # face, then edge. Exact containment and child quality gate each attempt.
        ranked = cell[sorted(range(4), key=weights.__getitem__, reverse=True)]
        return [ranked[:size] for size in range(sum(weight > 0 for weight in weights), 1, -1)]
    return []


def _replace_cells(nodes, elems, affected, children):
    """Replace a conforming neighborhood, sharing existing quadratic edge nodes."""
    if elems.shape[1] == 10:
        edge_nodes = {}
        for cell in elems[affected]:
            for edge, midpoint in zip(cell[:4][_TETGEN_EDGES], cell[4:]):
                edge_nodes[tuple(sorted(edge))] = midpoint
        added = []
        midpoints = []
        for cell in children:
            row = []
            for edge in cell[_TETGEN_EDGES]:
                key = tuple(sorted(edge))
                if key not in edge_nodes:
                    edge_nodes[key] = len(nodes) + len(added)
                    added.append(nodes[edge].mean(axis=0))
                row.append(edge_nodes[key])
            midpoints.append(row)
        if added:
            nodes = np.vstack([nodes, added])
        children = np.column_stack([children, midpoints])
    return nodes, np.vstack([np.delete(elems, affected, axis=0), children])


def recover_prescribed_points(nodes, elems, points, tol):
    """Connect omitted points by subdividing their containing cell, face or edge.

    A point must be inside the output volume at its exact coordinates. Points
    outside that volume remain unconnected for the caller's verification to reject.
    """
    if not len(elems):
        return nodes, elems
    attempted = np.zeros(len(points), dtype=bool)
    while True:
        # A quadratic edge split can disconnect a formerly prescribed midpoint.
        # Recheck all constraints, including those connected before the repair.
        distances = cKDTree(nodes[np.unique(elems)]).query(points)[0]
        missing = np.flatnonzero((distances > tol) & ~attempted)
        if not len(missing):
            return nodes, elems
        i = missing[0]
        attempted[i] = True
        point = points[i]
        for support in _containing_supports(nodes, elems, point):
            affected = np.flatnonzero(np.isin(elems[:, :4], support).sum(axis=1) == len(support))
            matches = np.flatnonzero(np.all(nodes == point, axis=1))
            point_id = int(matches[0]) if len(matches) else len(nodes)
            updated_nodes = nodes if len(matches) else np.vstack([nodes, point])
            boundary = _cavity_boundary(elems[affected, :4])
            children = np.column_stack([np.full(len(boundary), point_id), boundary])
            orientations = [_orientation(updated_nodes[child]) for child in children]
            if any(value < 0 for value in orientations):
                continue
            # A point exactly on an exterior face partitions that face; its flat
            # cone is omitted, leaving the nonzero cones to cover the cell.
            children = children[[value > 0 for value in orientations]]
            if not len(children):
                continue
            x = updated_nodes[children]
            determinants = np.linalg.det(x[:, 1:] - x[:, :1])
            quality = _scaled_jacobians(updated_nodes, children)
            if not (np.isfinite(determinants).all() and (determinants > 0).all()
                    and np.isfinite(quality).all() and (quality >= _MIN_SCALED_JACOBIAN).all()):
                continue
            nodes, elems = _replace_cells(updated_nodes, elems, affected, children)
            break


def _bad_tetrahedra(nodes, elems):
    quality = _scaled_jacobians(nodes, elems)
    x = nodes[elems[:, :4]]
    determinants = np.linalg.det(x[:, 1:] - x[:, :1])
    return np.flatnonzero(~np.isfinite(quality) | (quality < _MIN_SCALED_JACOBIAN)
                         | ~np.isfinite(determinants) | (determinants <= 0))


def _collinear_cavity(nodes, elems, cell_id):
    """Replace a nearly collinear face by an interior point's conforming star."""
    for triple in combinations(elems[cell_id, :4], 3):
        ids = np.asarray(triple)
        pairs = np.array([[0, 1], [0, 2], [1, 2]])
        lengths = np.linalg.norm(nodes[ids[pairs[:, 1]]] - nodes[ids[pairs[:, 0]]], axis=1)
        if lengths.max() == 0 or len(np.unique(ids)) != 3:
            continue
        ends = ids[pairs[lengths.argmax()]]
        middle = int(ids[~np.isin(ids, ends)][0])
        edge = nodes[ends[1]] - nodes[ends[0]]
        offset = nodes[middle] - nodes[ends[0]]
        fraction = np.dot(offset, edge) / np.dot(edge, edge)
        scale = max(float(np.abs(nodes[ids]).max()), float(lengths.max()))
        distance = np.linalg.norm(np.cross(offset, edge)) / lengths.max()
        if not 0 < fraction < 1 or distance > 64 * np.finfo(float).eps * scale:
            continue

        # Include every interior sample along the spanning edge. Overlapping
        # collinear triples must be repaired together, or their shared faces
        # can leave a newly coned cell nearly flat or inverted.
        corners = elems[:, :4]
        used = np.unique(corners)
        offsets = nodes[used] - nodes[ends[0]]
        fractions = offsets @ edge / np.dot(edge, edge)
        distances = np.linalg.norm(np.cross(offsets, edge), axis=1) / lengths.max()
        interior = used[(fractions > 0) & (fractions < 1)
                        & ~np.isin(used, ends)
                        & (distances <= 64 * np.finfo(float).eps * scale)]
        affected = np.flatnonzero(np.isin(corners, interior).any(axis=1)
                                 | (np.isin(corners, ends).sum(axis=1) == 2))
        boundary = _cavity_boundary(corners[affected])
        if np.isin(boundary, interior).any():
            continue  # Keep the tissue boundary, including its triangulation.
        children = np.column_stack([np.full(len(boundary), middle), boundary])
        orientations = [_orientation(nodes[child]) for child in children]
        if not orientations or any(value <= 0 for value in orientations):
            continue  # The point must see every cavity face from the inside.
        if sum(orientations) != sum(_orientation(nodes[parent]) for parent in corners[affected]):
            continue
        omitted = np.setdiff1d(np.unique(corners[affected]), np.unique(children))
        if len(omitted):
            # Reuse the original node ids and coordinates while splitting the
            # new cavity consistently along the remaining samples.
            _, children = recover_prescribed_points(nodes, children, nodes[omitted], tol=0.0)
        if not np.array_equal(np.unique(children), np.unique(corners[affected])):
            continue
        return _replace_cells(nodes, elems, affected, children)
    return None


def repair_degenerate_tetrahedra(nodes, elems):
    """Repair numerically flat cells after insertion without moving mesh nodes.

    Every candidate has positive exact orientation and conserved cavity volume.
    Overlapping defects may need several steps, each reducing the bad-cell count.
    Return only once all cells have positive floating-point volume and usable
    scaled Jacobians. Unsupported cavities remain explicit errors.
    """
    bad = _bad_tetrahedra(nodes, elems)
    while len(bad):
        for cell_id in bad:
            repaired = _collinear_cavity(nodes, elems, cell_id)
            if repaired is None:
                continue
            updated_nodes, updated_elems = repaired
            remaining = _bad_tetrahedra(updated_nodes, updated_elems)
            if len(remaining) >= len(bad):
                continue
            nodes, elems, bad = updated_nodes, updated_elems, remaining
            break
        else:
            raise RuntimeError(
                "TetGen output contains near-degenerate tetrahedra that cannot be repaired "
                f"while preserving mesh coordinates and boundary faces. Cell ids: {bad[:20].tolist()}."
            )
    return nodes, elems
