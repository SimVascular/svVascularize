import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from svv.utils.meshing.tetgen import get_packaged_tetgen_cli_path


def resolve_tetgen_exe(tetgen_exe=None):
    if tetgen_exe:
        exe = Path(tetgen_exe).expanduser()
        if exe.is_file():
            return str(exe)
        raise RuntimeError(f"TetGen executable not found: {tetgen_exe}")

    env_path = os.environ.get("SVV_TETGEN_PATH")
    if env_path:
        exe = Path(env_path).expanduser()
        if exe.is_file():
            return str(exe)

    packaged = get_packaged_tetgen_cli_path()
    if packaged:
        return packaged

    found = shutil.which("tetgen")
    if found:
        return found

    raise RuntimeError(
        "TetGen executable not found. Set tetgen_exe=..., set SVV_TETGEN_PATH, build the packaged CLI with "
        "`python setup.py build_ext --build-tetgen-cli`, or install tetgen on PATH."
    )


def _iter_data_lines(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                yield stripped


def write_poly(surface, path):
    surface = surface.extract_surface().triangulate()
    if not surface.is_all_triangles:
        raise ValueError("TetGen PLC export requires an all-triangle surface mesh.")

    faces = surface.faces.reshape(-1, 4)[:, 1:]
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(f"{surface.n_points} 3 0 0\n")
        for idx, point in enumerate(surface.points, start=1):
            stream.write(f"{idx} {point[0]} {point[1]} {point[2]}\n")

        stream.write(f"{faces.shape[0]} 0\n")
        for tri in faces:
            stream.write("1 0 0\n")
            stream.write(f"3 {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")

        stream.write("0\n")
        stream.write("0\n")


def write_a_node(points, path):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(f"{points.shape[0]} 3 0 0\n")
        for idx, point in enumerate(points, start=1):
            stream.write(f"{idx} {point[0]} {point[1]} {point[2]}\n")


def run_tetgen(exe, switches, stem, cwd):
    cmd = [exe, f"-{switches}", stem]
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"TetGen failed with code {result.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )


def read_node(path):
    line_iter = iter(_iter_data_lines(path))
    header = next(line_iter).split()
    n_points = int(header[0])
    dim = int(header[1])
    if dim != 3:
        raise ValueError(f"Expected 3D TetGen nodes, got dimension {dim}.")

    points = np.zeros((n_points, dim), dtype=float)
    index_map = {}
    for row in range(n_points):
        parts = next(line_iter).split()
        node_id = int(parts[0])
        points[row, :] = [float(parts[1]), float(parts[2]), float(parts[3])]
        index_map[node_id] = row
    return points, index_map


def read_ele(path, index_map):
    line_iter = iter(_iter_data_lines(path))
    header = next(line_iter).split()
    n_cells = int(header[0])
    nodes_per_cell = int(header[1])

    elems = np.zeros((n_cells, nodes_per_cell), dtype=np.int64)
    for row in range(n_cells):
        parts = next(line_iter).split()
        node_ids = [int(value) for value in parts[1:1 + nodes_per_cell]]
        elems[row, :] = [index_map[node_id] for node_id in node_ids]
    return elems


def _normalize_point_metadata(point_metadata, point_count):
    normalized = {}
    for name, values in (point_metadata or {}).items():
        array = np.asarray(values).reshape(-1)
        if array.shape[0] != point_count:
            raise ValueError(
                f"Point metadata '{name}' must have length {point_count}, got {array.shape[0]}."
            )
        normalized[name] = array
    return normalized


def _relative_enclosure_tolerance(surface, verify_tol):
    bounds = np.asarray(surface.bounds, dtype=float)
    spans = bounds[1::2] - bounds[::2]
    diagonal = float(np.linalg.norm(spans))
    if not np.isfinite(diagonal) or diagonal <= 0.0:
        return 1e-9

    abs_tol = max(float(verify_tol), 0.0)
    if abs_tol <= 0.0:
        abs_tol = diagonal * 1e-9

    rel_tol = abs_tol / diagonal
    return min(max(rel_tol * 10.0, 1e-12), 1e-4)


def filter_prescribed_points_to_surface(
    surface,
    prescribed_points,
    prescribed_lines=None,
    *,
    point_metadata=None,
    verify_tol=1e-6,
):
    prescribed_points = np.asarray(prescribed_points, dtype=float).reshape(-1, 3)
    prescribed_lines = (
        np.empty((0, 2), dtype=np.int64)
        if prescribed_lines is None
        else np.asarray(prescribed_lines, dtype=np.int64).reshape(-1, 2)
    )
    point_metadata = _normalize_point_metadata(point_metadata, prescribed_points.shape[0])

    if prescribed_points.shape[0] == 0:
        return {
            "points": prescribed_points,
            "lines": prescribed_lines,
            "point_metadata": point_metadata,
            "kept_point_ids": np.empty((0,), dtype=np.int64),
            "dropped_point_ids": np.empty((0,), dtype=np.int64),
        }

    surface_mesh = surface.extract_surface().triangulate().clean()
    if surface_mesh.n_points == 0 or surface_mesh.n_cells == 0:
        raise ValueError("Cannot filter prescribed points against an empty tissue surface mesh.")

    selector = pv.PolyData(prescribed_points)
    selected = selector.select_enclosed_points(
        surface_mesh,
        tolerance=_relative_enclosure_tolerance(surface_mesh, verify_tol),
        check_surface=True,
    )
    inside_mask = np.asarray(selected["SelectedPoints"]).reshape(-1).astype(bool)
    kept_point_ids = np.flatnonzero(inside_mask)
    dropped_point_ids = np.flatnonzero(~inside_mask)

    if kept_point_ids.size == 0:
        raise RuntimeError(
            "No prescribed spline points lie inside the tissue domain after filtering. "
            f"Dropped {dropped_point_ids.size} of {prescribed_points.shape[0]} points."
        )

    old_to_new = np.full(prescribed_points.shape[0], -1, dtype=np.int64)
    old_to_new[kept_point_ids] = np.arange(kept_point_ids.size, dtype=np.int64)

    remapped_lines = []
    seen_lines = set()
    for line in prescribed_lines:
        start = int(old_to_new[int(line[0])])
        end = int(old_to_new[int(line[1])])
        if start < 0 or end < 0 or start == end:
            continue
        key = (start, end)
        if key in seen_lines:
            continue
        seen_lines.add(key)
        remapped_lines.append([start, end])

    filtered_metadata = {
        name: np.asarray(values)[kept_point_ids]
        for name, values in point_metadata.items()
    }

    return {
        "points": prescribed_points[kept_point_ids],
        "lines": np.asarray(remapped_lines, dtype=np.int64).reshape(-1, 2),
        "point_metadata": filtered_metadata,
        "kept_point_ids": kept_point_ids,
        "dropped_point_ids": dropped_point_ids,
    }


def verify_prescribed_points(nodes, prescribed_points, tol):
    nodes = np.asarray(nodes, dtype=float).reshape(-1, 3)
    if nodes.shape[0] == 0:
        raise RuntimeError("TetGen output mesh has no nodes to verify prescribed spline points.")

    tree = cKDTree(nodes)
    distances, node_ids = tree.query(np.asarray(prescribed_points, dtype=float).reshape(-1, 3))
    distances = np.asarray(distances, dtype=float)
    node_ids = np.asarray(node_ids, dtype=np.int64)
    if np.any(distances > tol):
        failing = np.flatnonzero(distances > tol)
        preview = failing[:20].tolist()
        preview_suffix = "" if failing.size <= 20 else f" (showing first 20 of {failing.size})"
        max_distance = float(distances[failing].max())
        raise RuntimeError(
            "TetGen output does not contain all prescribed spline points within tolerance. "
            f"Tolerance: {tol:g}. Failed point ids: {preview}{preview_suffix}. "
            f"Max miss distance: {max_distance:.6g}"
        )
    return node_ids, distances


def attach_constraint_metadata(
    grid,
    node_ids,
    point_metadata=None,
    line_count=0,
    original_point_count=None,
    dropped_point_count=0,
    unique_node_count=None,
    max_assignment_distance=0.0,
    tolerance_exceeded_count=0,
):
    point_metadata = point_metadata or {}
    node_ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)

    centerline_node = np.zeros(grid.n_points, dtype=np.uint8)
    centerline_spline_id = np.full(grid.n_points, -1, dtype=np.int32)
    centerline_spline_order = np.full(grid.n_points, -1, dtype=np.int32)
    centerline_radius = np.full(grid.n_points, np.nan, dtype=float)

    centerline_node[node_ids] = 1
    if "spline_id" in point_metadata:
        centerline_spline_id[node_ids] = np.asarray(point_metadata["spline_id"], dtype=np.int32)
    if "spline_order" in point_metadata:
        centerline_spline_order[node_ids] = np.asarray(point_metadata["spline_order"], dtype=np.int32)
    if "radius" in point_metadata:
        centerline_radius[node_ids] = np.asarray(point_metadata["radius"], dtype=float)

    if original_point_count is None:
        original_point_count = node_ids.shape[0] + int(dropped_point_count)
    if unique_node_count is None:
        unique_node_count = np.unique(node_ids).shape[0]

    # Preserve the older centerline-specific name and add a generic alias for
    # all constraint sources, including explicit prescribed points.
    grid.point_data["constrained_point"] = centerline_node.copy()
    grid.point_data["centerline_node"] = centerline_node
    grid.point_data["centerline_spline_id"] = centerline_spline_id
    grid.point_data["centerline_spline_order"] = centerline_spline_order
    grid.point_data["centerline_radius"] = centerline_radius
    grid.field_data["centerline_constraint_count"] = np.array([node_ids.shape[0]], dtype=np.int32)
    grid.field_data["centerline_constraint_line_count"] = np.array([int(line_count)], dtype=np.int32)
    grid.field_data["centerline_constraint_original_count"] = np.array([int(original_point_count)], dtype=np.int32)
    grid.field_data["centerline_constraint_dropped_count"] = np.array([int(dropped_point_count)], dtype=np.int32)
    grid.field_data["centerline_constraint_unique_node_count"] = np.array([int(unique_node_count)], dtype=np.int32)
    grid.field_data["centerline_constraint_assignment_max_distance"] = np.array([float(max_assignment_distance)], dtype=float)
    grid.field_data["centerline_constraint_tolerance_exceeded_count"] = np.array([int(tolerance_exceeded_count)], dtype=np.int32)


def _cell_types(nodes_per_cell, n_cells):
    if nodes_per_cell == 4:
        return np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
    if nodes_per_cell == 10:
        return np.full(n_cells, pv.CellType.QUADRATIC_TETRA, dtype=np.uint8)
    raise ValueError(f"Unexpected number of vertices per TetGen element: {nodes_per_cell}")


def tetrahedralize_with_prescribed_points(
    surface,
    prescribed_points,
    prescribed_lines=None,
    *,
    minratio=1.1,
    mindihedral=10.0,
    order=1,
    verify_tol=1e-6,
    tetgen_exe=None,
    point_metadata=None,
):
    prescribed_points = np.asarray(prescribed_points, dtype=float).reshape(-1, 3)
    if prescribed_points.shape[0] == 0:
        raise ValueError("tetrahedralize_with_prescribed_points requires at least one prescribed point.")

    prescribed_lines = (
        np.empty((0, 2), dtype=np.int64)
        if prescribed_lines is None
        else np.asarray(prescribed_lines, dtype=np.int64).reshape(-1, 2)
    )
    point_metadata = _normalize_point_metadata(point_metadata, prescribed_points.shape[0])
    filtered = filter_prescribed_points_to_surface(
        surface,
        prescribed_points,
        prescribed_lines,
        point_metadata=point_metadata,
        verify_tol=verify_tol,
    )

    filtered_points = filtered["points"]
    filtered_lines = filtered["lines"]
    filtered_metadata = filtered["point_metadata"]
    dropped_point_ids = filtered["dropped_point_ids"]

    exe = resolve_tetgen_exe(tetgen_exe)
    order_switch = "o2" if int(order) == 2 else ""
    quality_switches = f"q{minratio}/{mindihedral}{order_switch}"

    with tempfile.TemporaryDirectory(prefix="svv_tetgen_constraints_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        base_name = "surface"
        poly_path = tmpdir_path / f"{base_name}.poly"
        write_poly(surface, poly_path)

        run_tetgen(exe, f"p{quality_switches}Q", poly_path.name, tmpdir)

        insert_path = tmpdir_path / f"{base_name}.1.a.node"
        write_a_node(filtered_points, insert_path)
        run_tetgen(exe, f"ri{order_switch}JMQ", f"{base_name}.1", tmpdir)

        node_path = tmpdir_path / f"{base_name}.2.node"
        ele_path = tmpdir_path / f"{base_name}.2.ele"
        nodes, index_map = read_node(node_path)
        elems = read_ele(ele_path, index_map)

    n_cells, n_vertices_per_cell = elems.shape
    cells = np.hstack(
        [
            np.full((n_cells, 1), n_vertices_per_cell, dtype=np.int64),
            elems.astype(np.int64),
        ]
    ).ravel()
    celltypes = _cell_types(n_vertices_per_cell, n_cells)
    grid = pv.UnstructuredGrid(cells, celltypes, nodes)

    node_ids, distances = verify_prescribed_points(nodes, filtered_points, verify_tol)
    unique_node_count = np.unique(node_ids).shape[0]
    max_assignment_distance = float(distances.max()) if distances.size else 0.0

    attach_constraint_metadata(
        grid,
        node_ids,
        point_metadata=filtered_metadata,
        line_count=filtered_lines.shape[0],
        original_point_count=prescribed_points.shape[0],
        dropped_point_count=dropped_point_ids.shape[0],
        unique_node_count=unique_node_count,
        max_assignment_distance=max_assignment_distance,
        tolerance_exceeded_count=0,
    )

    return grid, nodes, elems, {
        "node_ids": node_ids,
        "distances": distances,
        "line_count": int(filtered_lines.shape[0]),
        "original_point_count": int(prescribed_points.shape[0]),
        "retained_point_count": int(filtered_points.shape[0]),
        "dropped_point_count": int(dropped_point_ids.shape[0]),
        "dropped_point_ids": dropped_point_ids.tolist(),
        "unique_node_count": int(unique_node_count),
        "max_assignment_distance": max_assignment_distance,
        "insertion_switches": f"ri{order_switch}JMQ",
    }
