from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from scipy.interpolate import splprep, splev

from svv.forest.export.export_spline import export_spline
from svv.tree.export.write_splines import get_interpolated_sv_data


def export_spline_files(obj, destination, *, spline_sample_points: int = 100, separate: bool = False) -> list[Path]:
    """
    Export spline samples for a tree-like or forest-like object.

    The GUI holds concrete ``Tree`` / ``Forest`` instances, but this helper
    intentionally uses structural checks so it remains testable without
    importing the full generation stack.
    """
    out_path = Path(destination)
    if not out_path.suffix:
        out_path = out_path.with_suffix(".txt")

    if _is_connected_forest(obj):
        connected_outputs = _collect_connected_forest_outputs(obj)
        written = []
        for dst, (_, _, all_points, all_radii) in zip(
            _connected_output_paths(out_path, connected_outputs),
            connected_outputs,
        ):
            _write_connected_spline_file(
                all_points,
                all_radii,
                dst,
                spline_sample_points=spline_sample_points,
                separate=separate,
            )
            written.append(dst)
        return written

    if _is_forest(obj):
        trees = _collect_forest_trees(obj)
        if not trees:
            raise ValueError("Forest contains no trees to export.")
        if len(trees) == 1:
            _write_tree_splines(
                trees[0][2],
                out_path,
                spline_sample_points=spline_sample_points,
                separate=separate,
            )
            return [out_path]

        written = []
        stem = out_path.stem or "splines"
        suffix = out_path.suffix or ".txt"
        for net_idx, tree_idx, tree in trees:
            dst = out_path.with_name(f"{stem}_network{net_idx}_tree{tree_idx}{suffix}")
            _write_tree_splines(
                tree,
                dst,
                spline_sample_points=spline_sample_points,
                separate=separate,
            )
            written.append(dst)
        return written

    _write_tree_splines(
        obj,
        out_path,
        spline_sample_points=spline_sample_points,
        separate=separate,
    )
    return [out_path]


def _is_forest(obj) -> bool:
    return hasattr(obj, "networks")


def _is_connected_forest(obj) -> bool:
    if not _is_forest(obj):
        return False
    tree_connections = getattr(getattr(obj, "connections", None), "tree_connections", None)
    return bool(tree_connections)


def _collect_forest_trees(forest) -> list[tuple[int, int, object]]:
    trees = []
    for net_idx, network in enumerate(getattr(forest, "networks", []) or []):
        for tree_idx, tree in enumerate(network):
            trees.append((net_idx, tree_idx, tree))
    return trees


def _collect_connected_forest_outputs(forest) -> list[tuple[int, int, list, list]]:
    outputs = []
    for fallback_network_idx, tree_connection in enumerate(
        getattr(forest.connections, "tree_connections", []) or []
    ):
        network_id = int(getattr(tree_connection, "network_id", fallback_network_idx))
        _, _, _, all_points, all_radii, _ = export_spline(tree_connection)
        for component_idx, (points, radii) in enumerate(zip(all_points, all_radii)):
            outputs.append((network_id, component_idx, points, radii))
    if not outputs:
        raise ValueError("Connected forest has no spline networks to export.")
    return outputs


def _connected_output_paths(base_path: Path, outputs: list[tuple[int, int, list, list]]) -> list[Path]:
    if len(outputs) == 1:
        return [base_path]

    stem = base_path.stem or "splines"
    suffix = base_path.suffix or ".txt"
    counts_by_network = Counter(network_id for network_id, _, _, _ in outputs)
    paths = []
    for network_id, component_idx, _, _ in outputs:
        if counts_by_network[network_id] == 1:
            name = f"{stem}_network{network_id}{suffix}"
        else:
            name = f"{stem}_network{network_id}_{component_idx}{suffix}"
        paths.append(base_path.with_name(name))
    return paths


def _ensure_parent(dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)


def _write_tree_splines(tree, dst: Path, *, spline_sample_points: int, separate: bool) -> None:
    data = getattr(tree, "data", None)
    if data is None:
        raise ValueError("Tree has no data to export.")
    *_, interp_xyzr = get_interpolated_sv_data(data)
    if not interp_xyzr:
        raise ValueError("Tree has no spline branches to export.")

    _ensure_parent(dst)
    t = np.linspace(0, 1, num=spline_sample_points)
    with dst.open("w", encoding="utf-8") as spline_file:
        for vessel_idx, vessel_ctr in enumerate(interp_xyzr):
            spline_file.write(f"Vessel: {vessel_idx}, Number of Points: {spline_sample_points}\n\n")
            data = splev(t, vessel_ctr[0])
            _write_samples(
                spline_file,
                data,
                spline_sample_points=spline_sample_points,
                separate=separate,
            )
            spline_file.write("\n")


def _write_connected_spline_file(
    all_points,
    all_radii,
    dst: Path,
    *,
    spline_sample_points: int,
    separate: bool,
) -> None:
    _ensure_parent(dst)
    t = np.linspace(0, 1, num=spline_sample_points)
    with dst.open("w", encoding="utf-8") as spline_file:
        for vessel_idx, (points, radii) in enumerate(zip(all_points, all_radii)):
            pt_array = np.asarray(points)
            r_array = np.asarray(radii).reshape(-1, 1)
            vessel_ctr = splprep(np.hstack((pt_array, r_array)).T, s=0)
            spline_file.write(f"Vessel: {vessel_idx}, Number of Points: {spline_sample_points}\n\n")
            data = splev(t, vessel_ctr[0])
            _write_samples(
                spline_file,
                data,
                spline_sample_points=spline_sample_points,
                separate=separate,
            )
            spline_file.write("\n")


def _write_samples(spline_file, data, *, spline_sample_points: int, separate: bool) -> None:
    for sample_idx in range(spline_sample_points):
        row = [data[0][sample_idx], data[1][sample_idx], data[2][sample_idx], data[3][sample_idx]]
        if separate:
            row.append(1 if sample_idx > spline_sample_points // 2 else 0)
        spline_file.write(", ".join(str(value) for value in row))
        spline_file.write("\n")
