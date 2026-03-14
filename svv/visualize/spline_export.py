from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.interpolate import splprep, splev

from svv.forest.export.export_spline import export_spline
from svv.tree.export.write_splines import get_interpolated_sv_data


@dataclass(frozen=True)
class _ConnectedSplineOutput:
    network_id: int
    component_idx: int
    tree_indices: tuple[int, ...]
    all_points: list
    all_radii: list


def export_spline_files(
    obj,
    destination,
    *,
    spline_sample_points: int = 100,
    separate: bool = False,
    export_inlet_outlet_roots: bool = False,
    tree_root_role: Literal["inlet", "outlet"] | None = None,
    inlet_tree_indices_by_network: dict[int, set[int]] | None = None,
) -> list[Path]:
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
        inlet_tree_indices = (
            _normalize_inlet_tree_indices_by_network(obj, inlet_tree_indices_by_network)
            if export_inlet_outlet_roots
            else None
        )
        tree_connections_by_network = _tree_connections_by_network(obj)
        written = []
        for dst, output in zip(
            _connected_output_paths(out_path, connected_outputs),
            connected_outputs,
        ):
            _write_connected_spline_file(
                output.all_points,
                output.all_radii,
                dst,
                spline_sample_points=spline_sample_points,
                separate=separate,
            )
            if export_inlet_outlet_roots:
                inlet_points, outlet_points = _connected_output_root_points(
                    tree_connections_by_network[output.network_id],
                    output.tree_indices,
                    inlet_tree_indices[output.network_id],
                )
                _write_inlet_outlet_sidecar(
                    dst,
                    inlet_points=inlet_points,
                    outlet_points=outlet_points,
                )
            written.append(dst)
        return written

    if _is_forest(obj):
        trees = _collect_forest_trees(obj)
        if not trees:
            raise ValueError("Forest contains no trees to export.")
        inlet_tree_indices = (
            _normalize_inlet_tree_indices_by_network(obj, inlet_tree_indices_by_network)
            if export_inlet_outlet_roots
            else None
        )
        if len(trees) == 1:
            net_idx, tree_idx, tree = trees[0]
            _write_tree_splines(tree, out_path, spline_sample_points=spline_sample_points, separate=separate)
            if export_inlet_outlet_roots:
                _write_tree_root_sidecar(
                    out_path,
                    tree,
                    role=_forest_tree_root_role(net_idx, tree_idx, inlet_tree_indices),
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
            if export_inlet_outlet_roots:
                _write_tree_root_sidecar(
                    dst,
                    tree,
                    role=_forest_tree_root_role(net_idx, tree_idx, inlet_tree_indices),
                )
            written.append(dst)
        return written

    _write_tree_splines(
        obj,
        out_path,
        spline_sample_points=spline_sample_points,
        separate=separate,
    )
    if export_inlet_outlet_roots:
        _write_tree_root_sidecar(out_path, obj, role=_normalize_tree_root_role(tree_root_role))
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


def _tree_connections_by_network(forest) -> dict[int, object]:
    tree_connections = getattr(getattr(forest, "connections", None), "tree_connections", None) or []
    return {
        int(getattr(tree_connection, "network_id", fallback_network_idx)): tree_connection
        for fallback_network_idx, tree_connection in enumerate(tree_connections)
    }


def _forest_tree_counts(forest) -> dict[int, int]:
    counts = {
        net_idx: int(count)
        for net_idx, count in enumerate(getattr(forest, "n_trees_per_network", []) or [])
    }
    for net_idx, network in enumerate(getattr(forest, "networks", []) or []):
        counts[net_idx] = max(counts.get(net_idx, 0), len(network))
    for network_id, tree_connection in _tree_connections_by_network(forest).items():
        counts[network_id] = max(
            counts.get(network_id, 0),
            len(getattr(tree_connection, "connected_network", []) or []),
        )
    return counts


def _normalize_tree_root_role(role: Literal["inlet", "outlet"] | None) -> Literal["inlet", "outlet"]:
    if role not in {"inlet", "outlet"}:
        raise ValueError("Tree spline sidecar export requires tree_root_role to be 'inlet' or 'outlet'.")
    return role


def _normalize_inlet_tree_indices_by_network(
    forest,
    mapping: dict[int, set[int]] | None,
) -> dict[int, set[int]]:
    counts = _forest_tree_counts(forest)
    if not counts:
        raise ValueError("Forest contains no tree networks for inlet/outlet spline sidecar export.")

    normalized = {net_idx: set() for net_idx in counts}
    if mapping is None:
        return normalized

    for raw_network_id, raw_tree_indices in mapping.items():
        network_id = int(raw_network_id)
        if network_id not in counts:
            raise ValueError(f"Invalid forest network id for spline sidecar export: {network_id}.")
        if raw_tree_indices is None:
            continue
        if isinstance(raw_tree_indices, (str, bytes)):
            raise ValueError("Inlet tree indices must be a collection of integers.")
        for raw_tree_idx in raw_tree_indices:
            tree_idx = int(raw_tree_idx)
            if tree_idx < 0 or tree_idx >= counts[network_id]:
                raise ValueError(
                    f"Invalid inlet tree index {tree_idx} for network {network_id}; "
                    f"expected 0 <= index < {counts[network_id]}."
                )
            normalized[network_id].add(tree_idx)
    return normalized


def _connected_component_tree_indices(tree_connection) -> list[tuple[int, ...]]:
    n_trees = len(getattr(tree_connection, "connected_network", []) or [])
    if n_trees < 2:
        return []
    tree_indices = [(0, 1)]
    tree_indices.extend((tree_idx,) for tree_idx in range(2, n_trees))
    return tree_indices


def _collect_connected_forest_outputs(forest) -> list[_ConnectedSplineOutput]:
    outputs = []
    for fallback_network_idx, tree_connection in enumerate(_tree_connections_by_network(forest).values()):
        network_id = int(getattr(tree_connection, "network_id", fallback_network_idx))
        _, _, _, all_points, all_radii, _ = export_spline(tree_connection)
        component_tree_indices = _connected_component_tree_indices(tree_connection)
        if len(component_tree_indices) != len(all_points):
            raise ValueError(
                "Connected forest spline outputs do not match the expected tree-to-component mapping."
            )
        for component_idx, (tree_indices, points, radii) in enumerate(
            zip(component_tree_indices, all_points, all_radii)
        ):
            outputs.append(
                _ConnectedSplineOutput(
                    network_id=network_id,
                    component_idx=component_idx,
                    tree_indices=tree_indices,
                    all_points=points,
                    all_radii=radii,
                )
            )
    if not outputs:
        raise ValueError("Connected forest has no spline networks to export.")
    return outputs


def _connected_output_paths(base_path: Path, outputs: list[_ConnectedSplineOutput]) -> list[Path]:
    if len(outputs) == 1:
        return [base_path]

    stem = base_path.stem or "splines"
    suffix = base_path.suffix or ".txt"
    counts_by_network = Counter(output.network_id for output in outputs)
    paths = []
    for output in outputs:
        if counts_by_network[output.network_id] == 1:
            name = f"{stem}_network{output.network_id}{suffix}"
        else:
            name = f"{stem}_network{output.network_id}_{output.component_idx}{suffix}"
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


def _tree_root_point(tree) -> np.ndarray:
    data = np.asarray(getattr(tree, "data", None))
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] <= 17:
        raise ValueError("Tree has no root point data for inlet/outlet spline sidecar export.")

    populated = np.any(~np.isnan(data[:, :6]), axis=1)
    root_rows = np.flatnonzero(populated & np.isnan(data[:, 17]))
    if len(root_rows) == 0:
        raise ValueError("Tree root row could not be identified for spline sidecar export.")
    return np.asarray(data[root_rows[0], 0:3], dtype=float)


def _forest_tree_root_role(
    network_id: int,
    tree_idx: int,
    inlet_tree_indices_by_network: dict[int, set[int]] | None,
) -> Literal["inlet", "outlet"]:
    inlet_tree_indices_by_network = inlet_tree_indices_by_network or {}
    if tree_idx in inlet_tree_indices_by_network.get(network_id, set()):
        return "inlet"
    return "outlet"


def _connected_output_root_points(
    tree_connection,
    tree_indices: tuple[int, ...],
    inlet_tree_indices: set[int],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    connected_network = getattr(tree_connection, "connected_network", []) or []
    inlet_points = []
    outlet_points = []
    for tree_idx in tree_indices:
        if tree_idx < 0 or tree_idx >= len(connected_network):
            raise ValueError(
                f"Connected forest component references invalid tree index {tree_idx}."
            )
        point = _tree_root_point(connected_network[tree_idx])
        if tree_idx in inlet_tree_indices:
            inlet_points.append(point)
        else:
            outlet_points.append(point)
    return inlet_points, outlet_points


def _sidecar_path(spline_path: Path) -> Path:
    return spline_path.with_name(f"{spline_path.stem}_inlet_outlet.txt")


def _write_tree_root_sidecar(
    spline_path: Path,
    tree,
    *,
    role: Literal["inlet", "outlet"],
) -> Path:
    root_point = _tree_root_point(tree)
    inlet_points = [root_point] if role == "inlet" else []
    outlet_points = [root_point] if role == "outlet" else []
    return _write_inlet_outlet_sidecar(
        spline_path,
        inlet_points=inlet_points,
        outlet_points=outlet_points,
    )


def _write_inlet_outlet_sidecar(
    spline_path: Path,
    *,
    inlet_points: list[np.ndarray],
    outlet_points: list[np.ndarray],
) -> Path:
    sidecar_path = _sidecar_path(spline_path)
    _ensure_parent(sidecar_path)
    with sidecar_path.open("w", encoding="utf-8") as sidecar_file:
        sidecar_file.write("inlet\n")
        for point in inlet_points:
            sidecar_file.write(", ".join(f"{float(value):.16g}" for value in point))
            sidecar_file.write("\n")
        sidecar_file.write("outlet\n")
        for point in outlet_points:
            sidecar_file.write(", ".join(f"{float(value):.16g}" for value in point))
            sidecar_file.write("\n")
    return sidecar_path
