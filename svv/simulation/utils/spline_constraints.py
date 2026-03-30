import numpy as np


def _append_splines(container, item):
    if item is None:
        return
    if callable(item):
        container.append(item)
        return
    if isinstance(item, (list, tuple)):
        for child in item:
            _append_splines(container, child)
        return
    raise TypeError(f"Unsupported spline container element: {type(item)!r}")


def flatten_spline_functions(splines):
    """Return a flat list of callable spline evaluators."""
    flat = []
    _append_splines(flat, splines)
    return flat


def sample_spline_functions(splines, spline_sample_points=100):
    """Sample spline callables into point and line constraints."""
    if spline_sample_points < 2:
        raise ValueError("spline_sample_points must be at least 2.")

    flat_splines = flatten_spline_functions(splines)
    if not flat_splines:
        return {
            "points": np.empty((0, 3), dtype=float),
            "lines": np.empty((0, 2), dtype=np.int64),
            "spline_id": np.empty((0,), dtype=np.int32),
            "spline_order": np.empty((0,), dtype=np.int32),
            "radius": np.empty((0,), dtype=float),
        }

    all_points = []
    all_lines = []
    all_spline_ids = []
    all_spline_orders = []
    all_radii = []
    point_offset = 0

    for spline_id, spline in enumerate(flat_splines):
        t = np.linspace(0.0, 1.0, num=spline_sample_points)
        data = spline(t)
        coords = np.column_stack(
            (
                np.asarray(data[0], dtype=float),
                np.asarray(data[1], dtype=float),
                np.asarray(data[2], dtype=float),
            )
        )
        radii = np.asarray(data[3], dtype=float).reshape(-1)
        if coords.shape[0] != radii.shape[0]:
            raise ValueError("Spline evaluation returned inconsistent point and radius counts.")
        if coords.shape[0] < 2:
            raise ValueError("Each spline must yield at least two sample points.")

        all_points.append(coords)
        all_radii.append(radii)
        all_spline_ids.append(np.full(coords.shape[0], spline_id, dtype=np.int32))
        all_spline_orders.append(np.arange(coords.shape[0], dtype=np.int32))
        all_lines.append(
            np.column_stack(
                (
                    np.arange(point_offset, point_offset + coords.shape[0] - 1, dtype=np.int64),
                    np.arange(point_offset + 1, point_offset + coords.shape[0], dtype=np.int64),
                )
            )
        )
        point_offset += coords.shape[0]

    return {
        "points": np.vstack(all_points),
        "lines": np.vstack(all_lines),
        "spline_id": np.concatenate(all_spline_ids),
        "spline_order": np.concatenate(all_spline_orders),
        "radius": np.concatenate(all_radii),
    }


def deduplicate_spline_constraints(points, lines=None, spline_id=None, spline_order=None, radius=None, tol=1e-8):
    """Deduplicate sampled spline points and remap line connectivity."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    lines = np.empty((0, 2), dtype=np.int64) if lines is None else np.asarray(lines, dtype=np.int64).reshape(-1, 2)
    spline_id = np.full(points.shape[0], -1, dtype=np.int32) if spline_id is None else np.asarray(spline_id, dtype=np.int32).reshape(-1)
    spline_order = np.full(points.shape[0], -1, dtype=np.int32) if spline_order is None else np.asarray(spline_order, dtype=np.int32).reshape(-1)
    radius = np.full(points.shape[0], np.nan, dtype=float) if radius is None else np.asarray(radius, dtype=float).reshape(-1)

    if points.shape[0] == 0:
        return {
            "points": points,
            "lines": lines,
            "spline_id": spline_id,
            "spline_order": spline_order,
            "radius": radius,
        }

    old_to_new = np.empty(points.shape[0], dtype=np.int64)
    key_to_new = {}
    unique_points = []
    unique_spline_id = []
    unique_spline_order = []
    unique_radius = []

    for idx, point in enumerate(points):
        if tol > 0.0:
            key = tuple(np.round(point / tol).astype(np.int64).tolist())
        else:
            key = tuple(point.tolist())
        mapped = key_to_new.get(key)
        if mapped is None:
            mapped = len(unique_points)
            key_to_new[key] = mapped
            unique_points.append(point)
            unique_spline_id.append(int(spline_id[idx]))
            unique_spline_order.append(int(spline_order[idx]))
            unique_radius.append(float(radius[idx]))
        old_to_new[idx] = mapped

    remapped_lines = []
    seen = set()
    for line in lines:
        start = int(old_to_new[int(line[0])])
        end = int(old_to_new[int(line[1])])
        if start == end:
            continue
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        remapped_lines.append([start, end])

    return {
        "points": np.asarray(unique_points, dtype=float),
        "lines": np.asarray(remapped_lines, dtype=np.int64).reshape(-1, 2),
        "spline_id": np.asarray(unique_spline_id, dtype=np.int32),
        "spline_order": np.asarray(unique_spline_order, dtype=np.int32),
        "radius": np.asarray(unique_radius, dtype=float),
    }
