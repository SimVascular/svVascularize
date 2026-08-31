import numpy
import os
import tempfile
import multiprocessing as mp
import pyvista
import pymeshfix
from tqdm import trange, tqdm
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from svv.utils.remeshing.remesh import (
    _prepare_surface_for_mmgs,
    remesh_surface,
    write_medit_sol,
)
from svv.domain.routines.boolean import boolean


def find_unionable_pairs(lines, tubes, method='centerline', bbox_pad_factor=0.1, eps_factor=0.05):
    """
    Determine all tube index pairs (i, m) such that the initial point of the
    polyline `lines[i]` lies inside tube `tubes[m]`. These pairs are safe
    candidates for boolean unioning in a topology-aware manner (child into parent).

    Parameters
    ----------
    lines : list[pyvista.PolyData]
        Polylines with a single line cell each. Must have point array 'radius'.
    tubes : list[pyvista.PolyData]
        Triangulated surface meshes for each corresponding line (tube geometry).
    method : str, optional
        How to test containment. One of:
          - 'centerline' (default): project the line's initial point onto each
            candidate parent's polyline and compare distance to interpolated
            radius at the closest segment. Fast and robust to minor mesh issues.
          - 'surface': use VTK's select_enclosed_points against the tube surface
            to check if the point is inside. More expensive but purely geometric.
    bbox_pad_factor : float, optional
        Fraction of a tube's characteristic radius used to pad its axis-aligned
        bounding box for a quick inclusion prefilter. Default 0.1.
    eps_factor : float, optional
        Tolerance expressed as a fraction of the parent's median segment length
        for the centerline method. Default 0.05.

    Returns
    -------
    pairs : list[tuple[int, int]]
        List of (child_index, parent_index) pairs deemed unionable.

    Notes
    -----
    - This function assumes `len(lines) == len(tubes)` and that each line[i]
      corresponds to tube[i]. Pairs with identical indices (i == m) are skipped.
    - The 'centerline' method does not require watertight tubes and is typically
      much faster. The 'surface' method can be used when geometric certainty is
      preferred over speed.
    """
    import numpy as _np
    import pyvista as _pv

    assert len(lines) == len(tubes), "lines and tubes must have the same length"
    n = len(lines)

    # Precompute tube AABBs with padding and per-line helpers
    bounds = []
    pad = _np.zeros((n,), dtype=float)
    line_points = []
    line_radii = []
    line_seg_vecs = []
    line_seg_len2 = []
    line_seg_len = []
    line_med_seg = _np.zeros((n,), dtype=float)

    for i in range(n):
        l = lines[i]
        t = tubes[i]
        # Lines are expected to carry a 'radius' point array
        radii = _np.asarray(l['radius']).reshape(-1)
        pts = _np.asarray(l.points)
        line_points.append(pts)
        line_radii.append(radii)
        # Segment vectors and lengths for centerline projection
        if pts.shape[0] >= 2:
            vecs = pts[1:] - pts[:-1]
            seg_len2 = _np.einsum('ij,ij->i', vecs, vecs)
            seg_len = _np.sqrt(seg_len2)
            med = _np.median(seg_len) if seg_len.size > 0 else 0.0
        else:
            vecs = _np.zeros((0, 3))
            seg_len2 = _np.zeros((0,))
            seg_len = _np.zeros((0,))
            med = 0.0
        line_seg_vecs.append(vecs)
        line_seg_len2.append(seg_len2)
        line_seg_len.append(seg_len)
        line_med_seg[i] = med
        # Tube bounds and padding derived from characteristic radius
        b = t.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        bounds.append(b)
        r_char = float(_np.median(radii)) if radii.size else 0.0
        pad[i] = bbox_pad_factor * r_char

    def _in_padded_bounds(p, b, pad_val):
        return (b[0]-pad_val <= p[0] <= b[1]+pad_val and
                b[2]-pad_val <= p[1] <= b[3]+pad_val and
                b[4]-pad_val <= p[2] <= b[5]+pad_val)

    def _closest_distance_to_polyline(point, pts, vecs, seg_len2, radii):
        """Return (d_min, j_best, t_best, r_interp) for a point-to-polyline query."""
        if pts.shape[0] == 0:
            return _np.inf, -1, 0.0, 0.0
        if pts.shape[0] == 1:
            d = _np.linalg.norm(point - pts[0])
            return d, 0, 0.0, float(radii[0]) if radii.size else 0.0
        d_min = _np.inf
        j_best = -1
        t_best = 0.0
        r_best = 0.0
        p = point
        p1 = pts[:-1]
        v = vecs
        w = p - p1
        # Parametric projection along each segment
        with _np.errstate(invalid='ignore', divide='ignore'):
            t = _np.einsum('ij,ij->i', w, v) / seg_len2
        t = _np.clip(t, 0.0, 1.0, out=t)
        proj = p1 + (t[:, None] * v)
        diff = p - proj
        dists = _np.linalg.norm(diff, axis=1)
        if dists.size > 0:
            j = int(_np.argmin(dists))
            d_min = float(dists[j])
            j_best = j
            t_best = float(t[j])
            # Linear interpolate radius on segment
            if radii.size >= 2:
                r1 = float(radii[j])
                r2 = float(radii[j+1])
                r_best = (1.0 - t_best) * r1 + t_best * r2
            else:
                r_best = float(radii[0]) if radii.size else 0.0
        return d_min, j_best, t_best, r_best

    pairs = []
    # Build a single-point PolyData once for surface checks (reused per candidate)
    for i in range(n):
        # Initial point of child line i
        p0 = _np.asarray(lines[i].points[0])
        # Evaluate against all other tubes with quick AABB prefilter
        for m in range(n):
            if m == i:
                continue
            if not _in_padded_bounds(p0, bounds[m], pad[m]):
                continue
            if method == 'surface':
                # Use VTK select_enclosed_points
                pt_poly = _pv.PolyData(p0.reshape(1, 3))
                classified = tubes[m].select_enclosed_points(pt_poly, tolerance=0.0)
                # PyVista may name the array 'SelectedPoints' or 'Selected'
                if 'SelectedPoints' in classified.point_data:
                    inside = bool(int(classified.point_data['SelectedPoints'][0]))
                elif 'Selected' in classified.point_data:
                    inside = bool(int(classified.point_data['Selected'][0]))
                else:
                    inside = False
                if inside:
                    pairs.append((i, m))
            else:
                # Centerline projection test
                d_min, j_best, t_best, r_interp = _closest_distance_to_polyline(
                    p0, line_points[m], line_seg_vecs[m], line_seg_len2[m], line_radii[m]
                )
                # Tolerance scaled by parent's median segment length
                eps = eps_factor * (line_med_seg[m] if line_med_seg[m] > 0 else 1.0)
                if d_min <= (r_interp + eps):
                    pairs.append((i, m))

    return pairs

def get_longest_path(data, seed_edge):
    dig = True
    temp_edges = [seed_edge]
    while dig:
        keep_digging = []
        for edge in temp_edges:
            if not numpy.isnan(data[edge, 15]):
                temp_edges.extend([int(data[edge, 15])])
                if not numpy.isnan(data[edge, 16]):
                    temp_edges.extend([int(data[edge, 16])])
                temp_edges.remove(edge)
                keep_digging.append(True)
            else:
                keep_digging.append(False)
        dig = any(keep_digging)
    if len(temp_edges) == 1:
        return temp_edges
    edge_depths = []
    for edge in temp_edges:
        edge_depths.append(data[edge, 26])
    max_depth = max(edge_depths)
    max_edge_depths = [i for i, j in enumerate(edge_depths) if j == max_depth]
    paths = [[temp_edges[i]] for i in max_edge_depths]
    path_lengths = [data[i[0], 20] for i in paths]
    retrace = [True]*len(paths)
    while any(retrace):
        for i, path in enumerate(paths):
            if retrace[i]:
                path.insert(0, int(data[path[0], 17]))
                path_lengths[i] += data[path[0], 20]
                if paths[i][0] == seed_edge:
                    retrace[i] = False
    return paths[path_lengths.index(max(path_lengths))]


def get_alternate_path(data, seed_edge, reference=None):
    if reference is None:
        reference = get_longest_path(data, seed_edge)
    else:
        pass
    seed_edge_idx = reference.index(seed_edge)
    if seed_edge_idx == len(reference)-1:
        print('Seed edge is terminal and no alternative is possible. \n Computation finished.')
        return None
    else:
        children = [int(data[seed_edge, 15]), int(data[seed_edge, 16])]
        if children[0] not in reference:
            alternate_path = get_longest_path(data, children[0])
        else:
            alternate_path = get_longest_path(data, children[1])
    alternate_path.insert(0, seed_edge)
    return alternate_path


def get_branches(data):
    branches = []
    seed_edge = 0
    path = get_longest_path(data, seed_edge)
    branches.append(path)
    upcoming_evaluations = []
    upcoming_evaluations.extend(path[:-1])
    counter = [len(path[:-1])]
    idx = 0
    while len(upcoming_evaluations) > 0:
        if not numpy.isnan(data[upcoming_evaluations[-1], 15]) and not numpy.isnan(data[upcoming_evaluations[-1], 16]):
            pass
        else:
            upcoming_evaluations.pop(-1)
            counter[idx] -= 1
            if counter[idx] == 0:
                counter[idx] = None
                for i in reversed(range(len(counter))):
                    if counter[i] is not None:
                        idx = i
                        break
            continue
        path = get_alternate_path(data, upcoming_evaluations.pop(-1), reference=branches[idx])
        counter[idx] -= 1
        if counter[idx] == 0:
            counter[idx] = None
            for i in reversed(range(len(counter))):
                if counter[i] is not None:
                    idx = i
                    break
        branches.append(path)
        if len(path) > 2:
            upcoming_evaluations.extend(path[1:-1])
            counter.append(len(path[1:-1]))
            idx = len(counter) - 1
        else:
            counter.append(None)
    return branches


def get_points(data, branches):
    path_points = []
    primed = False
    for path in branches:
        branch_points = []
        for edge in reversed(path):
            if edge == path[0] and primed:
                branch_points.insert(0, data[edge, 3:6].tolist())
            elif edge == path[0] and edge == 0 and not primed:
                branch_points.insert(0, data[edge, 3:6].tolist())
                three_fourths = data[edge, 0:3]*(1/4) + data[edge, 3:6]*(3/4)
                branch_points.insert(0, three_fourths.tolist())
                mid_point = (data[edge, 0:3] + data[edge, 3:6])/2
                branch_points.insert(0, mid_point.tolist())
                one_fourth = data[edge, 0:3]*(3/4) + data[edge, 3:6]*(1/4)
                branch_points.insert(0, one_fourth.tolist())
                branch_points.insert(0, data[edge, 0:3].tolist())
                primed = True
            elif len(branch_points) == 0:
                branch_points.insert(0, data[edge, 3:6].tolist())
                three_fourths = data[edge, 0:3]*(1/4) + data[edge, 3:6]*(3/4)
                branch_points.insert(0, three_fourths.tolist())
                mid_point = (data[edge, 0:3] + data[edge, 3:6])/2
                branch_points.insert(0, mid_point.tolist())
                one_fourth = data[edge, 0:3]*(3/4) + data[edge, 3:6]*(1/4)
                branch_points.insert(0, one_fourth.tolist())
            else:
                branch_points.insert(0, data[edge, 3:6].tolist())
                three_fourths = data[edge, 0:3]*(1/4) + data[edge, 3:6]*(3/4)
                branch_points.insert(0, three_fourths.tolist())
                mid_point = (data[edge, 0:3] + data[edge, 3:6])/2
                branch_points.insert(0, mid_point.tolist())
                one_fourth = data[edge, 0:3]*(3/4) + data[edge, 3:6]*(1/4)
                branch_points.insert(0, one_fourth.tolist())
        path_points.append(branch_points)
    return path_points


def get_radii(data, branches):
    path_radii = []
    primed = False
    for path in branches:
        branch_radii = []
        for edge in reversed(path):
            if edge == path[0] and primed:
                branch_radii.insert(0, data[path[1], 21])
            elif edge == path[0] and edge == 0 and not primed:
                branch_radii.insert(0, data[edge, 21])
                branch_radii.insert(0, data[edge, 21])
                branch_radii.insert(0, data[edge, 21])
                branch_radii.insert(0, data[edge, 21])
                branch_radii.insert(0, data[edge, 21])
                primed = True
            elif len(branch_radii) == 0:
                branch_radii.insert(0, data[edge, 21])
                branch_radii.insert(0, data[edge, 21])
                branch_radii.insert(0, data[edge, 21])
                branch_radii.insert(0, data[edge, 21])
            else:
                branch_radii.insert(0, data[edge, 21])
                branch_radii.insert(0, data[edge, 21])
                branch_radii.insert(0, data[edge, 21])
                branch_radii.insert(0, data[edge, 21])
        path_radii.append(branch_radii)
    return path_radii


def get_normals(data, branches):
    path_normals = []
    primed = False
    for path in branches:
        branch_normals = []
        for edge in reversed(path):
            if edge == path[0] and primed:
                branch_normals.insert(0, data[path[1], 12:15].tolist())
            elif edge == path[0] and edge == 0 and not primed:
                vector_1 = data[edge, 12:15]
                mid_vector = (vector_1 + vector_2)/2
                branch_normals.insert(0, mid_vector.tolist())
                branch_normals.insert(0, data[edge, 12:15].tolist())
                branch_normals.insert(0, data[edge, 12:15].tolist())
                branch_normals.insert(0, data[edge, 12:15].tolist())
                branch_normals.insert(0, data[edge, 12:15].tolist())
                primed = True
            elif len(branch_normals) == 0:
                branch_normals.insert(0, data[edge, 12:15].tolist())
                branch_normals.insert(0, data[edge, 12:15].tolist())
                branch_normals.insert(0, data[edge, 12:15].tolist())
                branch_normals.insert(0, data[edge, 12:15].tolist())
                vector_2 = data[edge, 12:15]
            else:
                vector_1 = data[edge, 12:15]
                mid_vector = (vector_1+vector_2)/2
                branch_normals.insert(0, mid_vector.tolist())
                branch_normals.insert(0, data[edge, 12:15].tolist())
                branch_normals.insert(0, data[edge, 12:15].tolist())
                branch_normals.insert(0, data[edge, 12:15].tolist())
                vector_2 = data[edge, 12:15]
        path_normals.append(branch_normals)
    return path_normals


def get_interpolated_sv_data(data):
    branches = get_branches(data)
    points = get_points(data, branches)
    radii = get_radii(data, branches)
    normals = get_normals(data, branches)
    path_frames = []
    for idx in range(len(branches)):
        frames = []
        for jdx in range(len(points[idx])):
            frame = []
            frame.extend(points[idx][jdx])
            frame.append(radii[idx][jdx])
            frame.extend(normals[idx][jdx])
            frames.append(frame)
        path_frames.append(frames)
    interp_xyz = []
    interp_r = []
    interp_n = []
    interp_xyzr = []
    for idx in range(len(branches)):
        p = numpy.array(points[idx]).T
        r = numpy.array(radii[idx]).T
        n = numpy.array(normals[idx]).T
        if len(points[idx]) == 2:
            interp_xyz.append(splprep(p, k=1, s=0))
            rr = numpy.vstack((interp_xyz[-1][1], r))
            interp_r.append(splprep(rr, k=1, s=0))
            xyzr = numpy.vstack((p, r))
            interp_xyzr.append(splprep(xyzr, k=1, s=0))
            #interp_n.append(splprep(n, k=1, s=0))
        elif len(points[idx]) == 3:
            interp_xyz.append(splprep(p, k=2, s=0))
            rr = numpy.vstack((interp_xyz[-1][1], r))
            interp_r.append(splprep(rr, k=1, s=0))
            xyzr = numpy.vstack((p, r))
            interp_xyzr.append(splprep(xyzr, k=2, s=0))
            #interp_n.append(splprep(n, k=2, s=0))
        else:
            interp_xyz.append(splprep(p, s=0))
            rr = numpy.vstack((interp_xyz[-1][1], r))
            interp_r.append(splprep(rr, k=1, s=0))
            xyzr = numpy.vstack((p, r))
            interp_xyzr.append(splprep(xyzr, s=0))
            #interp_n.append(splprep(n, s=0))
    return interp_xyz, interp_r, interp_n, path_frames, branches, interp_xyzr


def polyline_from_points(pts, r):
    poly = pyvista.PolyData()
    poly.points = pts
    cell = numpy.arange(0, len(pts), dtype=numpy.int_)
    cell = numpy.insert(cell, 0, len(pts))
    poly.lines = cell
    poly['radius'] = r
    return poly


def generate_polylines(xyz, r, num=1000):
    """
    This function generates Pyvista polydata polyline objects from a given set of spline interpolated
    data.

    Parameters
    ----------
    xyz : list
        A list of tuples containing the vector of knots, B-spline coefficients, and the degree of the N-degree
        curve spline
    r : list
        A list of tuples containing the vector of knots, B-spline coefficients, and the degree of the 1-D radial
        spline
    num : int
        The number of points to sample from the spline.

    Returns
    -------
    polylines : list
        A list of Pyvista polydata polyline objects.
    """
    polylines = []
    t = numpy.linspace(0, 1, num)
    for xyz_i, r_i in zip(xyz, r):
        x, y, z = splev(t, xyz_i[0])
        _, radius = splev(t, r_i[0])
        points = numpy.zeros((num, 3))
        points[:, 0] = x
        points[:, 1] = y
        points[:, 2] = z
        polylines.append(polyline_from_points(points, radius))
    return polylines


def generate_tube(polyline, hsize=None, radius_based=True):
    """
    This function generates a tube from a given polyline representing a single vessel of a
    vascular object (tree or forest).

    Parameters
    ----------
    polyline : pyvista.PolyData
        A polyline object representing the centerline of a vessel.
    hsize : float
        The mesh element size for the surface mesh of the vessel. When
        radius_based is True, this acts as the target size at the minimum
        centerline radius and scales proportionally elsewhere.
    radius_based : bool
        If True, writes a per-vertex sizing function (in.sol) proportional to
        the local centerline radius so MMG adapts edge sizes accordingly.

    Returns
    -------
    tube : pyvista.PolyData
        A triangulated tube surface mesh object representing the vessel.
    """
    tube = polyline.tube(radius=min(polyline['radius']), scalars='radius',
                         radius_factor=max(polyline['radius'])/min(polyline['radius']), capping=True)
    tube = tube.triangulate()
    tube = tube.compute_normals(auto_orient_normals=True)
    fix = pymeshfix.MeshFix(tube)
    fix.repair()
    tube = fix.mesh
    tube = tube.compute_normals(auto_orient_normals=True)
    if isinstance(hsize, type(None)):
        hsize = (min(polyline['radius'])*2*numpy.pi)/25
    if radius_based:
        # Per-vertex sizing based on local centerline radius using k-NN from KDTree.
        # Scale so hsize matches the size at the minimum radius; elsewhere scales proportionally.
        poly_pts = numpy.asarray(polyline.points, dtype=float)
        poly_rad = numpy.asarray(polyline['radius'], dtype=float).reshape(-1)
        rmin = float(poly_rad.min()) if poly_rad.size else 1.0
        scale = float(hsize) / rmin if rmin > 0 else float(hsize)

        surf_pts = numpy.asarray(tube.points, dtype=float)
        n_poly = poly_pts.shape[0]
        k_nn = min(4, n_poly) if n_poly > 0 else 1
        if n_poly == 0:
            sizes = numpy.full(surf_pts.shape[0], float(hsize), dtype=float)
        else:
            tree = cKDTree(poly_pts)
            d, idx = tree.query(surf_pts, k=k_nn)
            # Handle k=1 return shape
            if k_nn == 1:
                r_local = poly_rad[numpy.asarray(idx).reshape(-1)].astype(float)
            else:
                d = numpy.asarray(d, dtype=float)
                idx = numpy.asarray(idx, dtype=int)
                # Inverse-distance weights; protect against zero distances
                w = 1.0 / (d + 1e-12)
                w /= w.sum(axis=1, keepdims=True)
                r_neighbors = poly_rad[idx]
                r_local = (w * r_neighbors).sum(axis=1)
            sizes = scale * r_local
        # Ensure strictly positive sizes for MMG
        sizes = numpy.maximum(sizes, numpy.finfo(float).eps)
        tube.point_data['MeshSizingFunction'] = sizes
        # Write MMG sizing file the remesher will pick up in this temp directory
        write_medit_sol(tube, 'in.sol', array_name='MeshSizingFunction', scale=1, default_size=hsize)
    # If using a sizing function, let MMG drive sizes solely from in.sol (omit -hsiz)
    tube = remesh_surface(tube, hsiz=(None if radius_based else hsize), verbosity=0)
    tube = tube.compute_normals(auto_orient_normals=True)
    fix = pymeshfix.MeshFix(tube)
    fix.repair()
    tube = fix.mesh
    tube = tube.compute_normals(auto_orient_normals=True)
    return tube


def generate_tubes(polylines, hsize=None):
    """
    This function generates a list of tube surface meshes from a list of polyline objects representing
    the vessels of a vascular object (tree or forest).

    Parameters
    ----------
    polylines : list
        A list of Pyvista polyline objects representing the centerlines of the vessels.
    hsize : float
        The mesh element size for the surface meshes of the vessels.

    Returns
    -------
    tubes : list
        A list of Pyvista surface mesh objects representing the vessels.
    """
    tubes = []
    for polyline in polylines:
        tube = generate_tube(polyline, hsize=hsize)
        tubes.append(tube)
    return tubes


def _tube_worker(args):
    """Worker that builds a single tube in an isolated temp directory.

    Args
    ----
    args : tuple
        (idx, points, radius, hsize, radius_based)

    Returns
    -------
    tuple
        (idx, points, faces) for reconstructed PolyData.
    """
    idx, pts, rad, hsize, radius_based = args
    old_cwd = os.getcwd()
    # Isolate MMG temp files to avoid collisions between processes
    # As in the tetrahedralizer, prefer TEMP/TMP on Windows to avoid
    # issues when TMPDIR is set to an invalid POSIX-style path.
    tmp_root = None
    if os.name == "nt":
        for env_var in ("TEMP", "TMP"):
            candidate = os.environ.get(env_var)
            if candidate and os.path.isdir(candidate):
                tmp_root = candidate
                break

    with tempfile.TemporaryDirectory(prefix="svv_remesh_", dir=tmp_root) as tmpdir:
        try:
            os.chdir(tmpdir)
            poly = polyline_from_points(numpy.asarray(pts), numpy.asarray(rad))
            tube = generate_tube(poly, hsize=hsize, radius_based=radius_based)
            # Return minimal geometry to avoid pickling VTK objects
            return idx, numpy.asarray(tube.points), numpy.asarray(tube.faces)
        finally:
            os.chdir(old_cwd)


def generate_tubes_parallel(polylines, hsize=None, processes=None, chunksize=1, start_method=None, show_progress=True, radius_based=False):
    """
    Parallel tube generation using multiprocessing. Each tube is built in a
    separate process and in a per-process temporary directory to avoid MMG
    temp-file collisions.

    Parameters
    ----------
    polylines : list[pyvista.PolyData]
        Centerline polylines with point-data array 'radius'.
    hsize : float, optional
        Target surface edge size for remeshing (forwarded to generate_tube).
    processes : int, optional
        Number of worker processes. Defaults to `os.cpu_count()`.
    chunksize : int, optional
        Chunk size for Pool.imap. Default 1.
    start_method : {"spawn","fork","forkserver"}, optional
        Multiprocessing start method. Defaults to 'spawn' when available.
    show_progress : bool, optional
        If True, display a tqdm progress bar.
    radius_based : bool, optional
        If True, build a per-vertex MMG sizing function based on the local
        centerline radius and pass it via in.sol, yielding radius-proportional
        edge sizes. Default False.

    Returns
    -------
    list[pyvista.PolyData]
        Reconstructed tube meshes in the same order as input polylines.
    """
    n = len(polylines)
    if n == 0:
        return []

    # Optional sequential fallback, useful on environments where
    # multiprocessing with the 'spawn' context is problematic
    # (e.g., inline scripts on Windows such as `python - <<` in CI).
    disable_env = os.environ.get("SVV_DISABLE_TUBE_PARALLEL", "")
    disable_parallel = disable_env.strip().lower() in {"1", "true", "yes", "on"}
    if disable_parallel:
        tubes = []
        iterable = polylines
        if show_progress:
            iterable = tqdm(iterable, total=n, desc='Generate tubes ', unit='tube', leave=False)
        for pl in iterable:
            tube = generate_tube(pl, hsize=hsize, radius_based=radius_based)
            tubes.append(tube)
        return tubes

    # Serialize inputs (avoid sending VTK objects across processes)
    tasks = []
    for i, pl in enumerate(polylines):
        pts = numpy.asarray(pl.points)
        if 'radius' not in pl.point_data:
            raise KeyError("Each polyline must have a 'radius' point-data array")
        rad = numpy.asarray(pl['radius']).reshape(-1)
        tasks.append((i, pts, rad, hsize, radius_based))

    # Choose a safe start method to avoid forking VTK state
    if start_method is None:
        try:
            ctx = mp.get_context('spawn')
        except ValueError:
            ctx = mp.get_context()
    else:
        ctx = mp.get_context(start_method)

    tubes_arrays = [None] * n
    with ctx.Pool(processes=processes) as pool:
        iterator = pool.imap(_tube_worker, tasks, chunksize)
        if show_progress:
            iterator = tqdm(iterator, total=n, desc='Generate tubes ', unit='tube', leave=False)
        for idx, pts, faces in iterator:
            tubes_arrays[idx] = (pts, faces)

    # Reconstruct PolyData objects in parent process
    tubes = []
    for i in range(n):
        pts, faces = tubes_arrays[i]
        tube = pyvista.PolyData(pts, faces)
        # Ensure normals are available
        tube = tube.compute_normals(auto_orient_normals=True)
        tubes.append(tube)
    return tubes

def _surface_topology(surface):
    """Return manifold and open-edge state for a polygonal surface."""
    try:
        manifold = bool(surface.is_manifold)
        open_edges = int(surface.n_open_edges)
    except Exception as exc:
        raise RuntimeError("Unable to inspect the tube-union surface topology.") from exc
    return manifold, open_edges


def _prepare_watertight_union_surface(surface, *, step, total_steps):
    """Sanitize and, when needed, repair a watertight union intermediate."""
    context = f"Tube union step {step}/{total_steps}"
    try:
        prepared, _, diagnostics = _prepare_surface_for_mmgs(surface, verbosity=1)
    except Exception as exc:
        raise RuntimeError(f"{context} produced an invalid surface: {exc}") from exc

    manifold, open_edges = _surface_topology(prepared)
    repaired = False
    removed_triangles = diagnostics["invalid_triangle_count"]
    invalid_indices = diagnostics["invalid_triangle_indices"][:10]

    if not manifold or open_edges != 0:
        repaired = True
        before_points = prepared.n_points
        before_triangles = prepared.n_cells
        try:
            fix = pymeshfix.MeshFix(prepared)
            fix.repair(verbose=False)
            prepared, _, repair_diagnostics = _prepare_surface_for_mmgs(
                fix.mesh, verbosity=1
            )
        except Exception as exc:
            raise RuntimeError(
                f"{context} could not repair its boolean result "
                f"(points={before_points}, triangles={before_triangles}, "
                f"removed_invalid_triangles={removed_triangles}, "
                f"invalid_triangle_indices={invalid_indices}, "
                f"manifold={manifold}, open_edges={open_edges})."
            ) from exc
        removed_triangles += repair_diagnostics["invalid_triangle_count"]

    manifold, open_edges = _surface_topology(prepared)
    if prepared.n_points == 0 or prepared.n_cells == 0 or not manifold or open_edges != 0:
        raise RuntimeError(
            f"{context} did not produce a watertight surface "
            f"(points={prepared.n_points}, triangles={prepared.n_cells}, "
            f"removed_invalid_triangles={removed_triangles}, repaired={repaired}, "
            f"manifold={manifold}, open_edges={open_edges})."
        )

    return prepared, {
        "removed_invalid_triangles": removed_triangles,
        "invalid_triangle_indices": invalid_indices,
        "repaired": repaired,
    }


def _union_and_remesh_tube(left, right, *, hsize, step, total_steps, tube_index):
    """Perform one diagnosed, watertight boolean-union and remeshing step."""
    context = f"Tube union step {step}/{total_steps} (adding tube index {tube_index})"
    try:
        model = boolean(left, right, operation='union', fix_mesh=False)
    except Exception as exc:
        raise RuntimeError(f"{context} failed during the boolean operation.") from exc

    model, preparation = _prepare_watertight_union_surface(
        model, step=step, total_steps=total_steps
    )
    model = model.compute_normals(auto_orient_normals=True)

    try:
        model = remesh_surface(model, hsiz=hsize)
    except Exception as exc:
        manifold, open_edges = _surface_topology(model)
        raise RuntimeError(
            f"{context} failed during surface remeshing "
            f"(points={model.n_points}, triangles={model.n_cells}, hsiz={hsize:.15g}, "
            f"removed_invalid_triangles={preparation['removed_invalid_triangles']}, "
            f"invalid_triangle_indices={preparation['invalid_triangle_indices']}, "
            f"repaired={preparation['repaired']}, manifold={manifold}, "
            f"open_edges={open_edges})."
        ) from exc

    model = model.compute_normals(auto_orient_normals=True)
    manifold, open_edges = _surface_topology(model)
    if model.n_points == 0 or model.n_cells == 0 or not manifold or open_edges != 0:
        raise RuntimeError(
            f"{context} remeshing returned a non-watertight surface "
            f"(points={model.n_points}, triangles={model.n_cells}, hsiz={hsize:.15g}, "
            f"manifold={manifold}, open_edges={open_edges})."
        )
    return model


def union_tubes(tubes, lines, cap_resolution=40):
    """
    This function performs iterative boolean union operations on a list of Pyvista polydata surface
    meshes to build one single surface mesh representing the entire vascular object.

    Parameters
    ----------
    tubes : list
        A list of Pyvista polydata objects representing the tubes of the vessels in a vascular object.

    Returns
    -------
    union : pyvista.PolyData
        The result of the boolean union operation.
    """
    if len(tubes) < 2:
        raise ValueError("union_tubes requires at least two tube surfaces")
    if len(lines) != len(tubes):
        raise ValueError("lines and tubes must have the same length")
    if cap_resolution <= 0:
        raise ValueError("cap_resolution must be positive")

    total_steps = len(tubes) - 1
    hsize = (min(min(lines[0]['radius']), min(lines[1]['radius']))*2*numpy.pi)/cap_resolution
    model = _union_and_remesh_tube(
        tubes[0], tubes[1], hsize=hsize, step=1, total_steps=total_steps, tube_index=1
    )
    if len(tubes) > 2:
        for i in range(2, len(tubes)):
            hsize = min(hsize, (min(lines[i]['radius'])*2*numpy.pi)/cap_resolution)
            model = _union_and_remesh_tube(
                model,
                tubes[i],
                hsize=hsize,
                step=i,
                total_steps=total_steps,
                tube_index=i,
            )
    model.cell_data['hsize'] = 0
    model.cell_data['hsize'][0] = hsize
    return model


def build_watertight_solid(tree, cap_resolution=40):
    """
    This function builds a solid surface mesh from a given vascular tree object.
    This mesh should be guaranteed to be watertight and define a closed manifold.

    Parameters
    ----------
    tree : svtoolkit.tree.tree.Tree
        A vascular tree object.

    Returns
    -------
    model : pyvista.PolyData
        A solid surface mesh representing the entire vascular tree.
    """
    xyz, r, _, _, branches, _ = get_interpolated_sv_data(tree.data)
    lines = generate_polylines(xyz, r)
    tubes = generate_tubes(lines)
    model = union_tubes(tubes, lines, cap_resolution=cap_resolution)
    if 'hsize' not in model.cell_data or model.n_cells == 0:
        raise RuntimeError("Watertight tube union did not retain its target surface size.")
    hsize = float(model.cell_data['hsize'][0])
    #tubes = generate_tubes_parallel(lines, radius_based=True)
    #model = union_tubes_balanced(tubes, lines, cap_resolution=cap_resolution)
    # Remove poor quality elements and repair the mesh.
    cell_quality = model.compute_cell_quality(quality_measure='scaled_jacobian')
    quality_values = numpy.asarray(cell_quality.cell_data["CellQuality"])
    keep = numpy.isfinite(quality_values) & (quality_values > 0.1)
    if not numpy.all(keep):
        print("Repairing poor quality elements in the mesh.")
        fix = pymeshfix.MeshFix(model)
        fix.repair(verbose=True)
        model, _ = _prepare_watertight_union_surface(
            fix.mesh,
            step=max(len(tubes) - 1, 1),
            total_steps=max(len(tubes) - 1, 1),
        )
        model = model.compute_normals(auto_orient_normals=True)
    manifold, open_edges = _surface_topology(model)
    if not manifold or open_edges != 0:
        raise RuntimeError(
            "Final vascular solid is not watertight "
            f"(points={model.n_points}, triangles={model.n_cells}, "
            f"manifold={manifold}, open_edges={open_edges})."
        )
    model.cell_data['hsize'] = numpy.zeros(model.n_cells)
    model.cell_data['hsize'][0] = hsize
    return model


def union_tubes_balanced(tubes, lines, cap_resolution=40, method='centerline', engine='manifold', fix_mesh=True):
    """
    Perform topology-aware, compute-balanced boolean unions over a set of vessel tubes.

    The function:
    - Detects unionable parent/child pairs using `find_unionable_pairs` (child's
      initial point inside parent's tube).
    - Schedules unions with a min-heap keyed by current mesh sizes so that the
      smallest unions occur first, keeping intermediates compact.
    - Updates the graph as components merge until each connected component is
      reduced to a single mesh.
    - Applies a final remesh and normal computation once on the result.

    Parameters
    ----------
    tubes : list[pyvista.PolyData]
        Per-vessel tube surface meshes (triangulated).
    lines : list[pyvista.PolyData]
        Per-vessel polylines carrying point-data array 'radius'.
    cap_resolution : int, optional
        Controls target edge size for final remeshing via: hsiz ≈ 2π·min_radius / cap_resolution.
        Default 40.
    method : {'centerline','surface'}, optional
        Containment test used by `find_unionable_pairs`. Default 'centerline' (fast).
    engine : str, optional
        Boolean engine forwarded to trimesh via `boolean`. Default 'manifold'.
    fix_mesh : bool, optional
        Whether to run mesh fixing during boolean operations. Forwarded to `boolean`.
        Default True.

    Returns
    -------
    model : pyvista.PolyData
        The watertight surface representing the union of all tubes, remeshed once
        at a global target size derived from cap_resolution.
    """
    import heapq
    import numpy as _np
    import pyvista as _pv

    n = len(tubes)
    if n == 0:
        return _pv.PolyData()
    if len(lines) != n:
        raise ValueError("lines and tubes must have the same length")

    # Determine all unionable pairs (child, parent)
    pairs = find_unionable_pairs(lines, tubes, method=method)
    if len(pairs) == 0:
        # Fallback: sequential union as last resort
        return union_tubes(tubes, lines, cap_resolution=cap_resolution)

    # Disjoint-set (union-find)
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return ra
        if rank[ra] < rank[rb]:
            parent[ra] = rb
            return rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
            return ra
        else:
            parent[rb] = ra
            rank[ra] += 1
            return ra

    # Component meshes and sizes
    comp_mesh = {i: tubes[i] for i in range(n)}
    comp_size = {i: int(tubes[i].n_cells) for i in range(n)}

    # Build neighbor sets from pairs (treat undirected for scheduling)
    neighbors = {i: set() for i in range(n)}
    for i, m in pairs:
        if i == m:
            continue
        neighbors[i].add(m)
        neighbors[m].add(i)

    # Min-heap of candidate unions keyed by sum of component sizes
    heap = []
    for i, m in pairs:
        ra, rb = find(i), find(m)
        if ra == rb:
            continue
        key = comp_size[ra] + comp_size[rb]
        heapq.heappush(heap, (key, ra, rb))

    # Compute total planned unions for progress: sum(|C|-1) over connected components
    # Build node set from pairs and traverse via neighbors
    nodes = set()
    for i, m in pairs:
        nodes.add(i); nodes.add(m)
    visited = set()
    total_merges = 0
    for v in nodes:
        if v in visited:
            continue
        # BFS/DFS over neighbor graph
        stack = [v]
        comp_count = 0
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            comp_count += 1
            for w in neighbors.get(u, ()):  # undirected traversal
                if w not in visited:
                    stack.append(w)
        if comp_count > 0:
            total_merges += max(comp_count - 1, 0)

    pbar = tqdm(total=total_merges, desc='Union tubes ', unit='union', leave=False)

    # Iteratively union smallest available pairs until components are reduced
    while heap:
        _, a0, b0 = heapq.heappop(heap)
        ra, rb = find(a0), find(b0)
        if ra == rb:
            continue  # stale
        # Perform boolean union between current component meshes
        try:
            merged = boolean(comp_mesh[ra], comp_mesh[rb], operation='union', fix_mesh=fix_mesh, engine=engine)
        except Exception as e:
            # If an error occurs, try with fix_mesh=True as a fallback
            if not fix_mesh:
                merged = boolean(comp_mesh[ra], comp_mesh[rb], operation='union', fix_mesh=True, engine=engine)
            else:
                raise e
        r = union(ra, rb)
        comp_mesh[r] = merged
        comp_size[r] = int(merged.n_cells)
        pbar.update(1)

        # Merge neighbor sets and push new edges against r
        na = neighbors.get(ra, set())
        nb = neighbors.get(rb, set())
        merged_neighbors = set()
        for c in (na | nb):
            rc = find(c)
            if rc != r:
                merged_neighbors.add(rc)
        neighbors[r] = merged_neighbors

        for rc in merged_neighbors:
            key = comp_size[r] + comp_size[find(rc)]
            heapq.heappush(heap, (key, r, rc))

    pbar.close()

    # Collect final components (roots)
    roots = {}
    for i in range(n):
        ri = find(i)
        roots[ri] = comp_mesh[ri]

    # Merge disjoint components without boolean (safe if disjoint)
    if len(roots) == 1:
        model = next(iter(roots.values()))
    else:
        model = _pv.merge(list(roots.values()))

    # Final remesh once using a global target size derived from min radius
    try:
        min_r = min(float(_np.min(lines[i]['radius'])) for i in range(len(lines)) if lines[i].n_points > 0)
        hsize = (2.0 * _np.pi * min_r) / float(cap_resolution)
    except Exception:
        # Fallback heuristic if radii missing
        hsize = None

    #if hsize is not None and hsize > 0:
    #    model = remesh_surface(model, hsiz=hsize, verbosity=0)
    model = model.compute_normals(auto_orient_normals=True)
    #fix = pymeshfix.MeshFix(model)
    #fix.repair()
    #model = fix.mesh
    if hsize is not None:
        model.cell_data['hsize'] = 0
        model.cell_data['hsize'][0] = hsize
    return model


def build_merged_solid(tree):
    """
    This function builds a solid surface mesh from a given vascular tree object.
    This mesh is not guaranteed to be watertight or define a closed manifold.
    Instead, this function may be useful in visualizing tree models and displaying
    reduced-order fluid flow simulations in an efficient manner.

    Parameters
    ----------
    tree : svtoolkit.tree.tree.Tree
        A vascular tree object.

    Returns
    -------
    merged_model : pyvista.PolyData
        A merged surface mesh representing the entire vascular tree.
    """
    vessels = []
    for i in trange(tree.data.shape[0], desc="Building Merged Solid"):
        center = (tree.data[i, 0:3] + tree.data[i, 3:6]) / 2
        direction = tree.data.get('w_basis', i)
        radius = tree.data.get('radius', i)
        length = tree.data.get('length', i)
        assert length > 0, "Length must be positive"
        assert radius > 0, "Radius must be positive"
        vessels.append(pyvista.Cylinder(center=center, direction=direction, radius=radius, height=length))
    merged_model = pyvista.merge(vessels)
    return merged_model
