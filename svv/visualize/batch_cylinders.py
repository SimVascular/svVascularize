"""
Batch cylinder construction for efficient VTK rendering.

Builds all cylinder geometry vectorized in numpy and returns a single
merged pv.PolyData, reducing VTK actors from N to 1 per logical group.
"""

import numpy as np
import pyvista as pv


def make_cylinders_batch(centers, directions, radii, heights, resolution=8):
    """
    Build a single merged PolyData containing all cylinders.

    Parameters
    ----------
    centers : ndarray (n, 3)
    directions : ndarray (n, 3)
    radii : ndarray (n,)
    heights : ndarray (n,)
    resolution : int
        Number of sides per cylinder cross-section.

    Returns
    -------
    pv.PolyData or None
    """
    if len(centers) == 0:
        return None

    centers = np.asarray(centers, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64).ravel()
    heights = np.asarray(heights, dtype=np.float64).ravel()

    # Filter invalid cylinders
    valid = (heights > 0) & np.all(np.isfinite(centers), axis=1)
    if not np.any(valid):
        return None
    centers = centers[valid]
    directions = directions[valid]
    radii = radii[valid]
    heights = heights[valid]

    n = len(centers)
    res = resolution

    # Normalize directions
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    w = directions / norms  # (n, 3)

    # Build local coordinate frames: u, v perpendicular to w
    # Pick reference vector that isn't parallel to w
    ref = np.zeros_like(w)
    ref[:, 0] = 1.0
    # Where w is nearly parallel to x-axis, use y-axis instead
    parallel = np.abs(np.einsum('ij,ij->i', w, ref)) > 0.9
    ref[parallel] = [0.0, 1.0, 0.0]

    u = np.cross(w, ref)
    u_norms = np.linalg.norm(u, axis=1, keepdims=True)
    u_norms = np.where(u_norms < 1e-12, 1.0, u_norms)
    u /= u_norms
    v = np.cross(w, u)  # Already unit length

    # Angles for the ring
    theta = np.linspace(0, 2 * np.pi, res, endpoint=False)  # (res,)
    cos_t = np.cos(theta)  # (res,)
    sin_t = np.sin(theta)  # (res,)

    # Ring offsets in local frame: (res, n, 3)
    # ring_offset[k, i, :] = radii[i] * (cos_t[k] * u[i] + sin_t[k] * v[i])
    ring_offset = (cos_t[:, None, None] * u[None, :, :] +
                   sin_t[:, None, None] * v[None, :, :]) * radii[None, :, None]
    # Shape: (res, n, 3)

    half_h = (heights / 2.0)[:, None] * w  # (n, 3)

    # Bottom ring: center - half_h + ring_offset
    # Top ring: center + half_h + ring_offset
    bottom_center = centers - half_h  # (n, 3)
    top_center = centers + half_h     # (n, 3)

    # Ring vertices: shape (res, n, 3) -> transpose to (n, res, 3)
    bottom_ring = bottom_center[None, :, :] + ring_offset  # (res, n, 3)
    top_ring = top_center[None, :, :] + ring_offset         # (res, n, 3)

    bottom_ring = bottom_ring.transpose(1, 0, 2)  # (n, res, 3)
    top_ring = top_ring.transpose(1, 0, 2)         # (n, res, 3)

    # Layout per cylinder: [bottom_ring(res), top_ring(res), bottom_cap_center, top_cap_center]
    # Total points per cylinder: 2*res + 2
    pts_per_cyl = 2 * res + 2

    all_points = np.empty((n * pts_per_cyl, 3), dtype=np.float64)
    # Bottom ring
    all_points[:n * res] = bottom_ring.reshape(n * res, 3)
    # Top ring
    all_points[n * res:2 * n * res] = top_ring.reshape(n * res, 3)
    # Bottom cap centers
    all_points[2 * n * res:2 * n * res + n] = bottom_center
    # Top cap centers
    all_points[2 * n * res + n:2 * n * res + 2 * n] = top_center

    # Rearrange so each cylinder's points are contiguous
    # Current layout: all bottom rings, all top rings, all bottom caps, all top caps
    # Need: cyl0_bottom_ring, cyl0_top_ring, cyl0_bot_cap, cyl0_top_cap, cyl1_...
    points = np.empty((n * pts_per_cyl, 3), dtype=np.float64)
    for i in range(n):
        base = i * pts_per_cyl
        points[base:base + res] = bottom_ring[i]
        points[base + res:base + 2 * res] = top_ring[i]
        points[base + 2 * res] = bottom_center[i]
        points[base + 2 * res + 1] = top_center[i]

    # Build faces
    # Per cylinder: res side quads + res bottom cap triangles + res top cap triangles
    # Side quad: 4 points each -> 5 ints per face (4, a, b, c, d)
    # Cap triangle: 3 points each -> 4 ints per face (3, a, b, c)
    faces_per_cyl = res * 3  # res side + res bottom + res top
    ints_per_side = 5  # [4, v0, v1, v2, v3]
    ints_per_tri = 4   # [3, v0, v1, v2]
    ints_per_cyl = res * ints_per_side + res * ints_per_tri + res * ints_per_tri

    faces = np.empty(n * ints_per_cyl, dtype=np.int64)

    # Precompute face pattern for cylinder 0, then broadcast
    j_vals = np.arange(res)
    j_next = (j_vals + 1) % res

    # Single cylinder face pattern (base offset = 0)
    # Bottom ring indices: 0..res-1
    # Top ring indices: res..2*res-1
    # Bottom cap center: 2*res
    # Top cap center: 2*res+1

    # Side quads: [4, bottom[j], bottom[j+1], top[j+1], top[j]]
    side_faces = np.empty((res, 5), dtype=np.int64)
    side_faces[:, 0] = 4
    side_faces[:, 1] = j_vals
    side_faces[:, 2] = j_next
    side_faces[:, 3] = res + j_next
    side_faces[:, 4] = res + j_vals

    # Bottom cap triangles: [3, cap_center, bottom[j+1], bottom[j]]
    bot_cap = np.empty((res, 4), dtype=np.int64)
    bot_cap[:, 0] = 3
    bot_cap[:, 1] = 2 * res  # cap center
    bot_cap[:, 2] = j_next
    bot_cap[:, 3] = j_vals

    # Top cap triangles: [3, cap_center, top[j], top[j+1]]
    top_cap = np.empty((res, 4), dtype=np.int64)
    top_cap[:, 0] = 3
    top_cap[:, 1] = 2 * res + 1  # cap center
    top_cap[:, 2] = res + j_vals
    top_cap[:, 3] = res + j_next

    # Concatenate pattern for one cylinder
    pattern = np.concatenate([side_faces.ravel(), bot_cap.ravel(), top_cap.ravel()])

    # Mask for which entries in pattern are vertex indices (not face-size prefixes)
    side_mask = np.ones(res * 5, dtype=bool)
    side_mask[::5] = False  # positions 0, 5, 10, ... are the "4" prefix
    bot_mask = np.ones(res * 4, dtype=bool)
    bot_mask[::4] = False
    top_mask = np.ones(res * 4, dtype=bool)
    top_mask[::4] = False
    vertex_mask = np.concatenate([side_mask, bot_mask, top_mask])

    # Broadcast across all cylinders
    offsets = np.arange(n, dtype=np.int64) * pts_per_cyl  # (n,)
    tiled = np.tile(pattern, n)  # (n * ints_per_cyl,)
    # Add offsets to vertex indices only
    tiled_mask = np.tile(vertex_mask, n)
    offset_array = np.repeat(offsets, ints_per_cyl)
    tiled[tiled_mask] += offset_array[tiled_mask]

    mesh = pv.PolyData(points, tiled)
    return mesh


def tree_to_merged_mesh(tree, resolution=8):
    """
    Convert a Tree's vessel data into a single merged cylinder mesh.

    Parameters
    ----------
    tree : svv.tree.Tree
    resolution : int

    Returns
    -------
    pv.PolyData or None
    """
    data = tree.data
    n = data.shape[0]
    if n == 0:
        return None

    centers = (data[:, 0:3] + data[:, 3:6]) / 2
    directions = data[:, 12:15]  # w_basis
    heights = data[:, 20]        # length
    radii = data[:, 21]          # radius

    return make_cylinders_batch(centers, directions, radii, heights, resolution)


def segments_to_merged_mesh(segments, resolution=8):
    """
    Convert connection vessel segments into a single merged cylinder mesh.

    Parameters
    ----------
    segments : ndarray (n, 7)
        Each row: [x0, y0, z0, x1, y1, z1, radius]
    resolution : int

    Returns
    -------
    pv.PolyData or None
    """
    segments = np.asarray(segments, dtype=np.float64)
    if segments.ndim == 1:
        segments = segments.reshape(1, -1)
    if segments.shape[0] == 0:
        return None

    p0 = segments[:, 0:3]
    p1 = segments[:, 3:6]
    radii = segments[:, 6]

    directions = p1 - p0
    heights = np.linalg.norm(directions, axis=1)
    centers = (p0 + p1) / 2

    return make_cylinders_batch(centers, directions, radii, heights, resolution)
