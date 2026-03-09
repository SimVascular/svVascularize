import numpy as np


def _point_to_segment_distance(px, py, pz, x0, y0, z0, x1, y1, z1) -> float:
    vx, vy, vz = x1 - x0, y1 - y0, z1 - z0
    wx, wy, wz = px - x0, py - y0, pz - z0
    seg_len_sq = vx*vx + vy*vy + vz*vz
    if seg_len_sq < 1e-14:
        dx, dy, dz = px - x0, py - y0, pz - z0
        return float(np.sqrt(dx*dx + dy*dy + dz*dz))
    proj = (wx*vx + wy*vy + wz*vz) / seg_len_sq
    proj = 0.0 if proj < 0.0 else (1.0 if proj > 1.0 else proj)
    cx, cy, cz = x0 + proj*vx, y0 + proj*vy, z0 + proj*vz
    dx, dy, dz = px - cx, py - cy, pz - cz
    return float(np.sqrt(dx*dx + dy*dy + dz*dz))


def _point_to_segment_distance_batch(points, seg_start, seg_end):
    """Vectorized point-to-segment distance for multiple points against one segment."""
    v = seg_end - seg_start
    seg_len_sq = np.dot(v, v)
    if seg_len_sq < 1e-14:
        return np.linalg.norm(points - seg_start, axis=1)
    w = points - seg_start
    proj = np.dot(w, v) / seg_len_sq
    proj = np.clip(proj, 0.0, 1.0)
    closest = seg_start + proj[:, None] * v
    return np.linalg.norm(points - closest, axis=1)


def minimum_segment_distance(data_0: np.ndarray, data_1: np.ndarray) -> np.ndarray:
    """Compute pairwise minimum distances between two sets of line segments.

    Uses fully vectorized numpy operations instead of Python loops.

    Parameters
    ----------
    data_0 : np.ndarray, shape (N, 6)
        First set of segments, columns [x0,y0,z0,x1,y1,z1].
    data_1 : np.ndarray, shape (M, 6)
        Second set of segments.

    Returns
    -------
    np.ndarray, shape (N, M)
        Pairwise minimum distances.
    """
    N = data_0.shape[0]
    M = data_1.shape[0]

    # Extract segment endpoints
    A0 = data_0[:, 0:3]  # (N, 3)
    A1 = data_0[:, 3:6]  # (N, 3)
    B0 = data_1[:, 0:3]  # (M, 3)
    B1 = data_1[:, 3:6]  # (M, 3)

    # Direction vectors
    AB = A1 - A0  # (N, 3)
    CD = B1 - B0  # (M, 3)

    # Squared lengths
    AB_AB = np.sum(AB * AB, axis=1)  # (N,)
    CD_CD = np.sum(CD * CD, axis=1)  # (M,)

    # Broadcast for pairwise computation: (N, 1, 3) and (1, M, 3)
    AB_exp = AB[:, None, :]  # (N, 1, 3)
    CD_exp = CD[None, :, :]  # (1, M, 3)
    A0_exp = A0[:, None, :]  # (N, 1, 3)
    B0_exp = B0[None, :, :]  # (1, M, 3)

    AB_CD = np.sum(AB_exp * CD_exp, axis=2)  # (N, M)
    CA = A0_exp - B0_exp  # (N, M, 3)
    CA_AB = np.sum(CA * AB_exp, axis=2)  # (N, M)
    CA_CD = np.sum(CA * CD_exp, axis=2)  # (N, M)

    AB_AB_exp = AB_AB[:, None]  # (N, 1)
    CD_CD_exp = CD_CD[None, :]  # (1, M)

    denom = AB_AB_exp * CD_CD_exp - AB_CD * AB_CD  # (N, M)

    # General case: skew lines
    t = (AB_CD * CA_CD - CA_AB * CD_CD_exp) / np.where(np.abs(denom) < 1e-14, 1.0, denom)
    s = (AB_AB_exp * CA_CD - AB_CD * CA_AB) / np.where(np.abs(denom) < 1e-14, 1.0, denom)
    t = np.clip(t, 0.0, 1.0)
    s = np.clip(s, 0.0, 1.0)

    P1 = A0_exp + t[:, :, None] * AB_exp  # (N, M, 3)
    P2 = B0_exp + s[:, :, None] * CD_exp  # (N, M, 3)
    general_dist = np.linalg.norm(P1 - P2, axis=2)  # (N, M)

    # Handle degenerate/parallel cases
    is_degen_A = AB_AB < 1e-14  # (N,)
    is_degen_B = CD_CD < 1e-14  # (M,)
    is_parallel = np.abs(denom) < 1e-14  # (N, M)

    # Only compute fallbacks where needed
    needs_fallback = is_parallel | is_degen_A[:, None] | is_degen_B[None, :]

    if np.any(needs_fallback):
        out = general_dist.copy()
        fb_i, fb_j = np.nonzero(needs_fallback)

        for idx in range(len(fb_i)):
            ii, jj = fb_i[idx], fb_j[idx]
            a0, a1 = A0[ii], A1[ii]
            c0, c1 = B0[jj], B1[jj]
            ab_ab_val = AB_AB[ii]
            cd_cd_val = CD_CD[jj]

            if ab_ab_val < 1e-14 and cd_cd_val < 1e-14:
                out[ii, jj] = float(np.linalg.norm(a0 - c0))
            elif ab_ab_val < 1e-14:
                out[ii, jj] = _point_to_segment_distance(*a0, *c0, *c1)
            elif cd_cd_val < 1e-14:
                out[ii, jj] = _point_to_segment_distance(*c0, *a0, *a1)
            else:
                # Parallel: check 4 endpoint-segment distances
                out[ii, jj] = min(
                    _point_to_segment_distance(*a0, *c0, *c1),
                    _point_to_segment_distance(*a1, *c0, *c1),
                    _point_to_segment_distance(*c0, *a0, *a1),
                    _point_to_segment_distance(*c1, *a0, *a1),
                )
        return out
    return general_dist


def minimum_self_segment_distance(data: np.ndarray) -> float:
    """Compute minimum distance between non-adjacent segments in a polyline.

    Uses vectorized pairwise computation with masking instead of
    calling minimum_segment_distance in a loop.
    """
    n = data.shape[0]
    if n < 3:
        return 1e20

    # Compute full pairwise distance matrix at once
    dist_matrix = minimum_segment_distance(data[:, :6], data[:, :6])

    # Mask out self-distances and adjacent segments
    mask = np.ones((n, n), dtype=bool)
    for i in range(n):
        mask[i, i] = False
        if i + 1 < n:
            mask[i, i + 1] = False
            mask[i + 1, i] = False

    valid = dist_matrix[mask]
    if valid.size == 0:
        return 1e20
    return float(np.min(valid))

