import numpy

from tqdm import trange, tqdm
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import splprep, splev, interp1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


def _make_linear_interp(point_a, point_b):
    """Create a linear interpolation function between two 3D points."""
    def func(t_, pa=point_a.copy(), pb=point_b.copy()):
        t_ = numpy.atleast_1d(t_)
        return (1.0 - t_)[:, None] * pa + t_[:, None] * pb
    return func


def _make_geodesic_interp(path_pts):
    """Create an interpolation function along a geodesic path."""
    t = numpy.linspace(0, 1, path_pts.shape[0])
    xpts = interp1d(t, path_pts[:, 0])
    ypts = interp1d(t, path_pts[:, 1])
    zpts = interp1d(t, path_pts[:, 2])
    def func(t_, xpts=xpts, ypts=ypts, zpts=zpts):
        return numpy.array([xpts(t_), ypts(t_), zpts(t_)]).T
    return func


def _compute_path_and_func(forest, pt_a, pt_b, convex):
    """Compute the interpolation function and distance for a terminal pair.

    For convex domains, uses a straight line. For non-convex domains,
    checks if the straight line stays inside the domain and falls back
    to geodesic pathfinding if not.

    Returns
    -------
    func : callable
        Interpolation function mapping t in [0,1] to 3D points.
    dist : float
        Total path distance.
    """
    if convex:
        func = _make_linear_interp(pt_a, pt_b)
        dist = float(numpy.linalg.norm(pt_b - pt_a))
        return func, dist

    # Non-convex: check if straight line stays inside domain
    path_pts = numpy.vstack((pt_a, pt_b))
    func = _make_linear_interp(pt_a, pt_b)
    sample_pts = func(numpy.linspace(0, 1, 10))
    values = forest.domain(sample_pts)
    dists = numpy.linalg.norm(numpy.diff(sample_pts, axis=0), axis=1)

    if numpy.any(values > 0):
        # Path exits domain - use geodesic
        path, dists, _ = forest.geodesic(pt_a, pt_b)
        path_pts = forest.domain.mesh.points[path, :]
        func = _make_geodesic_interp(path_pts)

    return func, float(numpy.sum(dists))


def assign_network(forest, *args, **kwargs):
    """
    Assign the terminal connections among tree objects within a
    forest network. The assignment is based on the minimum distance
    between terminal points of the tree.

    Parameters
    ----------
    forest : svtoolkit.forest.Forest
        A forest object that contains a collection of trees.
    args : int (optional)
        The index of the network to be assigned.

    Returns
    -------
    network_assignments : list of list of int
        A list of terminal indices for each tree in the network.
    network_connections : list of list of functions
        A list of functions that define the connection between
        terminal points of the trees in the network.
    """
    network_connections = []
    network_assignments = []
    t = kwargs.get('t', 0.5)
    show = kwargs.get('show', False)
    network_id = args[0] if len(args) > 0 else 0

    neighbors = kwargs.get('neighbors', int(t * numpy.sum(
        numpy.all(numpy.isnan(forest.networks[network_id][0].data[:, 15:17]), axis=1))))

    if forest.n_trees_per_network[network_id] < 2:
        return network_assignments, network_connections

    tree_0 = forest.networks[network_id][0].data
    tree_1 = forest.networks[network_id][1].data
    idx_0 = numpy.argwhere(numpy.all(numpy.isnan(tree_0[:, 15:17]), axis=1)).flatten()
    idx_1 = numpy.argwhere(numpy.all(numpy.isnan(tree_1[:, 15:17]), axis=1)).flatten()
    terminals_0_ind = idx_0
    terminals_0_pts = tree_0[idx_0, 3:6]
    terminals_0_tree = cKDTree(terminals_0_pts)
    terminals_1_ind = idx_1
    terminals_1_pts = tree_1[idx_1, 3:6]
    terminals_1_tree = cKDTree(terminals_1_pts)
    neighbors = min(neighbors, terminals_0_pts.shape[0], terminals_1_pts.shape[0])

    rows = numpy.repeat(numpy.arange(terminals_0_pts.shape[0]), neighbors)
    cols = numpy.repeat(numpy.arange(terminals_1_pts.shape[0]), neighbors)
    network_assignments.append(terminals_0_ind.tolist())

    # Query k-nearest neighbors bidirectionally
    dists_1, idxs_1 = terminals_1_tree.query(terminals_0_pts, k=neighbors)
    dists_0, idxs_0 = terminals_0_tree.query(terminals_1_pts, k=neighbors)

    all_rows = numpy.concatenate([rows, idxs_0.flatten()])
    all_cols = numpy.concatenate([idxs_1.flatten(), cols])

    # Deduplicate (i, j) pairs to avoid redundant geodesic computation
    pairs = numpy.column_stack([all_rows, all_cols])
    unique_pairs, inverse_idx = numpy.unique(pairs, axis=0, return_inverse=True)

    # Compute path and function for each unique pair
    unique_funcs = []
    unique_dists = []
    desc = 'Computing paths' if not forest.convex else 'Setting up paths'
    for k in tqdm(range(unique_pairs.shape[0]), desc=desc, leave=False):
        i, j = unique_pairs[k]
        func, dist = _compute_path_and_func(
            forest, terminals_0_pts[i], terminals_1_pts[j], forest.convex)
        unique_funcs.append(func)
        unique_dists.append(dist)

    # Map back to full arrays
    all_data = numpy.array(unique_dists)[inverse_idx]
    function_data = [unique_funcs[inv] for inv in inverse_idx]

    # Build sparse cost matrix and function lookup
    C_sparse = coo_matrix(
        (all_data, (all_rows, all_cols)),
        shape=(terminals_0_pts.shape[0], terminals_1_pts.shape[0])
    )
    M_sparse = {}
    for i, j, func in zip(all_rows, all_cols, function_data):
        key = (int(i), int(j))
        if key not in M_sparse:
            M_sparse[key] = func

    # Solve optimal assignment
    try:
        row_ind, col_ind = min_weight_full_bipartite_matching(C_sparse)
    except Exception:
        print("ERROR: Could not find optimal assignment. Try increasing the number of neighbors allowed in search.")
        return None, None

    midpoints = []
    for i, j in zip(row_ind, col_ind):
        m_val = M_sparse.get((int(i), int(j)))
        if m_val is None:
            print("ERROR: Missing function for assignment pair ({}, {})".format(i, j))
        midpoints.append(m_val)

    network_assignments.append(terminals_1_ind[col_ind].tolist())
    network_connections.append(midpoints)

    return network_assignments, network_connections


def assign_network_vector(forest, network_id, midpoints, **kwargs):
    """
    Assign the terminal connections among tree objects within a
    forest network for trees beyond the first two. These additional
    trees connect to the midpoints of the first two trees' connections.

    Parameters
    ----------
    forest : svtoolkit.forest.Forest
        A forest object that contains a collection of trees.
    network_id : int
        The index of the network to be assigned.
    midpoints : numpy.ndarray
        Midpoint positions from tree 0-1 connections.

    Returns
    -------
    network_assignments : list of list of int
        A list of terminal indices for each additional tree.
    network_connections : list of list of functions
        A list of connection functions for each additional tree.
    """
    network_connections = []
    network_assignments = []
    neighbors = kwargs.get('neighbors', 5)

    if forest.n_trees_per_network[network_id] <= 2:
        return network_assignments, network_connections

    for N in range(2, forest.n_trees_per_network[network_id]):
        tree_n = forest.networks[network_id][N].data
        idx_n = numpy.argwhere(numpy.all(numpy.isnan(tree_n[:, 15:17]), axis=1)).flatten()
        terminals_n_ind = idx_n
        terminals_n_pts = tree_n[idx_n, 3:6]
        neighbors = min(neighbors, midpoints.shape[0], terminals_n_pts.shape[0])
        terminals_n_tree = cKDTree(terminals_n_pts)
        midpoints_tree = cKDTree(midpoints)

        C = numpy.full((midpoints.shape[0], terminals_n_pts.shape[0]), 1e8)
        # Use proper 2D list initialization (avoid shared references)
        MN = [[None for _ in range(terminals_n_pts.shape[0])] for _ in range(midpoints.shape[0])]

        if forest.convex:
            dists_1, idxs_1 = midpoints_tree.query(terminals_n_pts, k=neighbors)
            dists_0, idxs_0 = terminals_n_tree.query(midpoints, k=neighbors)
            rows = numpy.repeat(numpy.arange(terminals_n_pts.shape[0]), neighbors)
            cols = numpy.repeat(numpy.arange(midpoints.shape[0]), neighbors)
            C[cols, idxs_1.flatten()] = dists_1.flatten()
            C[idxs_0.flatten(), rows] = dists_0.flatten()

            # Collect unique pairs and compute functions
            seen = set()
            for i in range(midpoints.shape[0]):
                for j in range(terminals_n_pts.shape[0]):
                    if (i, j) in seen:
                        continue
                    seen.add((i, j))
                    func = _make_linear_interp(terminals_n_pts[j], midpoints[i])
                    MN[i][j] = func
        else:
            dists_1, idxs_1 = midpoints_tree.query(terminals_n_pts, k=neighbors)
            dists_0, idxs_0 = terminals_n_tree.query(midpoints, k=neighbors)

            # Process neighbors of terminals -> midpoints
            for i in trange(idxs_1.shape[0], desc='Calculating geodesics I', leave=False):
                for j in range(idxs_1.shape[1]):
                    mid_idx = idxs_1[i, j]
                    func, dist = _compute_path_and_func(
                        forest, terminals_n_pts[i], midpoints[mid_idx], forest.convex)
                    MN[mid_idx][i] = func  # note: MN is indexed [midpoint][terminal]
                    C[mid_idx, i] = dist

            # Process neighbors of midpoints -> terminals
            for i in trange(idxs_0.shape[0], desc='Calculating geodesics II', leave=False):
                for j in range(idxs_0.shape[1]):
                    term_idx = idxs_0[i, j]
                    if MN[i][term_idx] is not None:
                        continue  # Already computed
                    func, dist = _compute_path_and_func(
                        forest, terminals_n_pts[term_idx], midpoints[i], forest.convex)
                    MN[i][term_idx] = func
                    C[i, term_idx] = dist

        _, assignment = linear_sum_assignment(C)
        midpoints_n = [MN[i][j] for i, j in enumerate(assignment)]
        network_assignments.append(terminals_n_ind[assignment].tolist())
        network_connections.extend([midpoints_n])

    return network_assignments, network_connections


def assign(forest, **kwargs):
    """
    Assign the terminal connections among tree objects within all
    forest networks.

    Parameters
    ----------
    forest : svtoolkit.forest.Forest
        A forest object that contains a collection of trees.

    Returns
    -------
    assignments : list of list of list of int
        A list of terminal indices for each tree in each network.
    connections : list of list of list of functions
        A list of functions that define the connection between
        terminal points of the trees in each network.
    """
    assignments = []
    connections = []
    for i in range(forest.n_networks):
        network_assignments, network_connections = assign_network(forest, i, **kwargs)
        assignments.append(network_assignments)
        connections.append(network_connections)
    return assignments, connections


def geodesic(path_pts, start=None, end=None):
    ctrl_pts = numpy.vstack((start, path_pts, end))
    if path_pts.shape[0] > 3:
        k = 3
    elif path_pts.shape[0] > 2:
        k = 2
    else:
        k = 1
    tck = splprep(path_pts.T, s=0, k=k)
    geo_func = lambda t: numpy.array(splev(t, tck[0])).T
    return geo_func


def geodesic_cost(data, curve_generator=None, boundary_func=None, sample=20):
    curve = curve_generator(data.reshape(-1, 3))
    t = numpy.linspace(0, 1, sample)
    pts = curve(t)
    length = numpy.sum(numpy.linalg.norm(numpy.diff(pts, axis=0), axis=1))
    values = boundary_func(pts)
    values = numpy.exp(numpy.sum(values[values > 0]))
    cost = length * values
    return cost
