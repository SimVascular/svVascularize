import numpy

from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def geodesic_constructor(domain, **kwargs):
    """
    Construct a general geodesic function solver for a given domain.

    Builds a sparse graph from the tetrahedral mesh edges and provides
    a function to compute shortest (geodesic) paths between any two
    3D points. Uses Dijkstra with source caching so that repeated
    queries from the same source node avoid redundant computation.

    Parameters
    ----------
    domain : svtoolkit.domain.Domain
        The domain object that defines the spatial region in which vascular
        trees are generated.

    Returns
    -------
    get_geodesic : function
        A function that computes the geodesic path between two points.
    """
    # Build edge list from tetrahedra using vectorized operations
    edge_pairs = numpy.array([[0, 1], [1, 2], [2, 0], [0, 3], [3, 1], [2, 3]])
    tetra = domain.mesh.cells.reshape(-1, 5)[:, 1:]

    # Extract all edges at once: shape (n_tetra * 6, 2)
    all_edges = tetra[:, edge_pairs].reshape(-1, 2)

    # Canonicalize edges (smaller index first) and deduplicate
    sorted_edges = numpy.sort(all_edges, axis=1)
    unique_edges = numpy.unique(sorted_edges, axis=0)

    # Compute lengths for unique edges
    pts = domain.mesh.points
    edge_lengths = numpy.linalg.norm(pts[unique_edges[:, 0]] - pts[unique_edges[:, 1]], axis=1)

    # Build symmetric edge arrays
    rows = numpy.concatenate([unique_edges[:, 0], unique_edges[:, 1]])
    cols = numpy.concatenate([unique_edges[:, 1], unique_edges[:, 0]])
    weights = numpy.concatenate([edge_lengths, edge_lengths])

    n_nodes = pts.shape[0]
    graph = csr_matrix((weights, (rows, cols)), shape=(n_nodes, n_nodes))

    tetra_node_tree = cKDTree(pts)

    # Cache Dijkstra results per source node to avoid recomputation
    _dijkstra_cache = {}

    def get_path(start, end):
        if start not in _dijkstra_cache:
            dist, pred = shortest_path(csgraph=graph, directed=False,
                                       indices=start, return_predecessors=True)
            _dijkstra_cache[start] = (dist, pred)
        dist, pred = _dijkstra_cache[start]

        path = [end]
        dists = []
        k = end
        while pred[k] != -9999:
            path.append(pred[k])
            dists.append(dist[k])
            k = pred[k]
        path = path[::-1]
        dists = dists[::-1]
        lines = [[path[i], path[i + 1]] for i in range(len(path) - 1)]
        return path, dists, lines

    def get_geodesic(start, end, tetra_node_tree=tetra_node_tree, get_path=get_path):
        """
        Get the geodesic path between two points.

        Parameters
        ----------
        start : numpy.ndarray
            The starting point (3D coordinates).
        end : numpy.ndarray
            The ending point (3D coordinates).

        Returns
        -------
        path : list
            The list of nodes (mesh indices) in the path.
        dists : list
            The list of distances between nodes.
        lines : list
            The list of lines between nodes.
        """
        ind = tetra_node_tree.query(start)[1]
        jnd = tetra_node_tree.query(end)[1]
        path, dists, lines = get_path(ind, jnd)
        return path, dists, lines

    return get_geodesic

