import numpy
import os
from copy import deepcopy
from scipy.spatial import cKDTree
from svv.tree.tree import Tree
from svv.tree.data.data import TreeData
from svv.tree.collision.tree_collision import tree_collision
from svv.tree.utils.TreeManager import KDTreeManager, USearchTree
from svv.forest.connect.geodesic import geodesic_constructor
from svv.forest.connect.forest_connection import ForestConnection
from svv.visualize.forest.show import show
from svv.forest.export.export_spline import export_spline, write_splines


class Forest(object):
    def __init__(self, **kwargs):
        """
        The Forest class defines collections of synthetic vascular tree
        structures that are used to abstract the physical representations
        of generated interpenetrating vascular networks.

        Parameters
        ----------
        kwargs : dict
            A dictionary of keyword arguments to be passed.
            Keyword arguments include:
                domain : svtoolkit.domain.Domain
                    A domain object that defines the spatial region in which vascular
                    trees are generated.
                n_networks : int
                    The number of vascular networks to be generated.
                n_trees_per_network : list of int
                    The number of trees to be generated per network.
                start_points : list of numpy.ndarray
                    A list of starting points for the trees.
                directions : list of numpy.ndarray
                    A list of directions for the trees.
                physical_clearance : float
                    The physical clearance of any vessel from the domain and other vessels.
                compete : bool
                    A flag that indicates whether trees should compete for space.
        """
        self.networks = []
        self.connections = None
        self.domain = kwargs.get('domain', None)
        self.n_networks = kwargs.get('n_networks', 1)
        self.n_trees_per_network = kwargs.get('n_trees_per_network', [2])
        self.start_points = kwargs.get('start_points', None)
        self.directions = kwargs.get('directions', None)
        self.physical_clearance = kwargs.get('physical_clearance', 0.0)
        self.compete = kwargs.get('compete', False)
        self.preallocation_step = kwargs.get('preallocation_step', None)
        self.geodesic = None
        self.convex = None
        if isinstance(self.start_points, type(None)):
            self.start_points = [[None for j in range(self.n_trees_per_network[i])] for i in range(self.n_networks)]
        if isinstance(self.directions, type(None)):
            self.directions = [[None for j in range(self.n_trees_per_network[i])] for i in range(self.n_networks)]
        networks = []
        for i in range(self.n_networks):
            network = []
            for j in range(self.n_trees_per_network[i]):
                if self.preallocation_step is not None:
                    tree = Tree(preallocation_step=self.preallocation_step)
                else:
                    tree = Tree()
                tree.physical_clearance = self.physical_clearance
                network.append(tree)
            networks.append(network)
        self.networks = networks

    def set_domain(self, domain, convexity_tolerance=1e-2):
        """
        Set the domain for the forest. The domain should be an implicit
        function of the form f(X) = scalar, where X is a cartesian
        coordinate in space.

        Parameters
        ----------
        domain : svtoolkit.domain.Domain
            A domain object that defines the spatial region in which vascular
            trees are generated.
        convexity_tolerance : float
            The tolerance for the convexity of the domain.
        """
        self.domain = domain

        # Determine convexity when available; when missing, treat as effectively
        # convex so algorithms fall back to simpler distance-based paths.
        domain_convexity = getattr(domain, "convexity", None)
        if domain_convexity is not None:
            self.convex = numpy.isclose(domain_convexity, 1.0, atol=convexity_tolerance)
        else:
            self.convex = True

        # Geodesic paths require an interior tetrahedral mesh. When the Domain
        # was loaded from a .dmn file without mesh data, or when tetrahedralize
        # previously failed, `domain.mesh` will be None. In that case we skip
        # geodesic construction and rely on the convex-distance paths.
        self.geodesic = None
        if getattr(domain, "mesh", None) is not None:
            try:
                self.geodesic = geodesic_constructor(domain)
            except Exception:
                # If geodesic construction fails for any reason, fall back to
                # convex handling so forest generation can still proceed.
                self.geodesic = None
                self.convex = True

        for network in self.networks:
            for tree in network:
                tree.set_domain(domain, convexity_tolerance)
                # Trees loaded from disk do not persist the sampling probability
                # distribution because it depends on the (unsaved) domain mesh.
                # When a mesh is available, seed each tree with the current
                # domain probabilities so subsequent growth can proceed.
                if getattr(tree, "probability", None) is None and getattr(domain, "mesh", None) is not None:
                    mesh_cd = domain.mesh.cell_data
                    base_prob = None
                    if 'probability' in mesh_cd:
                        base_prob = mesh_cd['probability']
                    elif 'Normalized_Volume' in mesh_cd:
                        base_prob = mesh_cd['Normalized_Volume']
                    elif 'Normalized_Area' in mesh_cd:
                        base_prob = mesh_cd['Normalized_Area']
                    if base_prob is not None:
                        tree.probability = numpy.array(base_prob)
                        tree.domain.cumulative_probability = numpy.cumsum(tree.probability)
        return None

    def set_roots(self, *args, **kwargs):
        """
        Set the root point of the forest.

        Parameters
        ----------
        start_points : list of numpy.ndarray
            A list of starting points for the trees.
        directions : list of numpy.ndarray
            A list of directions for the trees.
        kwargs : dict
        """
        if len(args) == 0:
            start_points = kwargs.pop('start_points', self.start_points)
            directions = kwargs.pop('directions', self.directions)
        elif len(args) == 1:
            start_points = args[0]
            directions = kwargs.pop('directions', self.directions)
        elif len(args) == 2:
            start_points = args[0]
            directions = args[1]
        else:
            raise ValueError("Too many arguments.")
        root_kwargs = kwargs.copy()
        tmp_roots = []
        tmp_root_maps = []
        tmp_connectivities = []
        tmp_probabilities = []
        tmp_tree_scales = []
        # Lightweight spatial index for accepted root segments. Candidate roots
        # are first screened with midpoint sphere overlap before exact OBB checks.
        accepted_midpoints = []
        accepted_prune_radii = []
        accepted_midpoint_tree = None
        accepted_max_radius = 0.0
        for i in range(self.n_networks):
            for j in range(self.n_trees_per_network[i]):
                success = False
                while not success:
                    tmp_root, tmp_root_map, tmp_connecivity, _, _, _, tmp_probability, tmp_tree_scale = (
                        self.networks[i][j].set_root(
                            start=start_points[i][j],
                            direction=directions[i][j],
                            inplace=False,
                            build_indices=False,
                            **root_kwargs,
                        )
                    )
                    candidate_midpoint = ((tmp_root[:, 0:3] + tmp_root[:, 3:6]) / 2).reshape(-1)
                    candidate_radius = (
                        0.01 * float(tmp_root[0, 20]) +
                        float(tmp_root[0, 21]) +
                        float(self.physical_clearance)
                    )
                    if accepted_midpoint_tree is None:
                        nearby_ids = numpy.empty((0,), dtype=numpy.int64)
                    else:
                        search_radius = candidate_radius + accepted_max_radius
                        rough_ids = accepted_midpoint_tree.query_ball_point(candidate_midpoint, search_radius)
                        if len(rough_ids) == 0:
                            nearby_ids = numpy.empty((0,), dtype=numpy.int64)
                        else:
                            rough_ids = numpy.array(rough_ids, dtype=numpy.int64)
                            rough_midpoints = numpy.vstack([accepted_midpoints[idx] for idx in rough_ids])
                            rough_radii = numpy.array([accepted_prune_radii[idx] for idx in rough_ids],
                                                      dtype=numpy.float64)
                            rough_distances = numpy.linalg.norm(
                                rough_midpoints - candidate_midpoint.reshape(1, -1),
                                axis=1
                            )
                            nearby_ids = rough_ids[rough_distances <= (rough_radii + candidate_radius)]
                    has_collision = False
                    for k in nearby_ids:
                        root_collision = tree_collision(tmp_roots[int(k)], tmp_root,
                                                        clearance=self.physical_clearance)
                        if root_collision:
                            has_collision = True
                            break
                    if not has_collision:
                        success = True
                tmp_roots.append(tmp_root)
                tmp_root_maps.append(tmp_root_map)
                tmp_connectivities.append(tmp_connecivity)
                tmp_probabilities.append(tmp_probability)
                tmp_tree_scales.append(tmp_tree_scale)
                accepted_midpoints.append(candidate_midpoint)
                accepted_prune_radii.append(candidate_radius)
                accepted_max_radius = max(accepted_max_radius, candidate_radius)
                accepted_midpoint_tree = cKDTree(numpy.vstack(accepted_midpoints))
        root_idx = 0
        for i in range(self.n_networks):
            for j in range(self.n_trees_per_network[i]):
                tree = self.networks[i][j]
                tree.preallocate[0, :] = tmp_roots[root_idx]
                tree.data = tree.preallocate[:1, :]
                tree.vessel_map.update(tmp_root_maps[root_idx])
                tree.n_terminals = 1
                tree.max_distal_node = 1
                tree.connectivity = tmp_connectivities[root_idx]
                tree.probability = tmp_probabilities[root_idx]
                tree.tree_scale = tmp_tree_scales[root_idx]
                tree.midpoints = (tree.data[:, 0:3] + tree.data[:, 3:6]) / 2
                tree.kdtm = KDTreeManager(tree.midpoints.reshape(1, 3))
                tree.hnsw_tree = USearchTree(tree.midpoints.reshape(1, 3).astype(numpy.float32))
                tree.hnsw_tree_id = id(tree.hnsw_tree)
                tree.segment_count = 1
                if len(tree._idx_cache) == 0:
                    tree._idx_cache.append(0)
                if len(tree._col21_cache) == 0:
                    tree._col21_cache.append(21)
                root_idx += 1

    def add(self, *args, **kwargs):
        """
        Add a tree to the forest.
        """
        decay_probability = kwargs.pop('decay_probability', 0.9)
        self.connections = None
        if len(args) == 0:
            n_vessels = 1
        else:
            n_vessels = args[0]
        progress = kwargs.get('progress', False)
        network_id = kwargs.get('network_id', None)
        if isinstance(network_id, type(None)):
            network_id = list(range(self.n_networks))
        for k in range(n_vessels):
            new_vessels = [[None for j in range(self.n_trees_per_network[i])] for i in range(self.n_networks)]
            added_vessels = [[None for j in range(self.n_trees_per_network[i])] for i in range(self.n_networks)]
            new_vessel_maps = [[None for j in range(self.n_trees_per_network[i])] for i in range(self.n_networks)]
            new_connectivity = [[None for j in range(self.n_trees_per_network[i])] for i in range(self.n_networks)]
            new_inds = [[None for j in range(self.n_trees_per_network[i])] for i in range(self.n_networks)]
            new_mesh_cell = [[None for j in range(self.n_trees_per_network[i])] for i in range(self.n_networks)]
            for i in network_id:
                for j in range(self.n_trees_per_network[i]):
                    success = False
                    while not success:
                        #tmp_new_vessels, tmp_added_vessels, tmp_new_vessel_map, _, _, tmp_nonconvex_count = self.networks[i][j].add(inplace=False)
                        results = self.networks[i][j].add(inplace=False)
                        change_i, change_j, new_tmp_data, old_tmp_data, tmp_new_vessel_map, connectivity, inds, mesh_cell, tmp_added_vessels = results
                        #tmp_new_vessels = numpy.vstack([self.networks[i][j].data, tmp_added_vessels[0], tmp_added_vessels[1]])
                        change_i = numpy.array(change_i, dtype=int)
                        change_j = numpy.array(change_j, dtype=int)
                        required_rows = int(self.networks[i][j].segment_count) + 2
                        if change_i.size:
                            try:
                                required_rows = max(required_rows, int(change_i.max()) + 1)
                            except Exception:
                                pass
                        self.networks[i][j].ensure_preallocation(required_rows)
                        self.networks[i][j].preallocate[self.networks[i][j].segment_count, :] = tmp_added_vessels[0]
                        self.networks[i][j].preallocate[self.networks[i][j].segment_count+1, :] = tmp_added_vessels[1]
                        self.networks[i][j].preallocate[change_i, change_j] = numpy.array(new_tmp_data)
                        self.networks[i][j].data = self.networks[i][j].preallocate[:self.networks[i][j].segment_count+2, :]
                        self.networks[i][j].preallocate_midpoints[:self.networks[i][j].segment_count+2, :] = (self.networks[i][j].data[:, 0:3] +
                                                                                                              self.networks[i][j].data[:, 3:6])/2
                        self.networks[i][j].midpoints = self.networks[i][j].preallocate_midpoints[:self.networks[i][j].segment_count + 2, :]
                        #tmp_new_vessels[change_i, change_j] = numpy.array(new_tmp_data)
                        tmp_new_vessels = self.networks[i][j].data
                        collisions = []
                        for h in range(self.n_networks):
                            for w in range(self.n_trees_per_network[h]):
                                if h == i and w == j:
                                    continue
                                for q in range(len(tmp_added_vessels)):
                                    if isinstance(new_vessels[h][w], type(None)):
                                        collision = tree_collision(self.networks[h][w].data, tmp_added_vessels[q],
                                                                   clearance=self.physical_clearance)
                                    else:
                                        collision = tree_collision(new_vessels[h][w], tmp_added_vessels[q],
                                                                   clearance=self.physical_clearance)
                                    collisions.append(collision)
                                    if collision:
                                        print("Collision detected {}-{}.".format(h, q))
                                        break
                                if any(collisions):
                                    break
                            if any(collisions):
                                break
                        if not any(collisions):
                            success = True
                            new_vessels[i][j] = tmp_new_vessels
                            added_vessels[i][j] = tmp_added_vessels
                            new_vessel_maps[i][j] = tmp_new_vessel_map
                            new_connectivity[i][j] = connectivity
                            new_inds[i][j] = inds
                            new_mesh_cell[i][j] = mesh_cell
                        else:
                            success = False
                            self.networks[i][j].preallocate[self.networks[i][j].segment_count, :] = numpy.nan
                            self.networks[i][j].preallocate[self.networks[i][j].segment_count + 1, :] = numpy.nan
                            self.networks[i][j].preallocate[change_i, change_j] = numpy.array(old_tmp_data)
                            self.networks[i][j].data = self.networks[i][j].preallocate[:self.networks[i][j].segment_count + 2,:]
                            self.networks[i][j].preallocate_midpoints[:self.networks[i][j].segment_count + 2, :] = (self.networks[i][j].data[:,0:3] +
                                                                                                                    self.networks[i][j].data[:,3:6]) / 2
                            self.networks[i][j].midpoints = self.networks[i][j].preallocate_midpoints[
                                                            :self.networks[i][j].segment_count, :]
            for i in network_id:
                for j in range(self.n_trees_per_network[i]):
                    #_data = TreeData(new_vessels[i][j].shape)
                    #_data[:, :] = new_vessels[i][j][:, :]
                    #self.networks[i][j].data = TreeData.from_array(new_vessels[i][j])
                    #self.networks[i][j].vessel_map.update(new_vessel_maps[i][j])
                    for key in new_vessel_maps[i][j].keys():
                        if key in self.networks[i][j].vessel_map.keys():
                            self.networks[i][j].vessel_map[key]['upstream'].extend(new_vessel_maps[i][j][key]['upstream'])
                            self.networks[i][j].vessel_map[key]['downstream'].extend(new_vessel_maps[i][j][key]['downstream'])
                        else:
                            self.networks[i][j].vessel_map[key] = deepcopy(new_vessel_maps[i][j][key])
                    self.networks[i][j].connectivity = new_connectivity[i][j]
                    self.networks[i][j].max_distal_node += 2
                    if new_mesh_cell[i][j] >= 0:
                        self.networks[i][j].probability[mesh_cell] *= decay_probability
                        self.networks[i][j].probability = self.networks[i][j].probability / self.networks[i][j].probability.sum()
                        self.networks[i][j].domain.cumulative_probability = numpy.cumsum(self.networks[i][j].probability)
                    self.networks[i][j].hnsw_tree.replace(
                        ((new_vessels[i][j][new_inds[i][j][0], 0:3] + new_vessels[i][j][new_inds[i][j][0], 3:6]) / 2).reshape(1, 3).astype(
                            numpy.float32), numpy.array([new_inds[i][j][0]]))
                    self.networks[i][j].hnsw_tree.add_items(
                        ((new_vessels[i][j][new_inds[i][j][1], 0:3] + new_vessels[i][j][new_inds[i][j][1], 3:6]) / 2).reshape(1, 3).astype(
                            numpy.float32), numpy.array([new_inds[i][j][1]]))
                    self.networks[i][j].hnsw_tree.add_items(
                        ((new_vessels[i][j][new_inds[i][j][2], 0:3] + new_vessels[i][j][new_inds[i][j][2], 3:6]) / 2).reshape(1, 3).astype(
                            numpy.float32), numpy.array([new_inds[i][j][2]]))
                    self.networks[i][j].tree_scale = self.networks[i][j].new_tree_scale
                    self.networks[i][j].n_terminals += 1
                    self.networks[i][j].segment_count += 2
                    next_idxs = [len(self.networks[i][j]._idx_cache), len(self.networks[i][j]._idx_cache) + 1]
                    self.networks[i][j]._idx_cache.extend(next_idxs)
                    self.networks[i][j]._col21_cache.extend([21, 21])

    def connect(self, *args, **kwargs):
        self.connections = ForestConnection(self)
        if len(args) == 0:
            self.connections.solve(1, **kwargs)
        elif len(args) == 1 and isinstance(args[0], int):
            self.connections.solve(*args, **kwargs)
        else:
            raise ValueError("Invalid arguments. Optional integer argument for default Bezier polynomial degree.")

    def show(self, **kwargs):
        return show(self, **kwargs)

    def export_solid(self, outdir=None, shell_thickness=0.0):
        """
        Export the domain as a solid object.
        """
        if isinstance(outdir, type(None)):
            outdir = '3d_tmp'
        os.makedirs(outdir, exist_ok=True)
        for i in range(self.n_networks):
            net_dir = os.path.join(outdir, f"network_{i}")
            os.makedirs(net_dir, exist_ok=True)
            for j in range(self.n_trees_per_network[i]):
                self.networks[i][j].export_solid(outdir=net_dir, shell_thickness=shell_thickness)
                tmp_path = os.path.join(net_dir, "tree_mesh.vtp")
                dst_path = os.path.join(net_dir, f"tree_{j}_mesh.vtp")
                if os.path.exists(tmp_path):
                    os.replace(tmp_path, dst_path)

    def export_splines(self, outdir=None):
        """
        Export networks splines
        """
        if isinstance(outdir, type(None)):
            outdir = 'splines_tmp'
            if not os.path.exists(outdir):
                os.makedirs(outdir)
        if not isinstance(self.connections, type(None)):
            for i in range(len(self.connections.tree_connections)):
                interp_xyz, interp_radii, interp_normals, all_points, all_radii, all_normals = export_spline(self.connections.tree_connections[i])
                _ = write_splines(all_points, all_radii, outdir=outdir, name_prefix="{}".format(i))

    def export_centerlines(self, points_per_unit_length: int = 100, **kwargs):
        """
        Export centerline geometry for every tree in the forest.

        Parameters
        ----------
        points_per_unit_length : int, optional
            Sampling density along each spline. Default is 100.

        Returns
        -------
        result : CenterlineResult
            A 2-tuple ``(centerlines, polys)`` for backward compatibility.
            Access ``result.boundary_points`` for inlet/outlet metadata
            (each dict also includes a ``tree_label`` key).
        """
        import pyvista

        all_centerlines = None
        all_polys = []
        all_boundary_points = []

        for net_idx in range(self.n_networks):
            for tree_idx in range(self.n_trees_per_network[net_idx]):
                tree = self.networks[net_idx][tree_idx]
                if getattr(tree, "data", None) is None:
                    continue
                data_arr = numpy.asarray(tree.data)
                if data_arr.ndim != 2 or data_arr.shape[0] == 0:
                    continue

                result = tree.export_centerlines(
                    points_per_unit_length=points_per_unit_length, **kwargs
                )
                cl, polys = result
                bp = getattr(result, 'boundary_points', [])
                all_polys.extend(polys)

                label = f"network{net_idx}_tree{tree_idx}"
                for pt in bp:
                    pt['tree_label'] = label
                all_boundary_points.extend(bp)

                if all_centerlines is None:
                    all_centerlines = cl
                else:
                    all_centerlines = all_centerlines.merge(cl, merge_points=False)

        if all_centerlines is None:
            raise ValueError("Forest contains no trees with data to export.")

        from svv.tree.tree import CenterlineResult
        return CenterlineResult(all_centerlines, all_polys, all_boundary_points)

    def save(self, path: str, include_timing: bool = False):
        """
        Save this Forest to a .forest file.

        The saved file contains all forest data, tree parameters, and connectivity
        information needed to reconstruct the forest. The domain is NOT saved
        and must be set separately after loading via :meth:`set_domain`.

        Notes
        -----
        ``.forest`` files are compressed NumPy ``.npz`` archives with three
        top-level entries:

        - ``metadata``: a single dict with forest-level settings and a format
          ``version``.
        - ``trees``: nested lists of per-tree dicts containing ``metadata``,
          ``parameters``, populated vessel ``data`` (``TreeData`` as ndarray),
          and the adjacency ``vessel_map``. Optional timing data may be stored
          as ``times``.
        - ``connections`` (optional): results from :meth:`connect`.

        Tree growth uses large preallocated NaN buffers; only the populated
        rows (up to ``segment_count``) are written to disk.

        Parameters
        ----------
        path : str
            Output filename. If no extension is provided, ".forest" is appended.
        include_timing : bool, optional
            Include generation timing data for each tree (useful for profiling).
            Default is False.

        Returns
        -------
        pathlib.Path
            Path to the saved file.

        Examples
        --------
        >>> forest.save("my_forest.forest")
        >>> # Later...
        >>> loaded_forest = Forest.load("my_forest.forest")
        >>> loaded_forest.set_domain(domain)
        """
        from pathlib import Path
        import numpy as np

        path = Path(path)
        if path.suffix.lower() != '.forest':
            path = path.with_suffix('.forest')

        # Forest-level metadata
        metadata = {
            'version': 1,
            'n_networks': int(self.n_networks),
            'n_trees_per_network': [int(n) for n in self.n_trees_per_network],
            'physical_clearance': float(self.physical_clearance),
            'compete': bool(self.compete),
            'convex': bool(self.convex) if self.convex is not None else None,
        }

        # Serialize start_points and directions (nested lists of arrays or None)
        start_points_serialized = []
        for network_pts in self.start_points:
            network_serialized = []
            for pt in network_pts:
                if pt is not None:
                    network_serialized.append(numpy.asarray(pt).tolist())
                else:
                    network_serialized.append(None)
            start_points_serialized.append(network_serialized)

        directions_serialized = []
        for network_dirs in self.directions:
            network_serialized = []
            for d in network_dirs:
                if d is not None:
                    network_serialized.append(numpy.asarray(d).tolist())
                else:
                    network_serialized.append(None)
            directions_serialized.append(network_serialized)

        metadata['start_points'] = start_points_serialized
        metadata['directions'] = directions_serialized

        def _trim_populated_rows(tree):
            """Return only populated vessel rows for serialization.

            Trees preallocate large NaN-filled buffers for growth.  We do not
            want to persist those NaN rows.  Prefer the tree's `segment_count`
            (authoritative during generation) but fall back to trimming trailing
            all-NaN rows when the data view is larger than expected.
            """
            data_arr = np.asarray(tree.data)
            if data_arr.ndim != 2:
                data_arr = np.atleast_2d(data_arr)
            if data_arr.size == 0:
                return data_arr
            seg_count = getattr(tree, "segment_count", None)
            try:
                seg_count = int(seg_count)
            except (TypeError, ValueError):
                seg_count = data_arr.shape[0]
            if seg_count <= 0 or seg_count > data_arr.shape[0]:
                valid = ~np.all(np.isnan(data_arr), axis=1)
                if valid.any():
                    seg_count = int(np.where(valid)[0][-1] + 1)
                else:
                    seg_count = 0
            return data_arr[:seg_count, :]

        # Save each tree's data
        trees_data = []
        for i in range(self.n_networks):
            network_trees = []
            for j in range(self.n_trees_per_network[i]):
                tree = self.networks[i][j]
                trimmed_data = _trim_populated_rows(tree)
                tree_dict = {
                    'metadata': {
                        'n_terminals': int(tree.n_terminals),
                        'physical_clearance': float(tree.physical_clearance),
                        'random_seed': tree.random_seed,
                        'characteristic_length': float(tree.characteristic_length) if tree.characteristic_length is not None else None,
                        'clamped_root': bool(tree.clamped_root),
                        'nonconvex_count': int(tree.nonconvex_count),
                        'convex': bool(tree.convex) if tree.convex is not None else None,
                        # Persist segment_count matching the serialized data so
                        # loaders do not rely on stale counters.
                        'segment_count': int(trimmed_data.shape[0]),
                        'max_distal_node': int(getattr(tree, "max_distal_node", 0)) if getattr(tree, "max_distal_node", None) is not None else None,
                        'tree_scale': float(getattr(tree, "tree_scale", 0.0)) if getattr(tree, "tree_scale", None) is not None else None,
                        'domain_clearance': float(tree.domain_clearance) if tree.domain_clearance is not None else None,
                    },
                    'parameters': {
                        'kinematic_viscosity': float(tree.parameters.kinematic_viscosity),
                        'fluid_density': float(tree.parameters.fluid_density),
                        'terminal_flow': float(tree.parameters.terminal_flow) if tree.parameters.terminal_flow is not None else None,
                        'root_flow': float(tree.parameters.root_flow) if tree.parameters.root_flow is not None else None,
                        'terminal_pressure': float(tree.parameters.terminal_pressure),
                        'root_pressure': float(tree.parameters.root_pressure),
                        'murray_exponent': float(tree.parameters.murray_exponent),
                        'radius_exponent': float(tree.parameters.radius_exponent),
                        'length_exponent': float(tree.parameters.length_exponent),
                        'max_nonconvex_count': int(tree.parameters.max_nonconvex_count),
                        # Persist base-unit symbols plus the current pressure
                        # symbol for introspection. On load, UnitSystem is
                        # always constructed from the base units so pressure
                        # remains a derived quantity.
                        'unit_system': {
                            'length': tree.parameters.unit_system.base.length.symbol,
                            'time': tree.parameters.unit_system.base.time.symbol,
                            'mass': tree.parameters.unit_system.base.mass.symbol,
                            'pressure': tree.parameters.unit_system.pressure.symbol,
                        }
                    },
                    # Store only the populated vessel table, not the full
                    # preallocated NaN buffer.
                    'data': trimmed_data,
                    'vessel_map': dict(tree.vessel_map),
                }
                if include_timing:
                    tree_dict['times'] = tree.times
                network_trees.append(tree_dict)
            trees_data.append(network_trees)

        # Save connections if present
        connections_data = None
        if self.connections is not None:
            connections_data = {
                'tree_connections': []
            }
            for tc in self.connections.tree_connections:
                # Serialize vessels (list of lists of numpy arrays)
                vessels_serialized = []
                for tree_vessels in tc.vessels:
                    tree_vessels_list = []
                    for v in tree_vessels:
                        tree_vessels_list.append(numpy.asarray(v))
                    vessels_serialized.append(tree_vessels_list)

                # Serialize connected_network data
                connected_network_data = []
                for net_tree in tc.connected_network:
                    # Connected_network trees follow the same preallocation
                    # scheme; trim to populated rows before serialization.
                    connected_network_data.append(_trim_populated_rows(net_tree))

                tc_data = {
                    'network_id': int(tc.network_id),
                    'assignments': tc.assignments,
                    'vessels': vessels_serialized,
                    'lengths': tc.lengths,
                    'curve_type': tc.curve_type,
                    'connected_network': connected_network_data,
                }
                connections_data['tree_connections'].append(tc_data)

        # Build save dict
        save_dict = {
            'metadata': numpy.array([metadata], dtype=object),
            'trees': numpy.array([trees_data], dtype=object),
        }
        if connections_data is not None:
            save_dict['connections'] = numpy.array([connections_data], dtype=object)

        # Use an explicit file handle so NumPy does not append its default
        # ".npz" extension; the on-disk filename remains ".forest" as documented.
        with path.open('wb') as f:
            numpy.savez_compressed(f, **save_dict)
        return path

    @classmethod
    def load(cls, path: str):
        """
        Load a Forest from a .forest file.

        The loaded forest will NOT have a domain set. You must call
        :meth:`set_domain` after loading to enable domain-dependent
        operations like collision detection and further vessel generation.

        Parameters
        ----------
        path : str
            Path to a .forest file.

        Returns
        -------
        Forest
            Loaded forest instance.

        Notes
        -----
        If the forest was saved with connections (after calling :meth:`connect`),
        the connection results (vessels, assignments, connected_network) will be
        restored. However, to re-solve connections or generate new vessels,
        you must first call :meth:`set_domain`.  During loading, per-tree
        preallocation buffers and spatial indices (KD-tree / USearch) are
        rebuilt from the stored vessel table so growth can continue.  The
        sampling probability distribution is domain-dependent and therefore
        re-seeded when :meth:`set_domain` is called.

        Examples
        --------
        >>> forest = Forest.load("my_forest.forest")
        >>> forest.set_domain(domain)  # Required for domain operations
        >>> forest.show()
        """
        import numpy as np
        from svv.tree.data.data import TreeData, TreeParameters, TreeMap
        from svv.tree.data.units import UnitSystem
        from svv.tree.utils.TreeManager import KDTreeManager, USearchTree

        with np.load(path, allow_pickle=True) as f:
            metadata = f['metadata'][0]
            trees_data = f['trees'][0]
            connections_data = f['connections'][0] if 'connections' in f else None

        # Check version
        version = metadata.get('version', 1)
        if version > 1:
            raise ValueError(f"Unsupported .forest file version: {version}")

        # Estimate a reasonable preallocation size from stored tree data to
        # avoid allocating extremely large default arrays when loading.  Older
        # .forest files may include full preallocation buffers; prefer the
        # recorded segment_count when present.
        max_rows = 1
        for i in range(metadata['n_networks']):
            for j in range(metadata['n_trees_per_network'][i]):
                td = trees_data[i][j]
                data_arr = td['data']
                seg_count = None
                try:
                    seg_count = int(td.get('metadata', {}).get('segment_count', 0))
                except Exception:
                    seg_count = None
                if seg_count is not None and seg_count > max_rows:
                    max_rows = seg_count
                elif hasattr(data_arr, 'shape') and data_arr.shape[0] > max_rows:
                    max_rows = int(data_arr.shape[0])
        preallocation_step = max(max_rows * 2, 1)

        # Reconstruct start_points and directions
        start_points = []
        for network_pts in metadata.get('start_points', []):
            network_list = []
            for pt in network_pts:
                if pt is not None:
                    network_list.append(numpy.array(pt))
                else:
                    network_list.append(None)
            start_points.append(network_list)

        directions = []
        for network_dirs in metadata.get('directions', []):
            network_list = []
            for d in network_dirs:
                if d is not None:
                    network_list.append(numpy.array(d))
                else:
                    network_list.append(None)
            directions.append(network_list)

        # Create forest with metadata
        forest = cls(
            n_networks=metadata['n_networks'],
            n_trees_per_network=metadata['n_trees_per_network'],
            start_points=start_points if start_points else None,
            directions=directions if directions else None,
            physical_clearance=metadata.get('physical_clearance', 0.0),
            compete=metadata.get('compete', False),
            preallocation_step=preallocation_step,
        )
        forest.convex = metadata.get('convex', None)

        # Restore each tree's data
        for i in range(forest.n_networks):
            for j in range(forest.n_trees_per_network[i]):
                tree_dict = trees_data[i][j]
                tree = forest.networks[i][j]

                # Restore unit system from base units only; pressure and other
                # derived quantities are always auto-derived from these.
                us_dict = tree_dict['parameters'].get('unit_system', {})
                length_unit = us_dict.get('length', 'cm')
                time_unit = us_dict.get('time', 's')
                mass_unit = us_dict.get('mass', 'g')
                unit_system = UnitSystem(
                    length=length_unit,
                    time=time_unit,
                    mass=mass_unit,
                )
                tree.parameters.set_unit_system(unit_system)

                # Restore parameters
                params = tree_dict['parameters']
                tree.parameters.kinematic_viscosity = params['kinematic_viscosity']
                tree.parameters.fluid_density = params['fluid_density']
                tree.parameters.terminal_flow = params['terminal_flow']
                tree.parameters.root_flow = params['root_flow']
                tree.parameters.terminal_pressure = params['terminal_pressure']
                tree.parameters.root_pressure = params['root_pressure']
                tree.parameters.murray_exponent = params['murray_exponent']
                tree.parameters.radius_exponent = params['radius_exponent']
                tree.parameters.length_exponent = params['length_exponent']
                tree.parameters.max_nonconvex_count = params['max_nonconvex_count']

                # Restore data
                data_array = np.asarray(tree_dict['data'])
                if data_array.ndim != 2:
                    data_array = np.atleast_2d(data_array)
                tree.data = TreeData.from_array(data_array)

                # Determine the number of populated vessel rows.  Prefer the
                # saved segment_count, but fall back to trimming trailing NaNs
                # for legacy files that stored preallocated buffers.
                n_rows = 0
                meta_seg = tree_dict.get('metadata', {}).get('segment_count', None)
                try:
                    meta_seg = int(meta_seg)
                except (TypeError, ValueError):
                    meta_seg = None
                if meta_seg is not None and 0 < meta_seg <= data_array.shape[0]:
                    n_rows = meta_seg
                else:
                    valid = ~np.all(np.isnan(data_array), axis=1)
                    if valid.any():
                        n_rows = int(np.where(valid)[0][-1] + 1)
                if n_rows <= 0:
                    n_rows = 1

                # Trim the in-memory vessel table to populated rows to avoid
                # keeping legacy preallocation buffers alive after load.
                if n_rows < data_array.shape[0]:
                    data_array = data_array[:n_rows, :]
                    tree.data = TreeData.from_array(data_array)

                # Ensure the preallocation buffers are large enough, then seed
                # them with the loaded vessel table so growth can continue.
                if n_rows > tree.preallocate.shape[0]:
                    new_size = max(n_rows * 2, tree.preallocate.shape[0])
                    tree.preallocation_step = int(new_size)
                    tree.preallocate = TreeData((new_size, tree.data.shape[1]))
                    tree.preallocate_midpoints = np.zeros((new_size, 3))

                tree.preallocate[:n_rows, :] = tree.data[:n_rows, :]
                midpoints = (tree.data[:n_rows, 0:3] + tree.data[:n_rows, 3:6]) / 2
                tree.preallocate_midpoints[:n_rows, :] = midpoints
                tree.midpoints = tree.preallocate_midpoints[:n_rows, :]

                # Rebuild connectivity/search structures derived from data.
                tree.connectivity = np.nan_to_num(tree.data[:n_rows, 15:18], nan=-1.0).astype(int)
                tree.kdtm = KDTreeManager(midpoints)
                tree.hnsw_tree = USearchTree(midpoints.astype(np.float32))
                tree.hnsw_tree_id = id(tree.hnsw_tree)
                distal_nodes = tree.data[:n_rows, 19]
                if n_rows and not np.all(np.isnan(distal_nodes)):
                    tree.max_distal_node = int(np.nanmax(distal_nodes))
                else:
                    tree.max_distal_node = n_rows
                tree.tree_scale = float(
                    np.pi * np.nansum(
                        (tree.data[:n_rows, 21] ** tree.parameters.radius_exponent)
                        * (tree.data[:n_rows, 20] ** tree.parameters.length_exponent)
                    )
                )

                # Restore vessel map
                tree.vessel_map = TreeMap(tree_dict['vessel_map'])

                # Restore metadata
                meta = tree_dict['metadata']
                tree.n_terminals = meta.get('n_terminals', 0)
                tree.physical_clearance = meta.get('physical_clearance', 0.0)
                tree.random_seed = meta.get('random_seed', None)
                tree.characteristic_length = meta.get('characteristic_length', None)
                tree.clamped_root = meta.get('clamped_root', False)
                tree.nonconvex_count = meta.get('nonconvex_count', 0)
                tree.convex = meta.get('convex', None)
                # Segment count must align with the loaded data length.
                tree.segment_count = n_rows
                tree.domain_clearance = meta.get('domain_clearance', 0.0)
                # Optional persisted values (older files may omit them).
                if meta.get('max_distal_node') is not None:
                    tree.max_distal_node = meta['max_distal_node']
                if meta.get('tree_scale') is not None:
                    tree.tree_scale = meta['tree_scale']

                # Restore timing if present
                if 'times' in tree_dict:
                    tree.times = tree_dict['times']

        # Restore connections if present
        if connections_data is not None:
            forest.connections = ForestConnection(forest)
            forest.connections.tree_connections = []

            for tc_data in connections_data['tree_connections']:
                # Create a minimal TreeConnection-like object to hold the data
                tc = _LoadedTreeConnection(
                    forest=forest,
                    network_id=tc_data['network_id'],
                    assignments=tc_data['assignments'],
                    vessels=tc_data['vessels'],
                    lengths=tc_data['lengths'],
                    curve_type=tc_data['curve_type'],
                    connected_network_data=tc_data['connected_network'],
                )
                forest.connections.tree_connections.append(tc)

        return forest


class _LoadedTreeConnection:
    """
    Minimal container for loaded TreeConnection data.

    This class holds the results of a previously solved tree connection
    without requiring access to the original domain or solving infrastructure.
    """

    def __init__(self, forest, network_id, assignments, vessels, lengths,
                 curve_type, connected_network_data):
        from svv.tree.data.data import TreeData

        self.forest = forest
        self.network_id = network_id
        self.assignments = assignments
        self.vessels = vessels
        self.lengths = lengths
        self.curve_type = curve_type
        self.other_vessels = []
        self.meshes = []
        self.plotting_vessels = None
        self.connections = []

        # Reconstruct connected_network as Tree objects with data
        self.connected_network = []
        for i, data_array in enumerate(connected_network_data):
            # Use a small preallocation sized to the stored vessel table.
            # The loaded connection trees are lightweight containers for
            # geometry export and should not allocate the multi‑million row
            # growth buffers that `Tree()` defaults to.
            n_rows = int(getattr(data_array, "shape", (1,))[0]) or 1
            tree = Tree(preallocation_step=max(n_rows * 2, 1))
            tree.parameters = forest.networks[network_id][i].parameters
            tree.data = TreeData.from_array(data_array)
            tree.domain = None  # Will be set when forest.set_domain() is called
            self.connected_network.append(tree)

        # Build network reference (matches TreeConnection structure)
        self.network = []
        for i in range(forest.n_trees_per_network[network_id]):
            n_rows = int(getattr(forest.networks[network_id][i].data, "shape", (1,))[0]) or 1
            tree = Tree(preallocation_step=max(n_rows * 2, 1))
            tree.parameters = forest.networks[network_id][i].parameters
            tree.data = forest.networks[network_id][i].data
            tree.domain = None
            self.network.append(tree)

    def export_solid(self, cap_resolution=40, extrude_roots=False):
        """Export connected network solids for this loaded connection.

        The original :class:`~svv.forest.connect.tree_connection.TreeConnection`
        implementation provides the meshing logic.  Loaded connections already
        contain the solved assignments, vessels, and connected_network, so we
        delegate to that method after populating a lightweight stand‑in
        instance.  This avoids duplicating a large amount of geometry code
        while keeping legacy `.forest` files functional.

        Parameters
        ----------
        cap_resolution : int, optional
            Resolution for end‑cap triangulation passed through to the
            underlying export routine.
        extrude_roots : bool, optional
            When True, extend root vessels slightly outside the domain before
            meshing (requires the Forest domain to be set).
        """
        from svv.forest.connect.tree_connection import TreeConnection

        # Create an uninitialized TreeConnection and graft the loaded state
        # onto it so we can reuse TreeConnection.export_solid.
        tmp = TreeConnection.__new__(TreeConnection)
        tmp.forest = self.forest
        tmp.network_id = self.network_id
        tmp.assignments = self.assignments
        tmp.ctrlpts_functions = None
        tmp.connections = self.connections
        tmp.vessels = self.vessels
        tmp.lengths = self.lengths
        tmp.other_vessels = self.other_vessels
        tmp.meshes = self.meshes
        tmp.plotting_vessels = self.plotting_vessels
        tmp.network = self.network
        tmp.curve_type = self.curve_type
        tmp.connected_network = self.connected_network

        return TreeConnection.export_solid(
            tmp,
            cap_resolution=cap_resolution,
            extrude_roots=extrude_roots,
        )
