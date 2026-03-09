import numpy as np
import os
import uuid
import pyvista
from time import perf_counter
from tqdm import trange
from copy import deepcopy
from types import MappingProxyType
import itertools
from typing import Optional
from svv.tree.data.data import TreeData, TreeParameters, TreeMap
from svv.tree.data.units import UnitSystem
from svv.tree.branch.root import set_root
from svv.visualize.tree.show import show
from svv.tree.branch.bifurcation import add_vessel, check_tree
from svv.tree.export.export_solid import build_watertight_solid, build_merged_solid
from svv.tree.export.export_centerlines import build_centerlines
from svv.tree.export.write_splines import get_interpolated_sv_data, write_splines
from svv.tree.utils.TreeManager import KDTreeManager, USearchTree
from svv.tree.utils.c_extend import build_c_vessel_map
from collections import ChainMap


class Tree(object):
    def __init__(
        self,
        *,
        parameters: Optional[TreeParameters] = None,
        unit_system: Optional[UnitSystem] = None,
        preallocation_step: int = int(1e5),
    ):
        """
        The Tree class defines a branching tree structure
        that is used to abstract the physical representation
        of the generated vascular network.
        """
        self.data = TreeData()
        if parameters is None:
            self.parameters = TreeParameters(unit_system=unit_system)
        else:
            self.parameters = parameters
            if unit_system is not None:
                self.parameters.set_unit_system(unit_system)
        self.vessel_map = TreeMap()
        #self.c_vessel_map = None
        self.physical_clearance = 0.0
        self.domain = None
        self.convex = None
        self.random_seed = None
        self.characteristic_length = None
        self.domain_clearance = None
        self.n_terminals = 0
        self.nonconvex_count = 0
        self.clamped_root = False
        self.rtree = None
        self.preallocation_step = int(preallocation_step)
        self.preallocate = TreeData((self.preallocation_step, 31))
        self.segment_count = 0
        self._idx_cache = []
        self._col21_cache = []
        #self.preallocate_connectivity = (np.ones((self.preallocation_step, 3)) * -1).astype(int)
        self.preallocate_midpoints = np.zeros((self.preallocation_step, 3))
        self.times = {'vessels':[],
                      'local_optimization':[],
                      'collision':[],
                      'collision_1':[],
                      'collision_2':[],
                      'get_points':[],
                      'get_points_0':[],
                      'get_points_1':[],
                      'get_points_2':[],
                      'get_points_3':[],
                      'chunk_1':[],
                      'chunk_2':[],
                      'chunk_3':[],
                      'chunk_3_0':[],
                      'chunk_3_1':[],
                      'chunk_3_2':[],
                      'chunk_3_3':[],
                      'chunk_3_4':[],
                      'chunk_3_4_alt':[],
                      'chunk_3_5':[],
                      'chunk_3_6':[],
                      'chunk_3_7':[],
                      'chunk_4':[],
                      'chunk_4_0':[],
                      'chunk_4_1':[],
                      'chunk_4_2':[],
                      'chunk_4_3':[],
                      'chunk_5':[],
                      'all':[]}

    def ensure_preallocation(self, required_rows: int) -> None:
        """
        Ensure that growth preallocation buffers can hold `required_rows` rows.

        Tree growth writes into `self.preallocate` and `self.preallocate_midpoints`
        using direct indexing. Historically the default preallocation was very
        large, which could lead to multi‑GB allocations for multi‑tree forests.
        This helper allows smaller initial buffers while still supporting
        arbitrarily large trees by resizing on demand.
        """
        try:
            required = int(required_rows)
        except Exception:
            return
        if required <= 0:
            return

        current = int(getattr(self.preallocate, "shape", (0,))[0]) or 0
        if required <= current:
            return

        new_size = max(required, max(1, current * 2))

        new_preallocate = TreeData((new_size, self.preallocate.shape[1]))
        if current:
            new_preallocate[:current, :] = self.preallocate[:current, :]
        self.preallocate = new_preallocate

        new_midpoints = np.zeros((new_size, 3))
        old_midpoints = int(getattr(self.preallocate_midpoints, "shape", (0,))[0]) or 0
        if old_midpoints:
            new_midpoints[:old_midpoints, :] = self.preallocate_midpoints[:old_midpoints, :]
        self.preallocate_midpoints = new_midpoints

        self.preallocation_step = int(new_size)

        # Refresh views that commonly point into the preallocated buffers.
        try:
            n_rows = int(getattr(self.data, "shape", (0,))[0]) or 0
        except Exception:
            n_rows = 0
        if n_rows:
            self.data = self.preallocate[:n_rows, :]

        midpoints = getattr(self, "midpoints", None)
        if isinstance(midpoints, np.ndarray):
            if midpoints.ndim == 2 and midpoints.shape[1] == 3:
                self.midpoints = self.preallocate_midpoints[:midpoints.shape[0], :]
            elif midpoints.ndim == 1 and midpoints.shape[0] == 3:
                self.midpoints = self.preallocate_midpoints[0, :]

    def set_domain(self, domain, convexity_tolerance=1e-2):
        """
        Set the domain for the tree. The domain should be an implicit
        function of the form f(X) = scalar, where X is a cartesian
        coordinate in space.
        """
        self.domain = domain
        self.characteristic_length = domain.characteristic_length
        self.convex = np.isclose(domain.convexity, 1.0, atol=convexity_tolerance)
        if self.physical_clearance > 0.0:
            # Calculate domain_clearance by evaluating a point offset from the surface
            # Try to use patches first, fall back to boundary if patches not available
            if len(getattr(domain, 'patches', [])) > 0:
                patch = domain.patches[0]
                pt = (patch.points[0, :] + patch.normals[0, :] * self.physical_clearance).reshape(1, patch.points.shape[1])
            elif getattr(domain, 'boundary', None) is not None and domain.boundary.n_points > 0:
                # Use boundary point and compute normal from the boundary mesh
                boundary_pt = domain.boundary.points[0, :domain.points.shape[1]]
                # Compute normal at this point using the boundary mesh
                if hasattr(domain.boundary, 'point_normals') and domain.boundary.point_normals is not None:
                    normal = domain.boundary.point_normals[0, :domain.points.shape[1]]
                else:
                    # Compute normals if not available
                    domain.boundary.compute_normals(point_normals=True, inplace=True)
                    normal = domain.boundary.point_normals[0, :domain.points.shape[1]]
                # Offset point inward (negative normal direction for inward)
                pt = (boundary_pt - normal * self.physical_clearance).reshape(1, domain.points.shape[1])
            else:
                # Fall back to using domain points if nothing else available
                pt = domain.points[0, :].reshape(1, domain.points.shape[1])
            value = domain(pt)
            self.domain_clearance = abs(value).flatten()[0]
        else:
            self.domain_clearance = 0.0
        if self.domain.mesh is not None:
            self.domain.mesh.cell_data['probability'] = self.domain.mesh.cell_data['Normalized_Volume'].copy()
        return None

    def set_root(self, *args, **kwargs):
        """
        Set the root point of the tree.
        """
        if len(args) == 0:
            pass
        elif len(args) == 1:
            kwargs['start'] = args[0]
        elif len(args) == 2:
            kwargs['start'] = args[0]
            kwargs['direction'] = args[1]
            self.clamped_root = True
        else:
            raise ValueError("Too many arguments.")
        inplace = kwargs.pop('inplace', True)
        build_indices = kwargs.pop('build_indices', True)
        if not isinstance(kwargs.get('direction', None), type(None)):
            self.clamped_root = True
        if self.physical_clearance > 0.0:
            kwargs['interior_range'] = [-1.0, 0.0-self.domain_clearance]
        root, root_map = set_root(self, **kwargs)
        midpoint = ((root[:, 0:3] + root[:, 3:6]) / 2).reshape(1, 3)
        if inplace:
            self.data = root
            self.preallocate[0, :] = root
            if len(self._idx_cache) == 0:
                self._idx_cache.append(0)
            if len(self._col21_cache) == 0:
                self._col21_cache.append(21)
            #self.data_copy = self.preallocate[:3, :]
            self.connectivity = np.nan_to_num(root[:, 15:18], nan=-1.0).astype(int).reshape(1, 3)
            #self.preallocate_connectivity[0, :] = self.connectivity
            #self.connectivity_copy = self.connectivity.copy()
            #self.connectivity_copy = self.preallocate_connectivity[:3, :]
            self.preallocate_midpoints[0, :] = (root[:, 0:3] + root[:, 3:6]) / 2
            self.midpoints = self.preallocate_midpoints[0, :]
            #self.midpoints_copy = self.preallocate_midpoints[:1, :]
            self.vessel_map.update(root_map)
            #self.c_vessel_map = build_c_vessel_map(root_map)
            self.vessel_map_copy = deepcopy(self.vessel_map)
            self.n_terminals = 1
            if build_indices:
                self.kdtm = KDTreeManager(midpoint)
                self.hnsw_tree = USearchTree(midpoint.astype(np.float32))
                self.hnsw_tree_id = id(self.hnsw_tree)
            else:
                self.kdtm = None
                self.hnsw_tree = None
                self.hnsw_tree_id = None
            self.probability = np.array(self.domain.mesh.cell_data['probability'])
            self.max_distal_node = 1
            self.tree_scale = np.pi * root[0, 21]**self.parameters.radius_exponent*root[0, 20]**self.parameters.length_exponent
            self.segment_count = 1
            #self.rtree = RTree()
            #self.rtree.insert(((root[:, 0:3] + root[:, 3:6]) / 2).reshape(1, 3))
        else:
            connectivity = np.nan_to_num(root[:, 15:18], nan=-1.0).astype(int).reshape(1, 3)
            if build_indices:
                kdtm = KDTreeManager(midpoint)
                hnsw_tree = USearchTree(midpoint.astype(np.float32))
                hnsw_tree_id = id(hnsw_tree)
            else:
                kdtm = None
                hnsw_tree = None
                hnsw_tree_id = None
            probability = np.array(self.domain.mesh.cell_data['probability'])
            tree_scale = np.pi * root[0, 21] ** self.parameters.radius_exponent * root[
                0, 20] ** self.parameters.length_exponent
            return root, root_map, connectivity, kdtm, hnsw_tree, hnsw_tree_id, probability, tree_scale

    def add(self, inplace=True, **kwargs):
        all_start = perf_counter()
        decay_probability = kwargs.pop('decay_probability', 0.9)
        new_data, added_vessels, new_vessel_map, history, lines, nonconvex_outside, new_inds, mesh_cell, connectivity, change_i, change_j, new_tmp_data, old_tmp_data = add_vessel(self, **kwargs)
        start = perf_counter()
        #native_volume = np.sum(np.pi*new_data[:, 21]**2.0*new_data[:, 20])
        #assert np.isclose(native_volume,self.new_tree_scale), "native {} cost {} volumes do not match ".format(native_volume, self.new_tree_scale)
        #results = tuple([new_data, added_vessels, new_vessel_map, history, lines])
        #check = check_tree(self, results)
        #if not check[0]:
        #    print("Tree is not valid.")
        #    return check[1], check[2]
        #start = perf_counter()
        if not self.convex:
            if nonconvex_outside:
                self.nonconvex_count = 0
            else:
                self.nonconvex_count += 1
            if self.nonconvex_count > self.parameters.max_nonconvex_count:
                self.convex = True
        if inplace:
            #start_5 = perf_counter()
            #_data = TreeData(new_data.shape)
            #_data[:, :] = new_data[:, :]
            #self.data = _data
            #end_5 = perf_counter()
            #self.times['chunk_5'].append(end_5-start_5)
            start_chunk_4_0 = perf_counter()
            required_rows = int(self.segment_count) + 2
            try:
                required_rows = max(required_rows, int(np.max(change_i)) + 1)
            except Exception:
                pass
            self.ensure_preallocation(required_rows)
            #new_data = np.vstack([self.data, added_vessels[0], added_vessels[1]])
            self.preallocate[self.segment_count,:] = added_vessels[0]
            self.preallocate[self.segment_count+1,:] = added_vessels[1]
            next_idxs = [len(self._idx_cache), len(self._idx_cache) + 1]
            self._idx_cache.extend(next_idxs)
            self._col21_cache.extend([21,21])
            end_chunk_4_0 = perf_counter()
            self.times['chunk_4_0'].append(end_chunk_4_0 - start_chunk_4_0)
            start_chunk_4_1 = perf_counter()
            #new_data[change_i, change_j] = np.array(new_tmp_data)
            change_i = np.array(change_i, dtype=int)
            change_j = np.array(change_j, dtype=int)
            #if change_i.shape[0] > 10000:
            #    #print('parallel')
            #    parallel_scatter_update(self.preallocate, change_i, change_j, np.array(new_tmp_data))
            #else:
            self.preallocate[change_i, change_j] = np.array(new_tmp_data)
            end_chunk_4_1 = perf_counter()
            self.times['chunk_4_1'].append(end_chunk_4_1 - start_chunk_4_1)
            start_chunk_4_2 = perf_counter()
            #self.data = TreeData.from_array(new_data)
            self.data = self.preallocate[:self.segment_count+2,:]
            new_data = self.data
            end_chunk_4_2 = perf_counter()
            self.times['chunk_4_2'].append(end_chunk_4_2 - start_chunk_4_2)
            #np.copyto(self.preallocate[:new_data.shape[0], :], new_data)
            #self.data_copy = self.preallocate[:(new_data.shape[0] + 2), :]
            #self.vessel_map_copy = ChainMap(new_vessel_map, self.vessel_map)
            start_chunk_4_3 = perf_counter()
            for key in new_vessel_map.keys():
                if key in self.vessel_map.keys():
                    #print("new_vessel_map: {}".format(new_vessel_map))
                    #print("vessel_map: {}".format(self.vessel_map))
                    self.vessel_map[key]['upstream'].extend(new_vessel_map[key]['upstream'])
                    #_, counts = np.unique(self.vessel_map[key]['upstream'], return_counts=True)
                    #assert np.all(counts == 1), "Fail in appending upstream idxs.\nkey: {}\nprevious: {}\nextension: {}".format(key, self.vessel_map[key]['upstream'],
                    #                                                                                                   new_vessel_map[key]['upstream'])
                    self.vessel_map[key]['downstream'].extend(new_vessel_map[key]['downstream'])
                    #_, counts = np.unique(self.vessel_map[key]['downstream'], return_counts=True)
                    #assert np.all(counts == 1), "Fail in appending downstream idxs"
                else:
                    self.vessel_map[key] = {'upstream': list(new_vessel_map[key]['upstream']),
                                             'downstream': list(new_vessel_map[key]['downstream'])}
            end_chunk_4_3 = perf_counter()
            self.times['chunk_4_3'].append(end_chunk_4_3 - start_chunk_4_3)
            #self.vessel_map = ChainMap(new_vessel_map, self.vessel_map)
            self.n_terminals += 1
            self.segment_count += 2
            self.connectivity = connectivity #self.connectivity_copy.view()
            #self.connectivity_copy = self.preallocate_connectivity[:(self.connectivity_copy.shape[0] + 2), :]
            self.max_distal_node += 2
            if mesh_cell >= 0:
                self.probability[mesh_cell] *= decay_probability
                self.probability = self.probability / self.probability.sum()
                self.domain.cumulative_probability = np.cumsum(self.probability)
                #self.domain.mesh.cell_data['probability'][mesh_cell] *= decay_probability
                #self.domain.mesh.cell_data['probability'] = self.domain.mesh.cell_data['probability'] / \
                #                                            self.domain.mesh.cell_data['probability'].sum()
                #self.domain.cumulative_probability = np.cumsum(self.domain.mesh.cell_data['probability'])
            #self.kdtm.start_update(((new_data[:, 0:3] + new_data[:, 3:6]) / 2))
            #self.rtree.replace(new_inds[0], ((new_data[new_inds[0], 0:3] + new_data[new_inds[0], 3:6]) / 2).flatten().tolist())
            #self.rtree.rtree_index.insert(new_inds[1], ((new_data[new_inds[1], 0:3] + new_data[new_inds[1], 3:6]) / 2).flatten().tolist())
            #self.rtree.points.append(((new_data[new_inds[1], 0:3] + new_data[new_inds[1], 3:6]) / 2).flatten().tolist())
            #self.rtree.rtree_index.insert(new_inds[2], ((new_data[new_inds[2], 0:3] + new_data[new_inds[2], 3:6]) / 2).flatten().tolist())
            #self.rtree.points.append(((new_data[new_inds[2], 0:3] + new_data[new_inds[2], 3:6]) / 2).flatten().tolist())
            #self.kdtm.wait_for_update()
            #self.kdtm.swap_trees()
            #if self.n_terminals % 10000 == 0:
            #    self.hnsw_tree = USearchTree(((new_data[:, 0:3] + new_data[:, 3:6])/2).astype(np.float32))
            #else:
            self.hnsw_tree.replace(((new_data[new_inds[0], 0:3] + new_data[new_inds[0], 3:6]) / 2).reshape(1, 3).astype(np.float32), np.array([new_inds[0]]))
            self.hnsw_tree.add_items(((new_data[new_inds[1], 0:3] + new_data[new_inds[1], 3:6]) / 2).reshape(1, 3).astype(np.float32), np.array([new_inds[1]]))
            self.hnsw_tree.add_items(((new_data[new_inds[2], 0:3] + new_data[new_inds[2], 3:6]) / 2).reshape(1, 3).astype(np.float32), np.array([new_inds[2]]))
            #self.preallocate_midpoints[new_inds, :] = (new_data[new_inds, 0:3] + new_data[new_inds, 3:6])/2
            #self.midpoints_copy = self.preallocate_midpoints[:new_data.shape[0], :]
            #self.preallocate_midpoints[:self.segment_count, :] = (new_data[:, 0:3] + new_data[:, 3:6])/2
            self.preallocate_midpoints[new_inds, :] = (new_data[new_inds, 0:3] + new_data[new_inds, 3:6]) / 2
            self.tree_scale = self.new_tree_scale
            end = perf_counter()
            self.times['chunk_4'].append(end - start)
            self.times['all'].append(end - all_start)
            return None
        else:
            end = perf_counter()
            self.times['chunk_4'].append(end - start)
            self.times['all'].append(end - all_start)
            #return new_data, added_vessels, new_vessel_map, history, lines, nonconvex_outside, new_inds, mesh_cell
            return change_i, change_j, new_tmp_data, old_tmp_data, new_vessel_map, connectivity, new_inds, mesh_cell, added_vessels

    def n_add(self, n, **kwargs):
        for i in trange(n, desc='Adding vessels', unit='vessel', leave=False):
            self.add(**kwargs)
        return None
    def prune(self):
        pass

    def show(self, **kwargs):
        return show(self, **kwargs)

    def save(self, path: str, include_timing: bool = False):
        """
        Save this Tree to a .tree file.

        The saved file contains all tree data, parameters, and connectivity
        information needed to reconstruct the tree. The domain is NOT saved
        and must be set separately after loading via :meth:`set_domain`.

        Parameters
        ----------
        path : str
            Output filename. If no extension is provided, ".tree" is appended.
        include_timing : bool, optional
            Include generation timing data (useful for profiling/debugging).
            Default is False.

        Returns
        -------
        pathlib.Path
            Path to the saved file.

        Examples
        --------
        >>> tree.save("my_tree.tree")
        >>> # Later...
        >>> loaded_tree = Tree.load("my_tree.tree")
        >>> loaded_tree.set_domain(domain)
        """
        from pathlib import Path

        path = Path(path)
        if path.suffix.lower() != '.tree':
            path = path.with_suffix('.tree')

        # Metadata dict (JSON-serializable scalars)
        metadata = {
            'version': 1,
            'n_terminals': int(self.n_terminals),
            'physical_clearance': float(self.physical_clearance),
            'random_seed': self.random_seed,
            'characteristic_length': float(self.characteristic_length) if self.characteristic_length is not None else None,
            'clamped_root': bool(self.clamped_root),
            'nonconvex_count': int(self.nonconvex_count),
            'convex': bool(self.convex) if self.convex is not None else None,
            'segment_count': int(self.segment_count),
            'domain_clearance': float(self.domain_clearance) if self.domain_clearance is not None else None,
        }

        # Parameters (serialize to dict)
        params_dict = {
            'kinematic_viscosity': float(self.parameters.kinematic_viscosity),
            'fluid_density': float(self.parameters.fluid_density),
            'terminal_flow': float(self.parameters.terminal_flow) if self.parameters.terminal_flow is not None else None,
            'root_flow': float(self.parameters.root_flow) if self.parameters.root_flow is not None else None,
            'terminal_pressure': float(self.parameters.terminal_pressure),
            'root_pressure': float(self.parameters.root_pressure),
            'murray_exponent': float(self.parameters.murray_exponent),
            'radius_exponent': float(self.parameters.radius_exponent),
            'length_exponent': float(self.parameters.length_exponent),
            'max_nonconvex_count': int(self.parameters.max_nonconvex_count),
            # Persist base units plus the current pressure symbol for
            # introspection. On load, UnitSystem is always constructed from
            # the base units so pressure remains a derived quantity.
            'unit_system': {
                'length': self.parameters.unit_system.base.length.symbol,
                'time': self.parameters.unit_system.base.time.symbol,
                'mass': self.parameters.unit_system.base.mass.symbol,
                'pressure': self.parameters.unit_system.pressure.symbol,
            }
        }

        # Build save dict
        save_dict = {
            'metadata': np.array([metadata], dtype=object),
            'parameters': np.array([params_dict], dtype=object),
            'data': np.asarray(self.data),  # TreeData as plain ndarray
            'vessel_map': np.array([dict(self.vessel_map)], dtype=object),
        }

        if include_timing:
            save_dict['times'] = np.array([self.times], dtype=object)

        np.savez_compressed(path, **save_dict)
        return path

    @classmethod
    def load(cls, path: str):
        """
        Load a Tree from a .tree file.

        The loaded tree will NOT have a domain set. You must call
        :meth:`set_domain` after loading to enable domain-dependent
        operations like collision detection.

        Parameters
        ----------
        path : str
            Path to a .tree file.

        Returns
        -------
        Tree
            Loaded tree instance.

        Examples
        --------
        >>> tree = Tree.load("my_tree.tree")
        >>> tree.set_domain(domain)  # Required for domain operations
        """
        with np.load(path, allow_pickle=True) as f:
            metadata = f['metadata'][0]
            params_dict = f['parameters'][0]
            data_array = f['data']
            vessel_map_dict = f['vessel_map'][0]
            times = f['times'][0] if 'times' in f else None

        # Check version
        version = metadata.get('version', 1)
        if version > 1:
            raise ValueError(f"Unsupported .tree file version: {version}")

        # Reconstruct unit system from base units only; pressure and other
        # derived quantities are always auto-derived from these.
        us_dict = params_dict.get('unit_system', {})
        unit_system = UnitSystem(
            length=us_dict.get('length', 'cm'),
            time=us_dict.get('time', 's'),
            mass=us_dict.get('mass', 'g'),
        )

        # Size preallocation to the stored data rather than using the very
        # large default growth preallocation, to avoid excessive memory use
        # when loading existing trees.
        n_rows = int(getattr(data_array, 'shape', (0,))[0]) or 1
        preallocation_step = max(n_rows * 2, 1)

        # Create tree with parameters and tailored preallocation
        tree = cls(unit_system=unit_system, preallocation_step=preallocation_step)

        # Restore parameters
        tree.parameters.kinematic_viscosity = params_dict['kinematic_viscosity']
        tree.parameters.fluid_density = params_dict['fluid_density']
        tree.parameters.terminal_flow = params_dict['terminal_flow']
        tree.parameters.root_flow = params_dict['root_flow']
        tree.parameters.terminal_pressure = params_dict['terminal_pressure']
        tree.parameters.root_pressure = params_dict['root_pressure']
        tree.parameters.murray_exponent = params_dict['murray_exponent']
        tree.parameters.radius_exponent = params_dict['radius_exponent']
        tree.parameters.length_exponent = params_dict['length_exponent']
        tree.parameters.max_nonconvex_count = params_dict['max_nonconvex_count']

        # Restore data
        tree.data = TreeData.from_array(data_array)

        # Restore vessel map
        tree.vessel_map = TreeMap(vessel_map_dict)

        # Restore metadata
        tree.n_terminals = metadata.get('n_terminals', 0)
        tree.physical_clearance = metadata.get('physical_clearance', 0.0)
        tree.random_seed = metadata.get('random_seed', None)
        tree.characteristic_length = metadata.get('characteristic_length', None)
        tree.clamped_root = metadata.get('clamped_root', False)
        tree.nonconvex_count = metadata.get('nonconvex_count', 0)
        tree.convex = metadata.get('convex', None)
        tree.segment_count = metadata.get('segment_count', 0)
        tree.domain_clearance = metadata.get('domain_clearance', 0.0)

        # Restore timing data if present
        if times is not None:
            tree.times = times

        return tree

    def export_solid(self, outdir=None, shell_thickness=0.0, watertight=False, **kwargs):
        if isinstance(outdir, type(None)):
            if not os.path.exists(os.getcwd() + os.sep + '3d_tmp'):
                outdir = os.getcwd() + os.sep + '3d_tmp'
            else:
                outdir = os.getcwd() + os.sep + '3d_tmp_' + str(uuid.uuid4())
        else:
            if os.path.exists(outdir) and not os.path.isdir(outdir):
                raise ValueError("Output path exists and is not a directory.")
        if not watertight:
            model = build_merged_solid(self)
        else:
            model = build_watertight_solid(self)
        model.save(outdir + os.sep + 'tree_mesh.vtp')
        return model


    def export_centerlines(self, points_per_unit_length: int = 100, **kwargs):
        """
        Export centerline geometry for this tree.

        Parameters
        ----------
        points_per_unit_length : int, optional
            Sampling density along each spline, in points per unit length of
            centerline. Higher values yield smoother centerlines at increased
            file size. Default is 100.

        Returns
        -------
        centerlines : pyvista.PolyData
            Centerline polydata with radius, section-area, and
            ``BoundaryType`` arrays (0 = interior, 1 = inlet, 2 = outlet).
        polys : list[pyvista.PolyData]
            Per-branch polylines used to construct the merged centerline set.
        boundary_points : list[dict]
            Each dict has keys ``type`` ('inlet'/'outlet'), ``point``
            (3-element array) and ``radius`` (float).
        """
        centerlines, polys, boundary_points = build_centerlines(
            self, points_per_unit_length=points_per_unit_length,
        )
        return centerlines, polys, boundary_points


    def export_gcode(self):
        pass

    def export_splines(self, spline_sample_points=100):
        interp_xyz, interp_r, interp_n, path_frames, branches, interp_xyzr = get_interpolated_sv_data(self.data)
        splines = write_splines(interp_xyzr, spline_sample_points=spline_sample_points)
        return splines
