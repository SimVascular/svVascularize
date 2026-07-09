import numpy
import numpy as np
import os
from scipy.spatial import cKDTree
import pyvista as pv
from scipy.optimize import minimize, brute
from copy import deepcopy
import pyvista as pv
from time import perf_counter
import itertools
from svv.tree.data.data import TreeData, TreeMap
try:
    from svv.tree.utils.c_local_optimize import tree_cost, create_new_vessels, update_vessels, tree_cost_2
    _LOCAL_OPT_AVAILABLE = True
except Exception:
    tree_cost = create_new_vessels = update_vessels = None  # type: ignore
    _LOCAL_OPT_AVAILABLE = False
from svv.tree.utils.c_close import close, close_exact_points, close_exact_point, sphere_proximity
from svv.tree.utils.c_basis import basis, basis_inplace
from svv.tree.utils.c_obb import obb_any
from svv.tree.utils.c_angle import get_angles
from svv.tree.utils.c_extend import update_alt
from scipy.sparse import coo_matrix, lil_matrix
import numexpr as ne
import matplotlib.pyplot as plt
ne.set_num_threads(min(16, max(1, os.cpu_count() or 1)))

#[TODO] angle constraint to make sure that daughters are wide enough apart
#[TODO] remove terminal and parent sister collision constraint
#[TODO] add angle constraint for terminal and new parent vessel?
#[TODO] might need a point to segment distance calculation to ensure that new vessels don't collapse
#[TODO] radius of rejection for bifurcation of parent and new terminal orrrr nominal mimumum length for new parent
#[TODO] adding a new vessel should track all data changes as lists that are returned as the solution
#[TODO] check why it is difficult to obtain points for adding vessels

#@profile
def add_vessel(tree, **kwargs):
    if not _LOCAL_OPT_AVAILABLE:
        raise ImportError(
            "Local optimization requires accelerators. Install with 'pip install svv[accel]' "
            "or build with SVV_BUILD_EXTENSIONS=1."
        )
    """
    Create a potential vessel for the current tree configuration.

    Parameters
    ----------
    tree : TreeData
        The current tree configuration.
    kwargs : dict
        A dictionary of keyword arguments to be passed.
        Keyword arguments:
            interior_range : list
                The range of interior values for the domain. Default is [-1.0, 0.0 - tree.domain_clearance].
            exterior_range : list
                The range of exterior values for the domain. Default is [0.0, 1.0].
            flow_ratio : float
                The ratio of the terminal flow to the flow of the closest vessel.
                Default is 10 (i.e. terminal vessels are appended to tree vessels
                that have a flow 10 times less than the new terminal flow).
            callback : bool
                A flag to enable the callback function. Default is True.
            x0 : np.ndarray
                The initial guess for the optimizer. Default is [0.5, 0.5].
            threshold_exponent : float
                The exponent for the threshold value. Default is 1.0
            threshold_adjuster : float
                The factor to adjust the threshold value. Default is 0.9.
            n_points : int
                The number of points to draw for the terminal point. Default is 100.
            n_closest_vessels : int
                The number of closest vessels to consider before trying
                a terminal point. Default is 10.
            nonconvex_sampling : int
                The number of points to sample for nonconvexity. Default is 10.
            homogeneous : bool
                A flag to enable homogeneous vessel distribution within the
                tissue domain. If not homogeneous, new vessels will be added
                in a directed manner from the root isosurface location
                inward. Default is True.
    """
    interior_range = kwargs.get('interior_range', [-1.0, 0.0 - tree.domain_clearance])
    exterior_range = kwargs.get('exterior_range', [0.0, 1.0])
    flow_ratio = kwargs.get('flow_ratio', 20)
    max_depth = kwargs.get('max_depth', 20)
    callback = kwargs.get('callback', True)
    x0 = kwargs.get('x0', numpy.array([0.5, 0.5]))
    threshold_exponent = kwargs.pop('threshold_exponent', 1.5)
    threshold_adjuster = kwargs.pop('threshold_adjuster', 0.9)
    n_points = kwargs.pop('n_points', 50)
    n_closest_vessels = kwargs.pop('n_closest_vessels', 2)
    nonconvex_sampling = kwargs.pop('nonconvex_sampling', 10)
    homogeneous = kwargs.pop('homogeneous', True)
    use_brute = kwargs.pop('use_brute', False)
    max_iter = kwargs.pop('max_iter', 20)
    return_cost = kwargs.pop('return_cost', False)
    #defualt_threshold = ((tree.domain.mesh.volume ** (1/3)) /
    #                     (tree.n_terminals ** threshold_exponent)) + tree.data[0, 21]*2.0
    defualt_threshold = ((tree.domain.volume ** (1/3)) /
                         (tree.n_terminals ** threshold_exponent)) #+ tree.data[0, 21]*2.0
    #tree_scale = numpy.pi * numpy.sum(numpy.power(tree.data[:, 21], tree.parameters.radius_exponent) *
    #                                  numpy.power(tree.data[:, 20], tree.parameters.length_exponent))
    #tree_scale = ne_scale(tree.data[:, 21], tree.data[:, 20],
    #                      tree.parameters.radius_exponent, tree.parameters.length_exponent)
    tree_scale = tree.tree_scale
    tree.volume_scale = tree_scale
    threshold = kwargs.pop('threshold', defualt_threshold)
    nonconvex_outside = False
    #search_tree = cKDTree((tree.data[:, 0:3] + tree.data[:, 3:6]) / 2)
    data = tree.data[:tree.segment_count, :]
    tree.times['vessels'].append(data.shape[0])
    tree.times['local_optimization'].append(0)
    tree.times['collision'].append(0)
    tree.times['chunk_1'].append(0)
    tree.times['chunk_2'].append(0)
    tree.times['chunk_3'].append(0)
    tree.times['get_points'].append(0)
    tree.times['chunk_3_0'].append(0)
    tree.times['chunk_3_1'].append(0)
    tree.times['chunk_3_2'].append(0)
    tree.times['chunk_3_3'].append(0)
    tree.times['chunk_3_4'].append(0)
    tree.times['chunk_3_4_alt'].append(0)
    tree.times['chunk_3_5'].append(0)
    tree.times['chunk_3_6'].append(0)
    tree.times['chunk_3_7'].append(0)
    tree.times['collision_1'].append(0)
    tree.times['collision_2'].append(0)
    data = tree.data[:tree.segment_count, :]
    if not homogeneous:
        raise NotImplementedError("Non-homogeneous trees are not supported.")
    else:
        #tree.midpoints = (data[:, 0:3] + data[:, 3:6]) / 2
        if tree.convex:
            success = False
            #tree.midpoints = (tree.data_copy[:-2, 0:3] + tree.data_copy[:-2, 3:6]) / 2
            #midpoints = np.empty(midpoints_base.shape, dtype=midpoints_base.dtype)
            #np.copyto(midpoints, midpoints_base)
            max_distal_node = tree.max_distal_node #tree.data[:, 19].max()
            proximity_check = numpy.full((data.shape[0],), False, dtype=bool)
            while not success:
                get_points_start = perf_counter()
                terminal_points, terminal_point_distances, closest_vessels, mesh_cells = get_points(tree, n_points, threshold=threshold,
                                                                                        interior_range=interior_range,
                                                                                        n_vessels=n_closest_vessels)
                if numpy.all(numpy.isnan(terminal_points)) or len(terminal_points) == 0:
                    threshold *= threshold_adjuster
                    #print('Error: all nan points')
                    continue
                elif numpy.any(numpy.isnan(terminal_points)):
                    terminal_point_distances = terminal_point_distances[:, ~numpy.isnan(terminal_points).any(axis=1)]
                    closest_vessels = closest_vessels[:, ~numpy.isnan(terminal_points).any(axis=1)]
                    mesh_cells = mesh_cells[~numpy.isnan(terminal_points).any(axis=1)]
                    terminal_points = terminal_points[~numpy.isnan(terminal_points).any(axis=1)]
                #closest_vessels = numpy.argsort(terminal_point_distances, axis=0)
                n_closest_vessels = min(n_closest_vessels, data.shape[0])
                get_points_end = perf_counter()
                tree.times['get_points'][-1] += get_points_end - get_points_start
                for i in range(terminal_points.shape[0]):
                    for j in range(n_closest_vessels):
                        start_1 = perf_counter()
                        if flow_ratio is not None:
                            if (data[closest_vessels[j, i], 22] / tree.parameters.terminal_flow) > flow_ratio:
                                #print('flow_ratio')
                                continue
                        bifurcation_vessel = closest_vessels[j, i]
                        terminal_point = terminal_points[i, :]
                        dist = close_exact_point(data[bifurcation_vessel, :].reshape(1,data.shape[1]),
                                          terminal_point)
                        if dist < data[bifurcation_vessel, 21]*4:
                            #print('too close')
                            continue
                        cost, triad, vol = construct_optimizer(tree, terminal_points[i, :], closest_vessels[j, i])
                        bifurcation_cell = mesh_cells[i]
                        if callback:
                            history = []
                            lines = numpy.zeros((6, 3), dtype=numpy.float64)
                            lines[0, :] = data[closest_vessels[j, i], 0:3]
                            lines[1, :] = data[closest_vessels[j, i], 3:6]
                            lines[2, :] = data[closest_vessels[j, i], 0:3]
                            lines[3, :] = terminal_points[i, :]
                            lines[4, :] = data[closest_vessels[j, i], 3:6]
                            lines[5, :] = terminal_points[i, :]

                            def callback(xk, history=history):
                                history.append(triad(xk))

                        else:
                            lines = []

                            def callback(xk):
                                pass
                        end_1 = perf_counter()
                        tree.times['chunk_1'][-1] += end_1 - start_1
                        start = perf_counter()
                        if use_brute:
                            result = brute(cost, [(0.0, 1.0), (0.0, 1.0)], Ns=max_iter)
                            bifurcation_point = triad(result)
                            tree.new_tree_scale = vol(result)
                        else:
                            #if tree.data.get('depth', closest_vessels[j, i]) > max_depth:
                            #    bifurcation_point = (tree.data[closest_vessels[j, i], 0:3] +
                            #                         tree.data[closest_vessels[j, i], 0:3])/2
                            if True:
                                #vals = np.linspace(0.001,1-0.001,50)
                                #X,Y = np.meshgrid(vals,vals)
                                #XX = np.vstack((X.flatten(), Y.flatten())).T
                                #V = []
                                #for ii in range(XX.shape[0]):
                                #    V.append(cost(XX[ii]))
                                #V = np.array(V)
                                #min_idx = np.argmin(V)
                                #print('BRUTE: {}'.format(XX[min_idx]))
                                #print('BRUTE FUN: {}'.format(V[min_idx]))
                                #print('MAX: {}'.format(np.max(V)))
                                #V = V.reshape(len(vals), len(vals))
                                #plt.contourf(X,Y,V,cmap='viridis',levels=50)
                                #plt.scatter(XX[min_idx,0],XX[min_idx,1],marker='x',color='red')
                                #plt.colorbar(label='Function values')
                                #plt.show()
                                cons = [{"type": "ineq", "fun": lambda a: 1 - a[0] - a[1]}]
                                result = minimize(cost, x0, bounds=[(0.05, 0.95), (0.05, 0.95)], callback=callback,
                                                  options={'maxiter':max_iter},constraints=cons, method="L-BFGS-B")
                                #print('SOLUTION: {}'.format(result.x))
                                #print('SOLUTION FUN: {}'.format(result.fun))
                                bifurcation_point = triad(result.x)
                                tree.new_tree_scale = vol(result.x)
                                if not result.success:
                                    #print(result.message)
                                    continue
                        #result = minimize(cost, x0, bounds=[(0.0, 1.0), (0.0, 1.0)], callback=callback)
                        end = perf_counter()
                        tree.times['local_optimization'][-1] += end - start
                        start_2 = perf_counter()
                        #if not result.success:
                        #    #midpoints[closest_vessels[j, i], :] = midpoints_base[closest_vessels[j, i], :]
                        #    continue
                        #bifurcation_point = triad(result.x)
                        #midpoints = (tree.data_copy[:, 0:3] + tree.data_copy[:, 3:6])/2
                        #midpoints[closest_vessels[j, i], :] = (tree.data_copy[closest_vessels[j, i], 0:3] + bifurcation_point)/2
                        #midpoints = numpy.vstack((midpoints_base, ((terminal_points[i, :] + bifurcation_point)/2),
                        #                                     (tree.data_copy[closest_vessels[j, i], 3:6] + bifurcation_point)/2))
                        #tree.kdtm.start_update(midpoints)
                        #bifurcation_point_value = tree.domain(bifurcation_point.reshape(1, -1))
                        #if numpy.any(bifurcation_point_value > interior_range[1]):
                        #    continue
                        #if numpy.any(bifurcation_point_value < interior_range[0]):
                        #    continue
                        #bifurcation_vessel = closest_vessels[j, i]
                        #terminal_point = terminal_points[i, :]
                        #dist = close_exact_point(data[bifurcation_vessel, :].reshape(1,data.shape[1]),
                        #                  terminal_point)
                        #if dist < data[bifurcation_vessel, 21]*4:
                        #    print('too close')
                        #    continue
                        terminal_vessel = TreeData()
                        ### CHECK ANGLES ###
                        vec_parent = (data[bifurcation_vessel, 0:3] - bifurcation_point).reshape(1,3)
                        vec_term = (terminal_point - bifurcation_point).reshape(1,3)
                        vec_daughter = (data[bifurcation_vessel, 3:6] - bifurcation_point).reshape(1,3)
                        angle = get_angles(vec_parent, vec_term)
                        #plotter = pv.Plotter()
                        #lines = [pv.Line(tree.data[bifurcation_vessel, 0:3], bifurcation_point),
                        #         pv.Line(bifurcation_point, terminal_point),
                        #         pv.Line(tree.data[bifurcation_vessel, 3:6], bifurcation_point)]
                        #plotter.add_mesh(lines[0],color='green',line_width=3)
                        #plotter.add_mesh(lines[1],color='blue',line_width=3)
                        #plotter.add_mesh(lines[2],color='yellow',line_width=3)
                        #plotter.show()
                        #if angle < 90:
                        #    print('parent-terminal angle fail. degrees: {}'.format(angle))
                        #    continue
                        #angle = get_angles(vec_parent, vec_daughter)
                        #if angle < 90:
                        #    print('parent-daughter angle fail. degrees: {}'.format(angle))
                        #    continue
                        #terminal_daughter_vessel = TreeData()
                        #parent_vessel = TreeData()
                        #connectivity = numpy.nan_to_num(tree.data[:, 15:18], nan=-1.0).astype(int)
                        connectivity = tree.connectivity.copy()
                        #create_new_vessels(bifurcation_point, tree.data, terminal_point, terminal_vessel,
                        #                   terminal_daughter_vessel, parent_vessel, max_distal_node,
                        #                   numpy.float64(tree.data.shape[0]),
                        #                   connectivity[:-2, :], bifurcation_vessel, tree.parameters.murray_exponent,
                        #                   tree.parameters.kinematic_viscosity*tree.parameters.fluid_density, tree.parameters.terminal_flow,
                        #                   tree.parameters.terminal_pressure, tree.parameters.root_pressure,
                        #                   tree.parameters.radius_exponent, tree.parameters.length_exponent)
                        terminal_vessel[0, 0:3] = bifurcation_point
                        terminal_vessel[0, 3:6] = terminal_point
                        basis_inplace(terminal_vessel[:, 0:3], terminal_vessel[:, 3:6],
                                                terminal_vessel[:, 6:9], terminal_vessel[:, 9:12],
                                                terminal_vessel[:, 12:15])
                        terminal_vessel[0, 17] = bifurcation_vessel
                        terminal_vessel[0, 20] = np.linalg.norm(terminal_vessel[0, 3:6] - terminal_vessel[0, 0:3])
                        terminal_vessel[0, 21] = data[bifurcation_vessel, 21]
                        #terminal_daughter_vessel = TreeData()
                        #terminal_daughter_vessel[0, 0:3] = bifurcation_point
                        #terminal_daughter_vessel[0, 3:6] = tree.data[bifurcation_vessel, 3:6]
                        #basis_inplace(terminal_daughter_vessel[:, 0:3], terminal_daughter_vessel[:, 3:6],
                        #              terminal_daughter_vessel[:, 6:9], terminal_daughter_vessel[:, 9:12],
                        #              terminal_daughter_vessel[:, 12:15])
                        #terminal_daughter_vessel[0, 15] = tree.data[bifurcation_vessel, 15]
                        #terminal_daughter_vessel[0, 16] = tree.data[bifurcation_vessel, 16]
                        #terminal_daughter_vessel[0, 17] = bifurcation_vessel
                        #terminal_daughter_vessel[0, 20] = np.linalg.norm(terminal_daughter_vessel[0, 3:6] -
                        #                                                 terminal_daughter_vessel[0, 0:3])
                        #terminal_daughter_vessel[0, 21] = tree.data[bifurcation_vessel, 21]
                        #parent_vessel = TreeData()
                        #parent_vessel[0, 0:3] = tree.data[bifurcation_vessel, 0:3]
                        #parent_vessel[0, 3:6] = tree.data[bifurcation_vessel, 3:6]
                        #basis_inplace(parent_vessel[:, 0:3], parent_vessel[:, 3:6],
                        #              parent_vessel[:, 6:9], parent_vessel[:, 9:12],
                        #              parent_vessel[:, 12:15])
                        #parent_vessel[0, 15] = tree.data.shape[0]
                        #parent_vessel[0, 16] = tree.data.shape[0] + 1
                        #parent_vessel[0, 17] = tree.data[bifurcation_vessel, 17]
                        #parent_vessel[0, 20] = np.linalg.norm(parent_vessel[0, 3:6] -
                        #                                      parent_vessel[0, 0:3])
                        #parent_vessel[0, 21] = tree.data[bifurcation_vessel, 21]
                        terminal_vessel[0, 21] += tree.physical_clearance
                        end_2 = perf_counter()
                        tree.times['chunk_2'][-1] += end_2 - start_2
                        start = perf_counter()
                        start_c_1 = perf_counter()
                        #search_radius = numpy.max(tree.data[:, 20])/2 + numpy.max(tree.data[:, 21]) + terminal_vessel[0, 20]/2 + terminal_vessel[0, 21]
                        search_radius = data[bifurcation_vessel, 20] + 2.0 * data[bifurcation_vessel, 21] + terminal_vessel[0, 20]/2 + 2.0 * terminal_vessel[0, 21]
                        #terminal_vessel_proximity = search_tree.query_ball_point((terminal_vessel[0, 0:3] +
                        #                                                          terminal_vessel[0, 3:6])/2, search_radius)
                        #terminal_vessel_proximity = tree.kdtm.query_ball_point((terminal_vessel[0, 0:3] +
                        #                                                          terminal_vessel[0, 3:6])/2, search_radius.mean())
                        terminal_vessel_proximity = tree.hnsw_tree.query_ball_point(((terminal_vessel[0, 0:3] +
                                                                                     terminal_vessel[0, 3:6])/2).reshape(1,3), search_radius.mean())
                        #terminal_vessel_proximity_distances = terminal_vessel_proximity_distances.flatten()
                        #terminal_vessel_proximity_distances = terminal_vessel_proximity_distances - \
                        #                                      tree.data[terminal_vessel_proximity, 20]/2 - \
                        #                                      terminal_vessel[0, 20]/2 - terminal_vessel[0, 21] - \
                        #                                      tree.data[terminal_vessel_proximity, 21]
                        #terminal_vessel_proximity_check = numpy.full((tree.data.shape[0],), False, dtype=bool)
                        proximity_check.fill(False)
                        proximity_check[terminal_vessel_proximity] = True
                        #terminal_vessel_proximity = terminal_vessel_proximity_check
                        #terminal_vessel_proximity = sphere_proximity(tree.data, terminal_vessel[0, :])
                        #terminal_vessel_proximity = terminal_vessel_proximity_distances < 0
                        #if isinstance(terminal_vessel_proximity, numpy.ndarray):
                        #plotter = pv.Plotter()
                        #center = (terminal_vessel[0, 0:3] + terminal_vessel[0, 3:6])/2
                        #direction = (terminal_vessel[0, 3:6] - terminal_vessel[0, 0:3])
                        #length = np.linalg.norm(direction)
                        #direction = direction/length
                        #cyl = pv.Cylinder(radius=terminal_vessel[0,21],center=center,direction=direction,height=length,capping=True)
                        #plotter.add_mesh(cyl, color='green', label='new terminal')
                        proximity_check[bifurcation_vessel] = False
                        #center = (tree.data[bifurcation_vessel, 0:3] + tree.data[bifurcation_vessel, 3:6])/2
                        #direction = (tree.data[bifurcation_vessel, 3:6] - tree.data[bifurcation_vessel, 0:3])
                        #length = np.linalg.norm(direction)
                        #direction = direction/length
                        #cyl = pv.Cylinder(radius=tree.data[bifurcation_vessel, 21],center=center,direction=direction,height=length,capping=True)
                        #plotter.add_mesh(cyl, color='red', label='bifurcation vessel')
                        if not numpy.isnan(data[bifurcation_vessel, 15]):
                            #terminal_daughter_vessel_proximity[int(terminal_daughter_vessel[0, 15])] = False
                            #proximity_check[int(tree.data[bifurcation_vessel, 15])] = False
                            #center = (tree.data[int(tree.data[bifurcation_vessel, 15]), 0:3] + tree.data[int(tree.data[bifurcation_vessel, 15]), 3:6]) / 2
                            #direction = (tree.data[int(tree.data[bifurcation_vessel, 15]), 3:6] - tree.data[int(tree.data[bifurcation_vessel, 15]), 0:3])
                            #length = np.linalg.norm(direction)
                            #direction = direction / length
                            #cyl = pv.Cylinder(radius=tree.data[int(tree.data[bifurcation_vessel, 15]), 21], center=center, direction=direction,
                            #                  height=length, capping=True)
                            #plotter.add_mesh(cyl, color='yellow', label='left daughter')
                            pass
                        if not numpy.isnan(data[bifurcation_vessel, 16]):
                            #terminal_daughter_vessel_proximity[int(terminal_daughter_vessel[0, 16])] = False
                            #proximity_check[int(tree.data[bifurcation_vessel, 16])] = False
                            #center = (tree.data[int(tree.data[bifurcation_vessel, 16]), 0:3] + tree.data[int(tree.data[bifurcation_vessel, 16]), 3:6]) / 2
                            #direction = (tree.data[int(tree.data[bifurcation_vessel, 16]), 3:6] - tree.data[int(tree.data[bifurcation_vessel, 16]), 0:3])
                            #length = np.linalg.norm(direction)
                            #direction = direction / length
                            #cyl = pv.Cylinder(radius=tree.data[int(tree.data[bifurcation_vessel, 16]), 21], center=center, direction=direction,
                            #                  height=length, capping=True)
                            #plotter.add_mesh(cyl, color='yellow', label='right daughter')
                            pass
                        if not numpy.isnan(data[bifurcation_vessel, 17]):
                            super_parent = int(data[bifurcation_vessel, 17])
                            #parent_vessel_proximity[int(tree.data[bifurcation_vessel, 17])] = False
                            proximity_check[int(data[bifurcation_vessel, 17])] = False
                            #center = (tree.data[int(tree.data[bifurcation_vessel, 17]), 0:3] + tree.data[int(tree.data[bifurcation_vessel, 17]), 3:6]) / 2
                            #direction = (tree.data[int(tree.data[bifurcation_vessel, 17]), 3:6] - tree.data[int(tree.data[bifurcation_vessel, 17]), 0:3])
                            #length = np.linalg.norm(direction)
                            #direction = direction / length
                            #cyl = pv.Cylinder(radius=tree.data[int(tree.data[bifurcation_vessel, 17]), 21], center=center, direction=direction,
                            #                  height=length, capping=True)
                            #plotter.add_mesh(cyl, color='blue', label='parent')
                            if int(data[super_parent, 15]) == bifurcation_vessel:
                                pass
                                #parent_vessel_proximity[int(tree.data[super_parent, 16])] = False
                                #proximity_check[int(tree.data[super_parent, 16])] = False
                                #center = (tree.data[int(tree.data[super_parent, 16]), 0:3] + tree.data[int(
                                #    tree.data[super_parent, 16]), 3:6]) / 2
                                #direction = (tree.data[int(tree.data[super_parent, 16]), 3:6] - tree.data[int(
                                #    tree.data[super_parent, 16]), 0:3])
                                #length = np.linalg.norm(direction)
                                #direction = direction / length
                                #cyl = pv.Cylinder(radius=tree.data[int(tree.data[super_parent, 16]), 21],
                                #                  center=center, direction=direction,
                                #                  height=length, capping=True)
                                #plotter.add_mesh(cyl, color='pink', label='parent sister')
                            else:
                                #parent_vessel_proximity[int(tree.data[super_parent, 15])] = False
                                #proximity_check[int(tree.data[super_parent, 15])] = False
                                #center = (tree.data[int(tree.data[super_parent, 15]), 0:3] + tree.data[int(
                                #    tree.data[super_parent, 15]), 3:6]) / 2
                                #direction = (tree.data[int(tree.data[super_parent, 15]), 3:6] - tree.data[int(
                                #    tree.data[super_parent, 15]), 0:3])
                                #length = np.linalg.norm(direction)
                                #direction = direction / length
                                #cyl = pv.Cylinder(radius=tree.data[int(tree.data[super_parent, 15]), 21],
                                #                  center=center, direction=direction,
                                #                  height=length, capping=True)
                                #plotter.add_mesh(cyl, color='pink', label='parent sister')
                                pass
                        #plotter.show()
                        if isinstance(proximity_check, numpy.ndarray):
                            #terminal_vessel_proximity[bifurcation_vessel] = False
                            #proximity_check[bifurcation_vessel] = False
                            pass
                        else:
                            proximity_check = numpy.array([proximity_check])
                            #terminal_vessel_proximity = numpy.array([terminal_vessel_proximity])
                        #if any(terminal_vessel_proximity):
                        if np.any(proximity_check):
                            if obb_any(data[proximity_check, :], terminal_vessel):
                                #midpoints[closest_vessels[j, i], :] = midpoints_base[closest_vessels[j, i], :]
                                end = perf_counter()
                                tree.times['collision'][-1] += end - start
                                continue
                        end_c_1 = perf_counter()
                        tree.times['collision_1'][-1] += end_c_1 - start_c_1
                        start_c_2 = perf_counter()
                        terminal_daughter_vessel = TreeData()
                        terminal_daughter_vessel[0, 0:3] = bifurcation_point
                        terminal_daughter_vessel[0, 3:6] = data[bifurcation_vessel, 3:6]
                        basis_inplace(terminal_daughter_vessel[:, 0:3], terminal_daughter_vessel[:, 3:6],
                                      terminal_daughter_vessel[:, 6:9], terminal_daughter_vessel[:, 9:12],
                                      terminal_daughter_vessel[:, 12:15])
                        terminal_daughter_vessel[0, 15] = data[bifurcation_vessel, 15]
                        terminal_daughter_vessel[0, 16] = data[bifurcation_vessel, 16]
                        terminal_daughter_vessel[0, 17] = bifurcation_vessel
                        terminal_daughter_vessel[0, 20] = np.linalg.norm(terminal_daughter_vessel[0, 3:6] -
                                                                         terminal_daughter_vessel[0, 0:3])
                        terminal_daughter_vessel[0, 21] = data[bifurcation_vessel, 21]
                        terminal_vessel[0, 21] -= tree.physical_clearance
                        terminal_daughter_vessel[0, 21] += tree.physical_clearance
                        search_radius = data[bifurcation_vessel, 20] + 2.0 * data[bifurcation_vessel, 21] + terminal_daughter_vessel[
                            0, 20] / 2 + 2.0 * terminal_daughter_vessel[0, 21]
                        #terminal_daughter_vessel_proximity = sphere_proximity(tree.data, terminal_daughter_vessel[0, :])
                        #terminal_daughter_vessel_proximity = search_tree.query_ball_point((terminal_daughter_vessel[0, 0:3] +
                        #                                                                   terminal_daughter_vessel[0, 3:6])/2,
                        #                                                                   search_radius)
                        #terminal_daughter_vessel_proximity = tree.kdtm.query_ball_point((terminal_daughter_vessel[0, 0:3] +
                        #                                                                 terminal_daughter_vessel[0, 3:6])/2,
                        #                                                                 search_radius.mean())
                        terminal_daughter_vessel_proximity = tree.hnsw_tree.query_ball_point(((terminal_daughter_vessel[0, 0:3] +
                                                                                         terminal_daughter_vessel[0, 3:6])/2).reshape(1,3),
                                                                                         search_radius)
                        #terminal_daughter_vessel_proximity_check = numpy.full((tree.data.shape[0],), False, dtype=bool)
                        proximity_check.fill(False)
                        #terminal_daughter_vessel_proximity_check[terminal_daughter_vessel_proximity] = True
                        #terminal_daughter_vessel_proximity = terminal_daughter_vessel_proximity_check
                        proximity_check[terminal_daughter_vessel_proximity] = True
                        #terminal_daughter_vessel_proximity[bifurcation_vessel] = False
                        proximity_check[bifurcation_vessel] = False
                        #plotter = pv.Plotter()
                        #center = (terminal_daughter_vessel[0, 0:3] + terminal_daughter_vessel[0, 3:6])/2
                        #direction = (terminal_daughter_vessel[0, 3:6] - terminal_daughter_vessel[0, 0:3])
                        #length = np.linalg.norm(direction)
                        #direction = direction/length
                        #cyl = pv.Cylinder(radius=terminal_daughter_vessel[0,21],center=center,
                        #                  direction=direction,height=length,capping=True)
                        #plotter.add_mesh(cyl, color='green', label='new terminal daughter')
                        #center = (tree.data[bifurcation_vessel, 0:3] + tree.data[bifurcation_vessel, 3:6])/2
                        #direction = (tree.data[bifurcation_vessel, 3:6] - tree.data[bifurcation_vessel, 0:3])
                        #length = np.linalg.norm(direction)
                        #direction = direction/length
                        #cyl = pv.Cylinder(radius=tree.data[bifurcation_vessel, 21],center=center,direction=direction,height=length,capping=True)
                        #plotter.add_mesh(cyl, color='red', label='bifurcation vessel')
                        if not numpy.isnan(data[bifurcation_vessel, 15]):
                            #terminal_daughter_vessel_proximity[int(terminal_daughter_vessel[0, 15])] = False
                            proximity_check[int(data[bifurcation_vessel, 15])] = False
                            #center = (tree.data[int(tree.data[bifurcation_vessel, 15]), 0:3] + tree.data[int(tree.data[bifurcation_vessel, 15]), 3:6]) / 2
                            #direction = (tree.data[int(tree.data[bifurcation_vessel, 15]), 3:6] - tree.data[int(tree.data[bifurcation_vessel, 15]), 0:3])
                            #length = np.linalg.norm(direction)
                            #direction = direction / length
                            #cyl = pv.Cylinder(radius=tree.data[int(tree.data[bifurcation_vessel, 15]), 21], center=center, direction=direction,
                            #                  height=length, capping=True)
                            #plotter.add_mesh(cyl, color='yellow', label='left daughter')
                        if not numpy.isnan(data[bifurcation_vessel, 16]):
                            #terminal_daughter_vessel_proximity[int(terminal_daughter_vessel[0, 16])] = False
                            proximity_check[int(data[bifurcation_vessel, 16])] = False
                            #center = (tree.data[int(tree.data[bifurcation_vessel, 16]), 0:3] + tree.data[int(tree.data[bifurcation_vessel, 16]), 3:6]) / 2
                            #direction = (tree.data[int(tree.data[bifurcation_vessel, 16]), 3:6] - tree.data[int(tree.data[bifurcation_vessel, 16]), 0:3])
                            #length = np.linalg.norm(direction)
                            #direction = direction / length
                            #cyl = pv.Cylinder(radius=tree.data[int(tree.data[bifurcation_vessel, 16]), 21], center=center, direction=direction,
                            #                  height=length, capping=True)
                            #plotter.add_mesh(cyl, color='yellow', label='left daughter')
                        if not numpy.isnan(data[bifurcation_vessel, 17]):
                            super_parent = int(data[bifurcation_vessel, 17])
                            #parent_vessel_proximity[int(tree.data[bifurcation_vessel, 17])] = False
                            proximity_check[int(data[bifurcation_vessel, 17])] = False
                            #center = (tree.data[int(tree.data[bifurcation_vessel, 17]), 0:3] + tree.data[int(tree.data[bifurcation_vessel, 17]), 3:6]) / 2
                            #direction = (tree.data[int(tree.data[bifurcation_vessel, 17]), 3:6] - tree.data[int(tree.data[bifurcation_vessel, 17]), 0:3])
                            #length = np.linalg.norm(direction)
                            #direction = direction / length
                            #cyl = pv.Cylinder(radius=data[int(tree.data[bifurcation_vessel, 17]), 21], center=center, direction=direction,
                            #                  height=length, capping=True)
                            #plotter.add_mesh(cyl, color='blue', label='parent')
                            if int(data[super_parent, 15]) == bifurcation_vessel:
                                #parent_vessel_proximity[int(tree.data[super_parent, 16])] = False
                                #proximity_check[int(tree.data[super_parent, 16])] = False
                                #center = (tree.data[int(tree.data[super_parent, 16]), 0:3] + tree.data[int(
                                #    tree.data[super_parent, 16]), 3:6]) / 2
                                #direction = (tree.data[int(tree.data[super_parent, 16]), 3:6] - tree.data[int(
                                #    tree.data[super_parent, 16]), 0:3])
                                #length = np.linalg.norm(direction)
                                #direction = direction / length
                                #cyl = pv.Cylinder(radius=tree.data[int(tree.data[super_parent, 16]), 21],
                                #                  center=center, direction=direction,
                                #                  height=length, capping=True)
                                #plotter.add_mesh(cyl, color='pink', label='parent sister')
                                pass
                            else:
                                #parent_vessel_proximity[int(tree.data[super_parent, 15])] = False
                                #proximity_check[int(tree.data[super_parent, 15])] = False
                                #center = (tree.data[int(tree.data[super_parent, 15]), 0:3] + tree.data[int(
                                #    tree.data[super_parent, 15]), 3:6]) / 2
                                #direction = (tree.data[int(tree.data[super_parent, 15]), 3:6] - tree.data[int(
                                #    tree.data[super_parent, 15]), 0:3])
                                #length = np.linalg.norm(direction)
                                #direction = direction / length
                                #cyl = pv.Cylinder(radius=tree.data[int(tree.data[super_parent, 15]), 21],
                                #                  center=center, direction=direction,
                                #                  height=length, capping=True)
                                #plotter.add_mesh(cyl, color='pink', label='parent sister')
                                pass
                        #plotter.show()
                        if not numpy.isnan(terminal_daughter_vessel[0, 15]):
                            #terminal_daughter_vessel_proximity[int(terminal_daughter_vessel[0, 15])] = False
                            proximity_check[int(terminal_daughter_vessel[0, 15])] = False
                        if not numpy.isnan(terminal_daughter_vessel[0, 16]):
                            #terminal_daughter_vessel_proximity[int(terminal_daughter_vessel[0, 16])] = False
                            proximity_check[int(terminal_daughter_vessel[0, 16])] = False
                        if np.any(proximity_check):
                            if obb_any(data[proximity_check, :], terminal_daughter_vessel):
                                #midpoints[closest_vessels[j, i], :] = midpoints_base[closest_vessels[j, i], :]
                                end = perf_counter()
                                tree.times['collision'][-1] += end - start
                                continue
                        terminal_daughter_vessel[0, 21] -= tree.physical_clearance
                        parent_vessel = TreeData()
                        parent_vessel[0, 0:3] = data[bifurcation_vessel, 0:3]
                        parent_vessel[0, 3:6] = data[bifurcation_vessel, 3:6]
                        basis_inplace(parent_vessel[:, 0:3], parent_vessel[:, 3:6],
                                      parent_vessel[:, 6:9], parent_vessel[:, 9:12],
                                      parent_vessel[:, 12:15])
                        parent_vessel[0, 15] = data.shape[0]
                        parent_vessel[0, 16] = data.shape[0] + 1
                        parent_vessel[0, 17] = data[bifurcation_vessel, 17]
                        parent_vessel[0, 20] = np.linalg.norm(parent_vessel[0, 3:6] -
                                                              parent_vessel[0, 0:3])
                        parent_vessel[0, 21] = data[bifurcation_vessel, 21]
                        parent_vessel[0, 21] += tree.physical_clearance
                        #parent_vessel_proximity = sphere_proximity(tree.data, parent_vessel[0, :])
                        #parent_vessel_proximity = search_tree.query_ball_point((parent_vessel[0, 0:3] + parent_vessel[0, 3:6])/2,
                        #                                                       search_radius)
                        search_radius = data[bifurcation_vessel, 20] + 2.0 * data[bifurcation_vessel, 21] + parent_vessel[
                            0, 20] / 2 + 2.0 * parent_vessel[0, 21]
                        #parent_vessel_proximity = tree.kdtm.query_ball_point((parent_vessel[0, 0:3] + parent_vessel[0, 3:6])/2,
                        #                                                       search_radius.mean())
                        parent_vessel_proximity = tree.hnsw_tree.query_ball_point(((parent_vessel[0, 0:3] + parent_vessel[0, 3:6])/2).reshape(1,3),
                                                                               search_radius)
                        #parent_vessel_proximity_check = numpy.full((tree.data.shape[0],), False, dtype=bool)
                        proximity_check.fill(False)
                        #parent_vessel_proximity_check[parent_vessel_proximity] = True
                        #parent_vessel_proximity = parent_vessel_proximity_check
                        #parent_vessel_proximity[bifurcation_vessel] = False
                        proximity_check[parent_vessel_proximity] = True
                        proximity_check[bifurcation_vessel] = False
                        if not numpy.isnan(data[bifurcation_vessel, 15]):
                            #terminal_daughter_vessel_proximity[int(terminal_daughter_vessel[0, 15])] = False
                            proximity_check[int(data[bifurcation_vessel, 15])] = False
                        if not numpy.isnan(data[bifurcation_vessel, 16]):
                            #terminal_daughter_vessel_proximity[int(terminal_daughter_vessel[0, 16])] = False
                            proximity_check[int(data[bifurcation_vessel, 16])] = False
                        if not numpy.isnan(data[bifurcation_vessel, 17]):
                            super_parent = int(data[bifurcation_vessel, 17])
                            #parent_vessel_proximity[int(tree.data[bifurcation_vessel, 17])] = False
                            proximity_check[int(data[bifurcation_vessel, 17])] = False
                            if int(data[super_parent, 15]) == bifurcation_vessel:
                                #parent_vessel_proximity[int(tree.data[super_parent, 16])] = False
                                proximity_check[int(data[super_parent, 16])] = False
                            else:
                                #parent_vessel_proximity[int(tree.data[super_parent, 15])] = False
                                proximity_check[int(data[super_parent, 15])] = False
                        if np.any(proximity_check):
                            if obb_any(data[proximity_check, :], parent_vessel):
                                #midpoints[closest_vessels[j, i], :] = midpoints_base[closest_vessels[j, i], :]
                                end = perf_counter()
                                tree.times['collision'][-1] += end - start
                                #print('collision parent')
                                continue
                        end = perf_counter()
                        end_c_2 = perf_counter()
                        tree.times['collision_2'][-1] += end_c_2 - start_c_2
                        tree.times['collision'][-1] += end - start
                        parent_vessel[0, 21] -= tree.physical_clearance
                        start_3 = perf_counter()
                        create_new_vessels(bifurcation_point, data, terminal_point, terminal_vessel,
                                           terminal_daughter_vessel, parent_vessel, max_distal_node,
                                           numpy.float64(data.shape[0]),
                                           connectivity, bifurcation_vessel, tree.parameters.murray_exponent,
                                           tree.parameters.kinematic_viscosity*tree.parameters.fluid_density, tree.parameters.terminal_flow,
                                           tree.parameters.terminal_pressure, tree.parameters.root_pressure,
                                           tree.parameters.radius_exponent, tree.parameters.length_exponent)
                        start_3_0 = perf_counter()
                        terminal_map = TreeMap()
                        #upstream = numpy.array(sorted(set(tree.vessel_map[bifurcation_vessel]['upstream'])),dtype=int)
                        #downstream = numpy.array(sorted(set(tree.vessel_map[bifurcation_vessel]['downstream'])), dtype=int)
                        #upstream = np.sort(np.unique(tree.vessel_map[bifurcation_vessel]['upstream'])).astype(np.int64)
                        #downstream = np.sort(np.unique(tree.vessel_map[bifurcation_vessel]['downstream'])).astype(np.int64)
                        upstream = sorted(set(tree.vessel_map[bifurcation_vessel]['upstream']))
                        downstream = sorted(set(tree.vessel_map[bifurcation_vessel]['downstream']))
                        terminal_map[data.shape[0]] = {'upstream': [], 'downstream': []}
                        #terminal_map[data.shape[0]]['upstream'] = numpy.append(upstream, numpy.array([bifurcation_vessel]))
                        terminal_map[data.shape[0]]['upstream'] = list(upstream)
                        #print("Before 0: {}".format(terminal_map[tree.data.shape[0]]['upstream']))
                        terminal_map[data.shape[0]]['upstream'].append(bifurcation_vessel)
                        #print("After 0: {}".format(terminal_map[tree.data.shape[0]]['upstream']))
                        terminal_daughter_map = TreeMap()
                        terminal_daughter_map[data.shape[0] + 1] = {'upstream': [], 'downstream': []}
                        terminal_daughter_map[data.shape[0] + 1]['upstream'] = list(upstream)
                        terminal_daughter_map[data.shape[0] + 1]['downstream'] = list(downstream)
                        #terminal_daughter_map[tree.data.shape[0] + 1]['upstream'] = numpy.append(upstream, numpy.array([bifurcation_vessel]))
                        #print("Before: {}".format(terminal_daughter_map[tree.data.shape[0] + 1]['upstream']))
                        terminal_daughter_map[data.shape[0] + 1]['upstream'].append(bifurcation_vessel)
                        #print("After: {}".format(terminal_daughter_map[tree.data.shape[0] + 1]['upstream']))
                        parent_map = TreeMap()
                        parent_map[bifurcation_vessel] = {'upstream': [], 'downstream': []}
                        #parent_map[bifurcation_vessel]['downstream'] = numpy.append(downstream, numpy.array([tree.data.shape[0],tree.data.shape[0] + 1]))
                        #parent_map[bifurcation_vessel]['upstream'] = deepcopy(upstream)
                        #parent_map[bifurcation_vessel]['downstream'] = deepcopy(downstream)
                        parent_map[bifurcation_vessel]['downstream'].append(data.shape[0])
                        parent_map[bifurcation_vessel]['downstream'].append(data.shape[0] + 1)
                        end_3_0 = perf_counter()
                        tree.times['chunk_3_0'][-1] += end_3_0 - start_3_0
                        start_3_1 = perf_counter()
                        #new_vessel_map = deepcopy(tree.vessel_map)
                        #new_vessel_map = tree.vessel_map_copy
                        new_vessel_map = TreeMap()
                        new_vessel_map.update(parent_map)
                        new_vessel_map.update(terminal_map)
                        new_vessel_map.update(terminal_daughter_map)
                        #_, counts = np.unique(parent_map[bifurcation_vessel]['downstream'], return_counts=True)
                        #assert np.all(counts == 1), "Duplicate in parent map downstream"
                        #_, counts = np.unique(parent_map[bifurcation_vessel]['upstream'], return_counts=True)
                        #assert np.all(counts == 1), "Duplicate in parent map upstream"
                        #_, counts = np.unique(terminal_map[tree.data.shape[0]]['downstream'], return_counts=True)
                        #assert np.all(counts == 1), "Duplicate in terminal map downstream"
                        #_, counts = np.unique(terminal_map[tree.data.shape[0]]['upstream'], return_counts=True)
                        #assert np.all(counts == 1), "Duplicate in terminal map upstream"
                        #_, counts = np.unique(terminal_daughter_map[tree.data.shape[0] + 1]['downstream'], return_counts=True)
                        #assert np.all(counts == 1), "Duplicate in terminal daughter map downstream"
                        #_, counts = np.unique(terminal_daughter_map[tree.data.shape[0] + 1]['upstream'], return_counts=True)
                        #assert np.all(counts == 1), "Duplicate in terminal daughter map upstream"
                        end_3_1 = perf_counter()
                        tree.times['chunk_3_1'][-1] += end_3_1 - start_3_1
                        start_3_2 = perf_counter()
                        added_vessels = [terminal_vessel, terminal_daughter_vessel, parent_vessel]
                        #new_vessels = tree.data.copy(order='C')
                        tmp_28 = data[:, 28].copy()
                        #new_vessels = deepcopy(tree.data)
                        change_i = []
                        change_j = []
                        new_data = []
                        old_data = []
                        end_3_2 = perf_counter()
                        tree.times['chunk_3_2'][-1] += end_3_2 - start_3_2
                        start_3_3 = perf_counter()
                        #connectivity = numpy.nan_to_num(tree.data[:, 15:18], nan=-1.0).astype(int)
                        connectivity = tree.connectivity.copy()
                        #if (np.any(connectivity != connectivity_2)):
                        #    print('Connectivity mismatch!')
                        #    print('Connectivity: ', connectivity)
                        #    print('Connectivity_2: ', connectivity_2)
                        #    raise ValueError('Connectivity mismatch!')
                        results = update_vessels(bifurcation_point, data, terminal_point,
                                                 connectivity, bifurcation_vessel, tree.parameters.murray_exponent,
                                                 tree.parameters.kinematic_viscosity * tree.parameters.fluid_density,
                                                 tree.parameters.terminal_flow,
                                                 tree.parameters.terminal_pressure, tree.parameters.root_pressure,
                                                 tree.parameters.radius_exponent, tree.parameters.length_exponent)
                        end_3_3 = perf_counter()
                        tree.times['chunk_3_3'][-1] += end_3_3 - start_3_3
                        start_3_4 = perf_counter()
                        reduced_resistance = numpy.array(results[0])
                        reduced_length = numpy.array(results[1])
                        main_idx = results[2]
                        main_scale = numpy.array(results[3])
                        alt_idx = results[4]
                        alt_scale = numpy.array(results[5])
                        bifurcation_ratios = numpy.array(results[6])
                        flows = numpy.array(results[7])
                        root_radius = results[8]
                        #new_vessels[0, 21] = root_radius
                        # --- BEGIN: preallocated change buffers (NumPy) ---
                        main_idx = np.asarray(main_idx, dtype=np.intp)
                        alt_idx = np.asarray(alt_idx, dtype=np.intp)
                        alt_scale = np.asarray(alt_scale, dtype=data.dtype, order="C")
                        flows = np.asarray(flows, dtype=data.dtype, order="C")
                        reduced_resistance = np.asarray(reduced_resistance, dtype=data.dtype, order="C")
                        reduced_length = np.asarray(reduced_length, dtype=data.dtype, order="C")
                        main_scale = np.asarray(main_scale, dtype=data.dtype, order="C")
                        bifurcation_ratios = np.asarray(bifurcation_ratios, dtype=data.dtype, order="C")
                        if bifurcation_ratios.ndim == 1:
                            #bifurcation_ratios = bifurcation_ratios.reshape(1, 2)
                            bifurcation_ratios = np.empty((1, 2), dtype=float)
                        N = data.shape[0]
                        W = data.shape[1]
                        downstream_bif = tree.vessel_map[bifurcation_vessel]["downstream"]
                        D0 = len(downstream_bif)

                        has_children = not np.any(np.isnan(terminal_daughter_vessel[0, 15:17]))
                        child_updates = 2 if has_children else 0  # matches current logic (either 2 or 0)

                        alt_valid_mask = alt_idx > -1
                        alt_valid = alt_idx[alt_valid_mask]
                        alt_scale_valid = alt_scale[alt_valid_mask]

                        # Count the exact number of updates that update_alt would emit (python fallback semantics)
                        alt_desc_total = 0
                        for idx in alt_valid:
                            alt_desc_total += len(tree.vessel_map[int(idx)]["downstream"])

                        n_update_alt = 1 + (6 * main_idx.size) + alt_desc_total + alt_valid.size
                        n_chunk_3_5 = (2 * D0) + child_updates
                        n_chunk_3_6 = N + W
                        n_total = n_update_alt + n_chunk_3_5 + n_chunk_3_6

                        change_i = np.empty(n_total, dtype=np.intp)
                        change_j = np.empty(n_total, dtype=np.intp)
                        new_data = np.empty(n_total, dtype=data.dtype)
                        old_data = np.empty(n_total, dtype=data.dtype)
                        p = 0

                        #change_i.append(0)
                        #change_j.append(21)
                        #new_data.append(root_radius)
                        #old_data.append(data[0, 21])

                        change_i[p] = 0
                        change_j[p] = 21
                        new_data[p] = root_radius
                        old_data[p] = data[0, 21]
                        p += 1

                        # Main idx updates: (22,25,27,28,23,24) for each main_idx
                        if main_idx.size:
                            cols_main = np.array([22, 25, 27, 28, 23, 24], dtype=np.intp)
                            ncols = cols_main.size
                            rows = np.repeat(main_idx, ncols)
                            cols = np.tile(cols_main, main_idx.size)

                            new_mat = np.column_stack(
                                (
                                    flows,
                                    reduced_resistance,
                                    reduced_length,
                                    main_scale,
                                    bifurcation_ratios[:, 0],
                                    bifurcation_ratios[:, 1],
                                )
                            ).astype(data.dtype, copy=False)
                            old_mat = data[main_idx][:, cols_main].astype(data.dtype, copy=False)

                            n = rows.size
                            change_i[p: p + n] = rows
                            change_j[p: p + n] = cols
                            new_data[p: p + n] = new_mat.ravel(order="C")
                            old_data[p: p + n] = old_mat.ravel(order="C")
                            p += n

                            tmp_28[main_idx] = main_scale


                        #tmp_28_copy = deepcopy(tmp_28)
                        #print("bifurcation: {}".format(bifurcation_ratios.shape))
                        #if len(bifurcation_ratios.shape) == 1:
                        #    bifurcation_ratios = np.empty((1,2),dtype=float)
                        start_chunk_3_4_alt = perf_counter()
                        #res_test = update_alt(reduced_resistance, reduced_length,
                        #                      main_idx, main_scale, alt_idx,
                        #                      alt_scale, bifurcation_ratios,
                        #                      flows, root_radius, data,
                        #                      tmp_28, tree.vessel_map)
                        #change_i = res_test[0]
                        #change_j = res_test[1]
                        #new_data = res_test[2]
                        #old_data = res_test[3]
                        # Alt subtree updates (col 28 for downstream + the alt root itself), and mutate tmp_28
                        for idx, scale in zip(alt_valid, alt_scale_valid):
                            idx_int = int(idx)
                            ds_list = tree.vessel_map[idx_int]["downstream"]
                            if ds_list:
                                ds = np.asarray(ds_list, dtype=np.intp)
                                m = ds.size
                                change_i[p: p + m] = ds
                                change_j[p: p + m] = 28

                                old_vals = data[ds, 28]
                                old_data[p: p + m] = old_vals
                                factor = scale / data[idx_int, 28]
                                new_data[p: p + m] = old_vals * factor
                                p += m

                                tmp_28[ds] *= factor

                            change_i[p] = idx_int
                            change_j[p] = 28
                            old_data[p] = data[idx_int, 28]
                            new_data[p] = scale
                            p += 1

                            tmp_28[idx_int] = scale

                        end_chunk_3_4_alt = perf_counter()
                        #start_3_4 = perf_counter()
                        tree.times['chunk_3_4_alt'][-1] += end_chunk_3_4_alt - start_chunk_3_4_alt
                        """
                        if len(main_idx) > 0:
                            # Flows
                            change_i.extend(main_idx)
                            change_j.extend([22]*len(main_idx))
                            new_data.extend(flows.tolist())
                            old_data.extend(tree.data[main_idx, 22].tolist())
                            #new_vessels[main_idx, 22] = flows
                            # Reduced Resistance
                            change_i.extend(main_idx)
                            change_j.extend([25]*len(main_idx))
                            new_data.extend(reduced_resistance.tolist())
                            old_data.extend(tree.data[main_idx, 25].tolist())
                            #new_vessels[main_idx, 25] = reduced_resistance
                            # Reduced lengths
                            #new_vessels[main_idx, 27] = reduced_length
                            change_i.extend(main_idx)
                            change_j.extend([27]*len(main_idx))
                            new_data.extend(reduced_length.tolist())
                            old_data.extend(tree.data[main_idx, 27].tolist())
                            # Radius scaling
                            change_i.extend(main_idx)
                            change_j.extend([28]*len(main_idx))
                            new_data.extend(main_scale.tolist())
                            old_data.extend(tree.data[main_idx, 28].tolist())
                            #new_vessels[main_idx, 28] = main_scale
                            tmp_28[main_idx] = main_scale
                            # Bifurcations
                            change_i.extend(main_idx)
                            change_j.extend([23]*len(main_idx))
                            new_data.extend(bifurcation_ratios[:,0].tolist())
                            old_data.extend(tree.data[main_idx, 23].tolist())
                            #new_vessels[main_idx, 23] = bifurcation_ratios[:, 0]
                            change_i.extend(main_idx)
                            change_j.extend([24]*len(main_idx))
                            new_data.extend(bifurcation_ratios[:,1].tolist())
                            old_data.extend(tree.data[main_idx, 24].tolist())
                            #new_vessels[main_idx, 24] = bifurcation_ratios[:, 1]
                        for k in range(len(alt_idx)):
                            if alt_idx[k] > -1:
                                downstream = tree.vessel_map[alt_idx[k]]['downstream']
                                #_, counts = np.unique(downstream, return_counts=True)
                                #if np.any(counts > 1):
                                #    print("DOUBLE COUNT!!!!!!!!!")
                                if len(tree.vessel_map[alt_idx[k]]['downstream']) > 0:
                                    #new_vessels[downstream, 28] /= new_vessels[alt_idx[k], 28]
                                    #new_vessels[alt_idx[k], 28] = alt_scale[k]
                                    #new_vessels[downstream, 28] *= new_vessels[alt_idx[k], 28]
                                    #new_vessels[downstream, 28] *= (alt_scale[k]/new_vessels[alt_idx[k], 28])
                                    tmp_28[downstream] *= (alt_scale[k]/tree.data[alt_idx[k], 28])
                                    change_i.extend(downstream)
                                    change_j.extend([28]*len(downstream))
                                    new_data.extend((tree.data[downstream, 28] * (alt_scale[k]/tree.data[alt_idx[k], 28])).tolist())
                                    old_data.extend(tree.data[downstream, 28].tolist())
                                    #new_vessels[alt_idx[k], 28] = alt_scale[k]
                                    tmp_28[alt_idx[k]] = alt_scale[k]
                                    change_i.append(alt_idx[k])
                                    change_j.append(28)
                                    new_data.append(alt_scale[k])
                                    old_data.append(tree.data[alt_idx[k], 28])
                                else:
                                    #new_vessels[alt_idx[k], 28] = alt_scale[k]
                                    tmp_28[alt_idx[k]] = alt_scale[k]
                                    change_i.append(alt_idx[k])
                                    change_j.append(28)
                                    new_data.append(alt_scale[k])
                                    old_data.append(tree.data[alt_idx[k], 28])
                        #if not np.all(np.isclose(change_i, res_test[0])):
                        #    print("change_i {} != \nnew change_i:{}".format(change_i, res_test[0]))
                        #if not np.all(np.isclose(change_j, res_test[1])):
                        #    print("change_i {} != \nnew change_i:{}".format(change_j, res_test[1]))
                        #if not np.all(np.isclose(new_data, res_test[2])):
                        #    print("change_i {} != \nnew change_i:{}".format(new_data, res_test[2]))
                        #if not np.all(np.isclose(old_data, res_test[3])):
                        #    print("change_i {} != \nnew change_i:{}".format(old_data, res_test[3]))
                        assert np.all(np.isclose(tmp_28,tmp_28_copy)), "tmp_28: {} != \ntmp_28_copy:{}".format(tmp_28[~np.isclose(tmp_28,tmp_28_copy)],tmp_28_copy[~np.isclose(tmp_28,tmp_28_copy)])
                        """
                        end_3_4 = perf_counter()
                        tree.times['chunk_3_4'][-1] += end_3_4 - start_3_4
                        start_3_5 = perf_counter()
                        """
                        if len(tree.vessel_map[bifurcation_vessel]['downstream']) > 0:
                            # new changes to add !!!!!!!!!!!!!!!!!!!!!!!
                            change_i.extend(tree.vessel_map[bifurcation_vessel]['downstream'])
                            change_j.extend([28]*len(tree.vessel_map[bifurcation_vessel]['downstream']))
                            tmp_new_data = (data[tree.vessel_map[bifurcation_vessel]['downstream'], 28] /
                                            data[bifurcation_vessel, 28])*terminal_daughter_vessel[0, 28]
                            new_data.extend(tmp_new_data.tolist())
                            old_data.extend(data[tree.vessel_map[bifurcation_vessel]['downstream'], 28].tolist())
                            #new_vessels[tree.vessel_map[bifurcation_vessel]['downstream'], 28] /= new_vessels[
                            #    bifurcation_vessel, 28]
                            tmp_28[tree.vessel_map[bifurcation_vessel]['downstream']] /= data[bifurcation_vessel, 28]
                            #new_vessels[tree.vessel_map[bifurcation_vessel]['downstream'], 28] *= \
                            #terminal_daughter_vessel[0, 28]
                            tmp_28[tree.vessel_map[bifurcation_vessel]['downstream']] *= terminal_daughter_vessel[0, 28]
                            change_i.extend(tree.vessel_map[bifurcation_vessel]['downstream'])
                            change_j.extend([26]*len(tree.vessel_map[bifurcation_vessel]['downstream']))
                            new_data.extend((data[tree.vessel_map[bifurcation_vessel]['downstream'], 26] + 1.0).tolist())
                            old_data.extend(data[tree.vessel_map[bifurcation_vessel]['downstream'], 26].tolist())
                            #new_vessels[tree.vessel_map[bifurcation_vessel]['downstream'], 26] += 1.0
                        #print('Bifurcation Vessel Upstream: ', new_vessel_map[bifurcation_vessel]['upstream'])
                        for k in tree.vessel_map[bifurcation_vessel]['upstream']:
                            #assert k != bifurcation_vessel, "reflexive insertion of bifurcation vessel"
                            #new_vessel_map[k]['downstream'].append(tree.data.shape[0])
                            #new_vessel_map[k]['downstream'].append(tree.data.shape[0] + 1)
                            new_vessel_map[k] = {'upstream': [], 'downstream': []}
                            new_vessel_map[k]['downstream'].extend([data.shape[0], data.shape[0]+1])
                            #new_vessel_map[k]['downstream'] = numpy.concatenate((new_vessel_map[k]['downstream'],
                            #                                                     numpy.array([tree.data.shape[0],
                            #                                                           tree.data.shape[0] + 1])))
                        if not numpy.any(numpy.isnan(terminal_daughter_vessel[0, 15:17])):
                            if not numpy.isnan(terminal_daughter_vessel[0, 15]):
                                change_i.append(int(terminal_daughter_vessel[0, 15]))
                                change_j.append(17)
                                new_data.append(data.shape[0] + 1)
                                old_data.append(data[int(terminal_daughter_vessel[0, 15]), 17])
                                #new_vessels[int(terminal_daughter_vessel[0, 15]), 17] = tree.data.shape[0] + 1
                            if not numpy.isnan(terminal_daughter_vessel[0, 16]):
                                change_i.append(int(terminal_daughter_vessel[0, 16]))
                                change_j.append(17)
                                new_data.append(data.shape[0] + 1)
                                old_data.append(data[int(terminal_daughter_vessel[0, 15]), 17])
                                #new_vessels[int(terminal_daughter_vessel[0, 16]), 17] = tree.data.shape[0] + 1
                        #for k in terminal_daughter_map[int(tree.data.shape[0] + 1)]['downstream']:
                        for k in tree.vessel_map[bifurcation_vessel]['downstream']:
                            #new_vessel_map[k]['upstream'].append(int(tree.data.shape[0] + 1))
                            new_vessel_map[k] = {'upstream': [], 'downstream': []}
                            new_vessel_map[k]['upstream'].append(int(data.shape[0] + 1))
                            #print("key: {} add upstream: {}".format(k, int(tree.data.shape[0] + 1)))
                            #new_vessel_map[k]['upstream'] = numpy.concatenate((new_vessel_map[k]['upstream'],
                            #                                                   numpy.array([int(tree.data.shape[0] + 1)])))
                        #new_vessels[:, 21] = new_vessels[0, 21] * new_vessels[:, 28]
                        """
                        # --- chunk_3_5: downstream_bif adjustments (cols 28 and 26) ---
                        if D0:
                            ds = np.asarray(downstream_bif, dtype=np.intp)
                            m = ds.size

                            # col 28 updates
                            change_i[p: p + m] = ds
                            change_j[p: p + m] = 28
                            old_vals_28 = data[ds, 28]
                            old_data[p: p + m] = old_vals_28
                            new_data[p: p + m] = (old_vals_28 / data[bifurcation_vessel, 28]) * \
                                                 terminal_daughter_vessel[0,
                                                 28]
                            p += m

                            tmp_28[ds] /= data[bifurcation_vessel, 28]
                            tmp_28[ds] *= terminal_daughter_vessel[0, 28]

                            # col 26 updates
                            change_i[p: p + m] = ds
                            change_j[p: p + m] = 26
                            old_vals_26 = data[ds, 26]
                            old_data[p: p + m] = old_vals_26
                            new_data[p: p + m] = old_vals_26 + 1.0
                            p += m

                        # parent (col 17) rewiring when bifurcation_vessel had two children
                        if has_children:
                            left = int(terminal_daughter_vessel[0, 15])
                            right = int(terminal_daughter_vessel[0, 16])
                            new_parent = data.shape[0] + 1

                            change_i[p: p + 2] = (left, right)
                            change_j[p: p + 2] = 17
                            new_data[p: p + 2] = new_parent
                            # NOTE: keep existing behavior/bug: old_data uses [15] for both entries
                            old_parent = data[left, 17]
                            old_data[p: p + 2] = (old_parent, old_parent)
                            p += 2

                        end_3_5 = perf_counter()
                        tree.times['chunk_3_5'][-1] += end_3_5 - start_3_5
                        start_3_6 = perf_counter()
                        """
                        #tmp_radii = np.zeros((data.shape[0], 1))
                        tmp_radii = np.empty(data.shape[0], dtype=data.dtype)
                        if tree.n_terminals < 10000:
                            #np.multiply(new_vessels[:, 28], new_vessels[0, 21], out=new_vessels[:, 21])
                            #np.multiply(tmp_28, root_radius, out=tmp_radii[:, 0])
                            np.multiply(tmp_28, root_radius, out=tmp_radii)
                        else:
                            #ne_multiply(new_vessels[:, 28], new_vessels[0, 21], new_vessels[:, 21])
                            #ne_multiply(tmp_28, root_radius, tmp_radii[:, 0])
                            ne_multiply(tmp_28, root_radius, tmp_radii)
                            #scale_column_with_multiply(new_vessels, 28, new_vessels[0, 21], 21)
                            #multiply_columns(new_vessels)
                        #new_vessels[:, 21] = multiply_elements(new_vessels[:, 28], new_vessels[0, 21])
                        #ne.set_num_threads(ne.ncores)
                        #new_vessels[:, 21] = ne.evaluate('v28 * scalar', local_dict={'v28': new_vessels[:, 27],
                        #                                                            'scalar': new_vessels[0, 21]})
                        #if not np.all(np.isclose(tmp_28,new_vessels[:, 28])):
                        #    print('col 28 mismatch')
                        #idxs = np.arange(data.shape[0]).astype(int)
                        #change_i.extend(idxs.tolist())
                        change_i.extend(tree._idx_cache)
                        #change_j.extend([21]*data.shape[0])
                        change_j.extend(tree._col21_cache)
                        #new_data.extend(tmp_radii.flatten().tolist())
                        new_data.extend(tmp_radii.tolist())
                        old_data.extend(data[:, 21].tolist())
                        #new_vessels[bifurcation_vessel, :] = parent_vessel
                        change_i.extend([bifurcation_vessel]*data.shape[1])
                        change_j.extend(np.arange(data.shape[1]).astype(int).tolist())
                        new_data.extend(parent_vessel[0, :].tolist())
                        old_data.extend(data[bifurcation_vessel, :].tolist())
                        appended_vessels = numpy.vstack([terminal_vessel, terminal_daughter_vessel])
                        #new_vessels = numpy.vstack([new_vessels, appended_vessels])
                        #new_vessels[-2, :] = terminal_vessel
                        #new_vessels[-1, :] = terminal_daughter_vessel
                        """
                        # --- chunk_3_6: radii for all existing vessels + parent row overwrite ---
                        tmp_radii = np.empty(N, dtype=data.dtype)
                        if tree.n_terminals < 10000:
                            np.multiply(tmp_28, root_radius, out=tmp_radii)
                        else:
                            ne_multiply(tmp_28, root_radius, tmp_radii)

                        rows = np.arange(N, dtype=np.intp)
                        change_i[p: p + N] = rows
                        change_j[p: p + N] = 21
                        new_data[p: p + N] = tmp_radii
                        old_data[p: p + N] = data[:, 21]
                        p += N

                        cols = np.arange(W, dtype=np.intp)
                        change_i[p: p + W] = bifurcation_vessel
                        change_j[p: p + W] = cols
                        new_data[p: p + W] = parent_vessel[0, :]
                        old_data[p: p + W] = data[bifurcation_vessel, :]
                        p += W

                        end_3_6 = perf_counter()
                        tree.times['chunk_3_6'][-1] += end_3_6 - start_3_6
                        start_3_7 = perf_counter()
                        #print("Bifurcation Vessel: ", bifurcation_vessel)
                        #print("Connectivity: ", tree.connectivity_copy)
                        #connectivity[bifurcation_vessel, :] = np.nan_to_num(tree.data[bifurcation_vessel, 15:18], nan=-1.0).astype(int)
                        connectivity[bifurcation_vessel, :] = np.nan_to_num(parent_vessel[0, 15:18],
                                                                            nan=-1.0).astype(int)
                        if not numpy.isnan(terminal_daughter_vessel[0, 15]):
                            connectivity[int(terminal_daughter_vessel[0, 15]), -1] = data.shape[0] + 1
                        if not numpy.isnan(terminal_daughter_vessel[0, 16]):
                            connectivity[int(terminal_daughter_vessel[0, 16]), -1] = data.shape[0] + 1
                        connectivity = numpy.vstack((connectivity,
                                                     np.nan_to_num(terminal_vessel[:,15:18], nan=-1.0).astype(int).reshape(1,3),
                                                     np.nan_to_num(terminal_daughter_vessel[:, 15:18], nan=-1.0).astype(int)))
                        #tree.connectivity_copy[-2, :] = np.nan_to_num(terminal_vessel[:, 15:18], nan=-1.0).astype(int).reshape(1,3)
                        #tree.connectivity_copy[-1, :] = np.nan_to_num(terminal_daughter_vessel[:, 15:18], nan=-1.0).astype(int)
                        #tree.kdtm.start_update((new_vessels[:,0:3]+new_vessels[:,3:6])/2)
                        success = True
                        end_3_7 = perf_counter()
                        tree.times['chunk_3_7'][-1] += end_3_7 - start_3_7
                        end_3 = perf_counter()
                        tree.times['chunk_3'][-1] += end_3 - start_3
                        new_vessels = None
                        break
                    if success:
                        break
                if not success:
                    threshold *= threshold_adjuster
                    #print('un-ideal threshold adjustment')
        else:
            success = False
            #pts = np.vstack((tree.data[:, 0:3], (tree.data[:, 0:3] + tree.data[:, 3:6])/2, tree.data[:, 3:6]))
            #search_tree = cKDTree(pts)
            volume_threshold = 1.5*tree.domain.mesh.volume ** (1 / 3)
            first_pass = True
            count = 0
            while not success:
                get_points_start = perf_counter()
                if first_pass:
                    terminal_points, terminal_point_distances, closest_vessels, mesh_cells = get_points(tree, n_points, threshold=threshold,
                                                                           interior_range=interior_range,
                                                                           n_vessels=n_closest_vessels)
                    first_pass = False
                else:
                    threshold *= threshold_adjuster
                    volume_threshold *= threshold_adjuster
                    count += 1
                    #if count > 5:
                    #    volume_threshold *= threshold_adjuster
                    #    count = 0
                    if volume_threshold < threshold:
                        volume_threshold = 1.5*threshold
                    #print(f"threshold: {threshold}, volume_threshold: {volume_threshold}")
                    terminal_points, terminal_point_distances, closest_vessels, mesh_cells = get_points(tree, n_points, volume_threshold=volume_threshold,
                                                                                            threshold=threshold,
                                                                                            interior_range=interior_range,
                                                                                            n_vessels=n_closest_vessels)
                    assert volume_threshold > threshold, "Volume threshold is not greater than threshold."
                    #search_tree=search_tree)
                if numpy.all(numpy.isnan(terminal_points)):
                    volume_threshold *= threshold_adjuster
                    threshold *= threshold_adjuster
                    continue
                elif numpy.any(numpy.isnan(terminal_points)):
                    terminal_point_distances = terminal_point_distances[:, ~numpy.isnan(terminal_points).any(axis=1)]
                    closest_vessels = closest_vessels[:, ~numpy.isnan(terminal_points).any(axis=1)]
                    mesh_cells = mesh_cells[~numpy.isnan(terminal_points).any(axis=1)]
                    terminal_points = terminal_points[~numpy.isnan(terminal_points).any(axis=1)]
                #closest_vessels = numpy.argsort(terminal_point_distances, axis=0)
                get_points_end = perf_counter()
                tree.times['get_points'][-1] += get_points_end - get_points_start
                n_closest_vessels = min(n_closest_vessels, data.shape[0])
                for i in range(terminal_points.shape[0]):
                    for j in range(n_closest_vessels):
                        start_1 = perf_counter()
                        if flow_ratio is not None:
                            if (data[closest_vessels[j, i], 22] / tree.parameters.terminal_flow) > flow_ratio:
                                continue
                        cost, triad, vol = construct_optimizer(tree, terminal_points[i, :], closest_vessels[j, i])
                        bifurcation_cell = mesh_cells[i]
                        if callback:
                            history = []
                            lines = numpy.zeros((6, 3), dtype=numpy.float64)
                            lines[0, :] = data[closest_vessels[j, i], 0:3]
                            lines[1, :] = data[closest_vessels[j, i], 3:6]
                            lines[2, :] = data[closest_vessels[j, i], 0:3]
                            lines[3, :] = terminal_points[i, :]
                            lines[4, :] = data[closest_vessels[j, i], 3:6]
                            lines[5, :] = terminal_points[i, :]
                            def callback(xk, history=history):
                                history.append(triad(xk))
                        else:
                            lines = []
                            def callback(xk):
                                pass
                        # [TODO] we need to add a brute force option here for optimization on a grid
                        end_1 = perf_counter()
                        tree.times['chunk_1'][-1] += end_1 - start_1
                        start = perf_counter()
                        if use_brute:
                            result = brute(cost, [(0.0, 1.0), (0.0, 1.0)], Ns=max_iter)
                            bifurcation_point = triad(result)
                            tree.new_tree_scale = vol(result)
                        else:
                            cons = [{"type": "ineq", "fun": lambda a: 1 - a[0] - a[1]}]
                            result = minimize(cost, x0, bounds=[(0.0, 1.0), (0, 1.0)],
                                              options={'maxiter': max_iter}, constraints=cons, method="L-BFGS-B")
                            #result = minimize(cost, x0, bounds=[(0.0, 1.0), (0.0, 1.0)], callback=callback,
                            #                  options={'maxiter':max_iter})
                            if not result.success:
                                #print('Failure in optimization')
                                #print(result.message)
                                continue
                            bifurcation_point = triad(result.x)
                            tree.new_tree_scale = vol(result.x)
                        end = perf_counter()
                        tree.times['local_optimization'][-1] += end - start
                        start_2 = perf_counter()
                        #midpoints = (tree.data_copy[:-2, 0:3] + tree.data_copy[:-2, 3:6])/2
                        #midpoints[closest_vessels[j, i], :] = (tree.data_copy[closest_vessels[j, i], 0:3] + bifurcation_point)/2
                        #midpoints = numpy.vstack((midpoints, ((terminal_points[i, :] + bifurcation_point)/2),
                        #                                     (tree.data_copy[closest_vessels[j, i], 3:6] + bifurcation_point)/2))
                        #tree.kdtm.start_update(midpoints)
                        bifurcation_point_value = tree.domain(bifurcation_point.reshape(1, -1))
                        #plotter = tree.show(plot_domain=True, return_plotter=True)
                        #cy1 = pv.Cylinder(center=(tree.data[closest_vessels[j, i], 0:3] + bifurcation_point) / 2,
                        #                  direction=(bifurcation_point - tree.data[closest_vessels[j, i], 0:3]),
                        #                  radius=tree.data[closest_vessels[j,i], 21],
                        #                  height=numpy.linalg.norm(bifurcation_point - tree.data[closest_vessels[j, i], 0:3]))
                        #cy2 = pv.Cylinder(center=(tree.data[closest_vessels[j, i], 3:6] + bifurcation_point) / 2,
                        #                  direction=(bifurcation_point - tree.data[closest_vessels[j, i], 3:6]),
                        #                  radius=tree.data[closest_vessels[j,i], 21],
                        #                  height=numpy.linalg.norm(bifurcation_point - tree.data[closest_vessels[j, i], 3:6]))
                        #cy3 = pv.Cylinder(center=(terminal_points[i, :] + bifurcation_point) / 2,
                        #                  direction=(bifurcation_point - terminal_points[i,:]),
                        #                  radius=tree.data[closest_vessels[j,i], 21],
                        #                  height=numpy.linalg.norm(bifurcation_point - terminal_points[i,:]))
                        #plotter.add_mesh(cy1, color='green')
                        #plotter.add_mesh(cy2, color='green')
                        #plotter.add_mesh(cy3, color='green')
                        #plotter.show()
                        #print('Bifurcation Point: ', bifurcation_point)
                        #print('Bifurcation Point Value: ', bifurcation_point_value)
                        bifurcation_vessel = closest_vessels[j, i]
                        if numpy.any(bifurcation_point_value > interior_range[1]):
                            #print('Bifurcation point GREATER THAN interior range')
                            continue
                        if numpy.any(bifurcation_point_value < interior_range[0]):
                            #print('Bifurcation point LESS THAN interior range')
                            continue
                        terminal_point = terminal_points[i, :]
                        dist = close_exact_point(data[bifurcation_vessel, :].reshape(1,data.shape[1]),
                                          terminal_point)
                        if dist < data[bifurcation_vessel, 21]*2:
                            #print('too close')
                            continue

                        dist_bifurcation_to_proximal = np.linalg.norm(bifurcation_point.reshape(1,-1) - data[closest_vessels[j, i], 0:3].reshape(1, -1)).flatten()
                        if dist_bifurcation_to_proximal < data[bifurcation_vessel,21]*2:
                            #print('too close to proximal')
                            continue

                        dist_bifurcation_to_distal = np.linalg.norm(bifurcation_point.reshape(1,-1) - data[closest_vessels[j, i], 3:6].reshape(1, -1)).flatten()
                        if dist_bifurcation_to_distal < data[bifurcation_vessel,21]*2:
                            #print('too close to distal')
                            continue

                        line = numpy.linspace(0, 1, nonconvex_sampling).reshape(-1, 1)
                        interior_terminal = tree.domain(terminal_points[i, :].reshape(1, -1)) < interior_range[1]
                        interior_bifurcation = tree.domain(bifurcation_point.reshape(1, -1)) < interior_range[1]
                        interior_proximal = tree.domain(data[closest_vessels[j, i], 0:3].reshape(1, -1)) < interior_range[1]
                        interior_distal = tree.domain(data[closest_vessels[j, i], 3:6].reshape(1, -1)) < interior_range[1]
                        if interior_bifurcation and interior_terminal:
                            terminal_line = bifurcation_point * line + terminal_points[i, :] * (1 - line)
                            values = tree.domain(terminal_line)
                            count_diff = numpy.sum(numpy.abs(numpy.diff(numpy.sign(values.flatten() - interior_range[1]) / 2)))
                            count_outside = values.flatten() > interior_range[1]
                            #if numpy.any(values.flatten() > interior_range[1]):
                            if count_diff > 1:
                                nonconvex_outside = True
                                #print('Vessel outside interior range (interior terminal)')
                                #print(f"count: {count}; count_outside: {numpy.sum(count_outside)}")
                                #plotter = pv.Plotter()
                                #plotter.add_mesh(tree.domain.boundary,opacity=0.2)
                                #plotter.add_points(terminal_line[count_outside], color='red', point_size=10,
                                #                   render_points_as_spheres=True)
                                #plotter.add_points(terminal_line[~count_outside], color='green', point_size=10,
                                #                   render_points_as_spheres=True)
                                #plotter.show()
                                #print("too many interior sign changes")
                                continue
                        else:
                            terminal_line = bifurcation_point * line + terminal_points[i, :] * (1 - line)
                            values = tree.domain(terminal_line)
                            count_diff = numpy.sum(numpy.abs(numpy.diff(numpy.sign(values.flatten() - interior_range[1])/2)))
                            count_outside = values.flatten() > interior_range[1]
                            if count_diff > 1:
                                nonconvex_outside = True
                                #print('Vessel outside interior range 2 (interior terminal)')
                                #plotter = pv.Plotter()
                                #plotter.add_mesh(tree.domain.boundary,opacity=0.2)
                                #plotter.add_points(terminal_line[count_outside], color='red', point_size=10,
                                #                   render_points_as_spheres=True)
                                #plotter.add_points(terminal_line[~count_outside], color='green', point_size=10,
                                #                   render_points_as_spheres=True)
                                #plotter.show()
                                continue
                        if interior_bifurcation and interior_proximal:
                            proximal_line = (data[closest_vessels[j, i], 0:3] * line +
                                             bifurcation_point * (1 - line))
                            values = tree.domain(proximal_line)
                            count_diff = numpy.sum(numpy.abs(numpy.diff(numpy.sign(values.flatten() - interior_range[1]) / 2)))
                            count_outside = values.flatten() > interior_range[1]
                            #if numpy.any(values > interior_range[1]):
                            if count_diff > 1:
                                nonconvex_outside = True
                                #print('Vessel outside interior range (interior proximal)')
                                #plotter = pv.Plotter()
                                #plotter.add_mesh(tree.domain.boundary,opacity=0.2)
                                #plotter.add_points(proximal_line[count_outside], color='red', point_size=10,
                                #                   render_points_as_spheres=True)
                                #plotter.add_points(proximal_line[~count_outside], color='green', point_size=10,
                                #                   render_points_as_spheres=True)
                                #plotter.show()
                                continue
                        else:
                            proximal_line = (data[closest_vessels[j, i], 0:3] * line +
                                             bifurcation_point * (1 - line))
                            values = tree.domain(proximal_line)
                            count_diff = numpy.sum(numpy.abs(numpy.diff(numpy.sign(values.flatten() - interior_range[1])/2)))
                            count_outside = values.flatten() > interior_range[1]
                            if count_diff > 1:
                                nonconvex_outside = True
                                #print('Vessel outside interior range 2 (interior proximal)')
                                #plotter = pv.Plotter()
                                #plotter.add_mesh(tree.domain.boundary,opacity=0.2)
                                #plotter.add_points(proximal_line[count_outside], color='red', point_size=10,
                                #                   render_points_as_spheres=True)
                                #plotter.add_points(proximal_line[~count_outside], color='green', point_size=10,
                                #                   render_points_as_spheres=True)
                                #plotter.show()
                                continue
                        if interior_bifurcation and interior_distal:
                            distal_line = (data[closest_vessels[j, i], 3:6] * line +
                                           bifurcation_point * (1 - line))
                            values = tree.domain(distal_line)
                            count_diff = numpy.sum(numpy.abs(numpy.diff(numpy.sign(values.flatten() - interior_range[1])/2)))
                            count_outside = values.flatten() > interior_range[1]
                            #if numpy.any(values > interior_range[1]):
                            if count_diff > 1:
                                nonconvex_outside = True
                                #print('Vessel outside interior range (interior distal)')
                                #plotter = pv.Plotter()
                                #plotter.add_mesh(tree.domain.boundary,opacity=0.2)
                                #plotter.add_points(distal_line[count_outside], color='red', point_size=10,
                                #                   render_points_as_spheres=True)
                                #plotter.add_points(distal_line[~count_outside], color='green', point_size=10,
                                #                   render_points_as_spheres=True)
                                #plotter.show()
                                continue
                        else:
                            distal_line = (data[closest_vessels[j, i], 3:6] * line +
                                           bifurcation_point * (1 - line))
                            values = tree.domain(distal_line)
                            count_diff = numpy.sum(numpy.abs(numpy.diff(numpy.sign(values.flatten() - interior_range[1])/2)))
                            count_outside = values.flatten() > interior_range[1]
                            if count_diff > 1:
                                nonconvex_outside = True
                                #print('Vessel outside interior range 2 (interior distal)')
                                #plotter = pv.Plotter()
                                #plotter.add_mesh(tree.domain.boundary,opacity=0.2)
                                #plotter.add_points(distal_line[count_outside], color='red', point_size=10,
                                #                   render_points_as_spheres=True)
                                #plotter.add_points(distal_line[~count_outside], color='green', point_size=10,
                                #                   render_points_as_spheres=True)
                                #plotter.show()
                                continue
                        #else:
                        #    continue
                        terminal_vessel = TreeData()
                        terminal_daughter_vessel = TreeData()
                        parent_vessel = TreeData()
                        #connectivity = numpy.nan_to_num(tree.data[:, 15:18], nan=-1.0).astype(int)
                        connectivity = tree.connectivity
                        create_new_vessels(bifurcation_point, data, terminal_point, terminal_vessel,
                                           terminal_daughter_vessel, parent_vessel, data[:, 19].max(),
                                           numpy.float64(data.shape[0]),
                                           connectivity, bifurcation_vessel, tree.parameters.murray_exponent,
                                           tree.parameters.kinematic_viscosity*tree.parameters.fluid_density, tree.parameters.terminal_flow,
                                           tree.parameters.terminal_pressure, tree.parameters.root_pressure,
                                           tree.parameters.radius_exponent, tree.parameters.length_exponent)
                        terminal_vessel[0, 21] += tree.physical_clearance
                        end_2 = perf_counter()
                        tree.times['chunk_2'][-1] += end_2 - start_2
                        start = perf_counter()
                        #terminal_vessel_proximity = sphere_proximity(tree.data, terminal_vessel[0, :])
                        search_radius = numpy.max(data[:, 20]) / 2 + numpy.max(data[:, 21]) + terminal_vessel[
                            0, 20] / 2 + terminal_vessel[0, 21]
                        #terminal_vessel_proximity = search_tree.query_ball_point((terminal_vessel[0, 0:3] +
                        #                                                          terminal_vessel[0, 3:6]) / 2,
                        #                                                         search_radius)
                        #terminal_vessel_proximity = tree.kdtm.query_ball_point((terminal_vessel[0, 0:3] +
                        #                                                          terminal_vessel[0, 3:6]) / 2,
                        #                                                          search_radius)
                        terminal_vessel_proximity = tree.hnsw_tree.query_ball_point(((terminal_vessel[0, 0:3] +
                                                                                  terminal_vessel[0, 3:6]) / 2).reshape(1,3),
                                                                                  search_radius)
                        terminal_vessel_proximity_check = numpy.full((data.shape[0],), False, dtype=bool)
                        terminal_vessel_proximity_check[terminal_vessel_proximity] = True
                        terminal_vessel_proximity = terminal_vessel_proximity_check
                        terminal_vessel_proximity[bifurcation_vessel] = False
                        if any(terminal_vessel_proximity):
                            if obb_any(data[terminal_vessel_proximity, :], terminal_vessel):
                                #print('Terminal Vessel in collision')
                                continue
                        terminal_vessel[0, 21] -= tree.physical_clearance
                        terminal_daughter_vessel[0, 21] += tree.physical_clearance
                        #terminal_daughter_vessel_proximity = sphere_proximity(tree.data, terminal_daughter_vessel[0, :])
                        #terminal_daughter_vessel_proximity = search_tree.query_ball_point(
                        #    (terminal_daughter_vessel[0, 0:3] +
                        #     terminal_daughter_vessel[0, 3:6]) / 2,
                        #    search_radius)
                        #terminal_daughter_vessel_proximity = tree.kdtm.query_ball_point(
                        #    (terminal_daughter_vessel[0, 0:3] +
                        #     terminal_daughter_vessel[0, 3:6]) / 2,
                        #    search_radius)
                        terminal_daughter_vessel_proximity = tree.hnsw_tree.query_ball_point(
                            ((terminal_daughter_vessel[0, 0:3] +
                             terminal_daughter_vessel[0, 3:6]) / 2).reshape(1,3),
                            search_radius)
                        terminal_daughter_vessel_proximity_check = numpy.full((data.shape[0],), False, dtype=bool)
                        terminal_daughter_vessel_proximity_check[terminal_daughter_vessel_proximity] = True
                        terminal_daughter_vessel_proximity = terminal_daughter_vessel_proximity_check
                        terminal_daughter_vessel_proximity[bifurcation_vessel] = False
                        if not numpy.isnan(terminal_daughter_vessel[0, 15]):
                            terminal_daughter_vessel_proximity[int(terminal_daughter_vessel[0, 15])] = False
                        if not numpy.isnan(terminal_daughter_vessel[0, 16]):
                            terminal_daughter_vessel_proximity[int(terminal_daughter_vessel[0, 16])] = False
                        if any(terminal_daughter_vessel_proximity):
                            if obb_any(data[terminal_daughter_vessel_proximity, :], terminal_daughter_vessel):
                                #print('Terminal Daughter Vessel in collision')
                                continue
                        terminal_daughter_vessel[0, 21] -= tree.physical_clearance
                        parent_vessel[0, 21] += tree.physical_clearance
                        #parent_vessel_proximity = sphere_proximity(tree.data, parent_vessel[0, :])
                        #parent_vessel_proximity = search_tree.query_ball_point((parent_vessel[0, 0:3] + parent_vessel[0, 3:6])/2,
                        #                                                       search_radius)
                        #parent_vessel_proximity = tree.kdtm.query_ball_point((parent_vessel[0, 0:3] + parent_vessel[0, 3:6])/2,
                        #                                                       search_radius)
                        parent_vessel_proximity = tree.hnsw_tree.query_ball_point(((parent_vessel[0, 0:3] + parent_vessel[0, 3:6])/2).reshape(1,3),
                                                                               search_radius)
                        parent_vessel_proximity_check = numpy.full((data.shape[0],), False, dtype=bool)
                        parent_vessel_proximity_check[parent_vessel_proximity] = True
                        parent_vessel_proximity = parent_vessel_proximity_check
                        parent_vessel_proximity[bifurcation_vessel] = False
                        if not numpy.isnan(data[bifurcation_vessel, 17]):
                            super_parent = int(data[bifurcation_vessel, 17])
                            parent_vessel_proximity[int(data[bifurcation_vessel, 17])] = False
                            if int(data[super_parent, 15]) == bifurcation_vessel:
                                parent_vessel_proximity[int(data[super_parent, 16])] = False
                            else:
                                parent_vessel_proximity[int(data[super_parent, 15])] = False
                        if any(parent_vessel_proximity):
                            if obb_any(data[parent_vessel_proximity, :], parent_vessel):
                                #print('Parent Vessel in collision')
                                continue
                        parent_vessel[0, 21] -= tree.physical_clearance
                        end = perf_counter()
                        tree.times['collision'][-1] += end - start
                        start_3 = perf_counter()
                        terminal_map = TreeMap()
                        #upstream = numpy.array(sorted(set(tree.vessel_map[bifurcation_vessel]['upstream'])),dtype=int)
                        #downstream = numpy.array(sorted(set(tree.vessel_map[bifurcation_vessel]['downstream'])), dtype=int)
                        upstream = sorted(set(tree.vessel_map[bifurcation_vessel]['upstream']))
                        downstream = sorted(set(tree.vessel_map[bifurcation_vessel]['downstream']))
                        terminal_map[data.shape[0]] = {'upstream': [], 'downstream': []}
                        #terminal_map[tree.data.shape[0]]['upstream'] = numpy.append(upstream, numpy.array([bifurcation_vessel]))
                        terminal_map[data.shape[0]]['upstream'] = list(upstream)
                        terminal_map[data.shape[0]]['upstream'].append(bifurcation_vessel)
                        terminal_daughter_map = TreeMap()
                        terminal_daughter_map[data.shape[0] + 1] = {'upstream': [], 'downstream': []}
                        terminal_daughter_map[data.shape[0] + 1]['upstream'] = list(upstream)
                        terminal_daughter_map[data.shape[0] + 1]['downstream'] = list(downstream)
                        #terminal_daughter_map[data.shape[0] + 1]['upstream'] = numpy.append(upstream, numpy.array([bifurcation_vessel]))
                        terminal_daughter_map[data.shape[0] + 1]['upstream'].append(bifurcation_vessel)
                        parent_map = TreeMap()
                        parent_map[bifurcation_vessel] = {'upstream': [], 'downstream': []}
                        #parent_map[bifurcation_vessel]['downstream'] = numpy.append(downstream, numpy.array([tree.data.shape[0],tree.data.shape[0] + 1]))
                        parent_map[bifurcation_vessel]['upstream'] = deepcopy(upstream)
                        parent_map[bifurcation_vessel]['downstream'] = deepcopy(downstream)
                        parent_map[bifurcation_vessel]['downstream'].append(data.shape[0])
                        parent_map[bifurcation_vessel]['downstream'].append(data.shape[0] + 1)
                        """
                        upstream = sorted(set(tree.vessel_map[bifurcation_vessel]['upstream']))
                        downstream = sorted(set(tree.vessel_map[bifurcation_vessel]['downstream']))
                        terminal_map[tree.data.shape[0]] = {'upstream': [], 'downstream': []}
                        terminal_map[tree.data.shape[0]]['upstream'].extend(deepcopy(upstream))
                        terminal_map[tree.data.shape[0]]['upstream'].append(deepcopy(bifurcation_vessel))
                        #terminal_map[tree.data.shape[0]]['upstream'] = numpy.append(upstream,
                        #                                                            numpy.array([bifurcation_vessel]))
                        terminal_daughter_map = TreeMap()
                        terminal_daughter_map[tree.data.shape[0] + 1] = {'upstream': [], 'downstream': []}
                        terminal_daughter_map[tree.data.shape[0] + 1]['downstream'].extend(deepcopy(downstream))
                        terminal_daughter_map[tree.data.shape[0] + 1]['upstream'].extend(deepcopy(upstream))
                        terminal_daughter_map[tree.data.shape[0] + 1]['upstream'].append(bifurcation_vessel)
                        #terminal_daughter_map[tree.data.shape[0] + 1]['downstream'] = downstream
                        #terminal_daughter_map[tree.data.shape[0] + 1]['upstream'] = numpy.append(upstream,
                        #                                                                         numpy.array([bifurcation_vessel]))
                        parent_map = TreeMap()
                        parent_map[bifurcation_vessel] = {'upstream': [], 'downstream': []}
                        parent_map[bifurcation_vessel]['downstream'].extend(deepcopy(downstream))
                        parent_map[bifurcation_vessel]['upstream'].extend(deepcopy(upstream))
                        parent_map[bifurcation_vessel]['downstream'].append(tree.data.shape[0])
                        parent_map[bifurcation_vessel]['downstream'].append(tree.data.shape[0] + 1)
                        """
                        #parent_map[bifurcation_vessel]['downstream'] = numpy.append(downstream, numpy.array([tree.data.shape[0],tree.data.shape[0] + 1]))
                        #parent_map[bifurcation_vessel]['upstream'] = upstream
                        #new_vessel_map = deepcopy(tree.vessel_map)
                        #new_vessel_map = tree.vessel_map_copy
                        new_vessel_map = TreeMap()
                        new_vessel_map.update(parent_map)
                        new_vessel_map.update(terminal_map)
                        new_vessel_map.update(terminal_daughter_map)
                        added_vessels = [terminal_vessel, terminal_daughter_vessel, parent_vessel]
                        #new_vessels = deepcopy(tree.data)
                        #new_vessels = tree.data.copy(order='C')
                        #new_vessels = tree.data_copy
                        tmp_28 = data[:, 28].copy()
                        change_i = []
                        change_j = []
                        new_data = []
                        old_data = []
                        #connectivity = numpy.nan_to_num(tree.data[:, 15:18], nan=-1.0).astype(int)
                        connectivity = tree.connectivity.copy()
                        results = update_vessels(bifurcation_point, data, terminal_point,
                                                 connectivity, bifurcation_vessel, tree.parameters.murray_exponent,
                                                 tree.parameters.kinematic_viscosity * tree.parameters.fluid_density,
                                                 tree.parameters.terminal_flow,
                                                 tree.parameters.terminal_pressure, tree.parameters.root_pressure,
                                                 tree.parameters.radius_exponent, tree.parameters.length_exponent)
                        reduced_resistance = numpy.array(results[0])
                        reduced_length = numpy.array(results[1])
                        main_idx = results[2]
                        main_scale = numpy.array(results[3])
                        alt_idx = results[4]
                        alt_scale = numpy.array(results[5])
                        bifurcation_ratios = numpy.array(results[6])
                        flows = numpy.array(results[7])
                        root_radius = results[8]
                        # --- BEGIN: preallocated change buffers (NumPy) ---
                        main_idx = np.asarray(main_idx, dtype=np.intp)
                        alt_idx = np.asarray(alt_idx, dtype=np.intp)
                        alt_scale = np.asarray(alt_scale, dtype=data.dtype, order="C")
                        flows = np.asarray(flows, dtype=data.dtype, order="C")
                        reduced_resistance = np.asarray(reduced_resistance, dtype=data.dtype, order="C")
                        reduced_length = np.asarray(reduced_length, dtype=data.dtype, order="C")
                        main_scale = np.asarray(main_scale, dtype=data.dtype, order="C")
                        bifurcation_ratios = np.asarray(bifurcation_ratios, dtype=data.dtype, order="C")
                        if bifurcation_ratios.ndim == 1:
                            #bifurcation_ratios = bifurcation_ratios.reshape(1, 2)
                            bifurcation_ratios = np.empty((1, 2), dtype=float)
                        N = data.shape[0]
                        W = data.shape[1]
                        downstream_bif = tree.vessel_map[bifurcation_vessel]["downstream"]
                        D0 = len(downstream_bif)

                        has_children = not np.any(np.isnan(terminal_daughter_vessel[0, 15:17]))
                        child_updates = 2 if has_children else 0  # matches current logic (either 2 or 0)

                        alt_valid_mask = alt_idx > -1
                        alt_valid = alt_idx[alt_valid_mask]
                        alt_scale_valid = alt_scale[alt_valid_mask]

                        # Count the exact number of updates that update_alt would emit (python fallback semantics)
                        alt_desc_total = 0
                        for idx in alt_valid:
                            alt_desc_total += len(tree.vessel_map[int(idx)]["downstream"])

                        n_update_alt = 1 + (6 * main_idx.size) + alt_desc_total + alt_valid.size
                        n_chunk_3_5 = (2 * D0) + child_updates
                        n_chunk_3_6 = N + W
                        n_total = n_update_alt + n_chunk_3_5 + n_chunk_3_6

                        change_i = np.empty(n_total, dtype=np.intp)
                        change_j = np.empty(n_total, dtype=np.intp)
                        new_data = np.empty(n_total, dtype=data.dtype)
                        old_data = np.empty(n_total, dtype=data.dtype)
                        p = 0

                        #change_i.append(0)
                        #change_j.append(21)
                        #new_data.append(root_radius)
                        #old_data.append(data[0, 21])

                        change_i[p] = 0
                        change_j[p] = 21
                        new_data[p] = root_radius
                        old_data[p] = data[0, 21]
                        p += 1

                        # Main idx updates: (22,25,27,28,23,24) for each main_idx
                        if main_idx.size:
                            cols_main = np.array([22, 25, 27, 28, 23, 24], dtype=np.intp)
                            ncols = cols_main.size
                            rows = np.repeat(main_idx, ncols)
                            cols = np.tile(cols_main, main_idx.size)

                            new_mat = np.column_stack(
                                (
                                    flows,
                                    reduced_resistance,
                                    reduced_length,
                                    main_scale,
                                    bifurcation_ratios[:, 0],
                                    bifurcation_ratios[:, 1],
                                )
                            ).astype(data.dtype, copy=False)
                            old_mat = data[main_idx][:, cols_main].astype(data.dtype, copy=False)

                            n = rows.size
                            change_i[p: p + n] = rows
                            change_j[p: p + n] = cols
                            new_data[p: p + n] = new_mat.ravel(order="C")
                            old_data[p: p + n] = old_mat.ravel(order="C")
                            p += n

                            tmp_28[main_idx] = main_scale


                        #tmp_28_copy = deepcopy(tmp_28)
                        #print("bifurcation: {}".format(bifurcation_ratios.shape))
                        #if len(bifurcation_ratios.shape) == 1:
                        #    bifurcation_ratios = np.empty((1,2),dtype=float)
                        start_chunk_3_4_alt = perf_counter()
                        #res_test = update_alt(reduced_resistance, reduced_length,
                        #                      main_idx, main_scale, alt_idx,
                        #                      alt_scale, bifurcation_ratios,
                        #                      flows, root_radius, data,
                        #                      tmp_28, tree.vessel_map)
                        #change_i = res_test[0]
                        #change_j = res_test[1]
                        #new_data = res_test[2]
                        #old_data = res_test[3]
                        # Alt subtree updates (col 28 for downstream + the alt root itself), and mutate tmp_28
                        for idx, scale in zip(alt_valid, alt_scale_valid):
                            idx_int = int(idx)
                            ds_list = tree.vessel_map[idx_int]["downstream"]
                            if ds_list:
                                ds = np.asarray(ds_list, dtype=np.intp)
                                m = ds.size
                                change_i[p: p + m] = ds
                                change_j[p: p + m] = 28

                                old_vals = data[ds, 28]
                                old_data[p: p + m] = old_vals
                                factor = scale / data[idx_int, 28]
                                new_data[p: p + m] = old_vals * factor
                                p += m

                                tmp_28[ds] *= factor

                            change_i[p] = idx_int
                            change_j[p] = 28
                            old_data[p] = data[idx_int, 28]
                            new_data[p] = scale
                            p += 1

                            tmp_28[idx_int] = scale

                        #new_vessels[0, 21] = root_radius
                        #change_i.append(0)
                        #change_j.append(21)
                        #new_data.append(root_radius)
                        #old_data.append(data[0, 21])
                        #tmp_28_copy = deepcopy(tmp_28)
                        #if len(bifurcation_ratios.shape) == 1:
                        #    bifurcation_ratios = np.empty((1, 2), dtype=float)
                        #res_test = update_alt(reduced_resistance, reduced_length,
                        #                      main_idx, main_scale, alt_idx,
                        #                      alt_scale, bifurcation_ratios,
                        #                      flows, root_radius, data,
                        #                      tmp_28, tree.vessel_map)
                        #change_i = res_test[0]
                        #change_j = res_test[1]
                        #new_data = res_test[2]
                        #old_data = res_test[3]
                        """
                        change_i.append(0)
                        change_j.append(21)
                        new_data.append(root_radius)
                        old_data.append(tree.data[0, 21])
                        if len(main_idx) > 0:
                            # Flows
                            change_i.extend(main_idx)
                            change_j.extend([22]*len(main_idx))
                            new_data.extend(flows.tolist())
                            old_data.extend(tree.data[main_idx, 22].tolist())
                            #new_vessels[main_idx, 22] = flows
                            # Reduced Resistance
                            change_i.extend(main_idx)
                            change_j.extend([25]*len(main_idx))
                            new_data.extend(reduced_resistance.tolist())
                            old_data.extend(tree.data[main_idx, 25].tolist())
                            #new_vessels[main_idx, 25] = reduced_resistance
                            # Reduced lengths
                            #new_vessels[main_idx, 27] = reduced_length
                            change_i.extend(main_idx)
                            change_j.extend([27]*len(main_idx))
                            new_data.extend(reduced_length.tolist())
                            old_data.extend(tree.data[main_idx, 27].tolist())
                            # Radius scaling
                            change_i.extend(main_idx)
                            change_j.extend([28]*len(main_idx))
                            new_data.extend(main_scale.tolist())
                            old_data.extend(tree.data[main_idx, 28].tolist())
                            #new_vessels[main_idx, 28] = main_scale
                            tmp_28[main_idx] = main_scale
                            # Bifurcations
                            change_i.extend(main_idx)
                            change_j.extend([23]*len(main_idx))
                            new_data.extend(bifurcation_ratios[:,0].tolist())
                            old_data.extend(tree.data[main_idx, 23].tolist())
                            #new_vessels[main_idx, 23] = bifurcation_ratios[:, 0]
                            change_i.extend(main_idx)
                            change_j.extend([24]*len(main_idx))
                            new_data.extend(bifurcation_ratios[:,1].tolist())
                            old_data.extend(tree.data[main_idx, 24].tolist())
                            #new_vessels[main_idx, 24] = bifurcation_ratios[:, 1]
                        for k in range(len(alt_idx)):
                            if alt_idx[k] > -1:
                                downstream = tree.vessel_map[alt_idx[k]]['downstream']
                                if len(tree.vessel_map[alt_idx[k]]['downstream']) > 0:
                                    # new_vessels[downstream, 28] /= new_vessels[alt_idx[k], 28]
                                    # new_vessels[alt_idx[k], 28] = alt_scale[k]
                                    # new_vessels[downstream, 28] *= new_vessels[alt_idx[k], 28]
                                    # new_vessels[downstream, 28] *= (alt_scale[k]/new_vessels[alt_idx[k], 28])
                                    tmp_28[downstream] *= (alt_scale[k] / tree.data[alt_idx[k], 28])
                                    change_i.extend(downstream)
                                    change_j.extend([28] * len(downstream))
                                    new_data.extend((tree.data[downstream, 28] * (
                                                alt_scale[k] / tree.data[alt_idx[k], 28])).tolist())
                                    old_data.extend(tree.data[downstream, 28].tolist())
                                    # new_vessels[alt_idx[k], 28] = alt_scale[k]
                                    tmp_28[alt_idx[k]] = alt_scale[k]
                                    change_i.append(alt_idx[k])
                                    change_j.append(28)
                                    new_data.append(alt_scale[k])
                                    old_data.append(tree.data[alt_idx[k], 28])
                                else:
                                    # new_vessels[alt_idx[k], 28] = alt_scale[k]
                                    tmp_28[alt_idx[k]] = alt_scale[k]
                                    change_i.append(alt_idx[k])
                                    change_j.append(28)
                                    new_data.append(alt_scale[k])
                                    old_data.append(tree.data[alt_idx[k], 28])
                        """
                        # --- chunk_3_5: downstream_bif adjustments (cols 28 and 26) ---
                        if D0:
                            ds = np.asarray(downstream_bif, dtype=np.intp)
                            m = ds.size

                            # col 28 updates
                            change_i[p: p + m] = ds
                            change_j[p: p + m] = 28
                            old_vals_28 = data[ds, 28]
                            old_data[p: p + m] = old_vals_28
                            new_data[p: p + m] = (old_vals_28 / data[bifurcation_vessel, 28]) * \
                                                 terminal_daughter_vessel[0,
                                                 28]
                            p += m

                            tmp_28[ds] /= data[bifurcation_vessel, 28]
                            tmp_28[ds] *= terminal_daughter_vessel[0, 28]

                            # col 26 updates
                            change_i[p: p + m] = ds
                            change_j[p: p + m] = 26
                            old_vals_26 = data[ds, 26]
                            old_data[p: p + m] = old_vals_26
                            new_data[p: p + m] = old_vals_26 + 1.0
                            p += m

                        # parent (col 17) rewiring when bifurcation_vessel had two children
                        if has_children:
                            left = int(terminal_daughter_vessel[0, 15])
                            right = int(terminal_daughter_vessel[0, 16])
                            new_parent = data.shape[0] + 1

                            change_i[p: p + 2] = (left, right)
                            change_j[p: p + 2] = 17
                            new_data[p: p + 2] = new_parent
                            # NOTE: keep existing behavior/bug: old_data uses [15] for both entries
                            old_parent = data[left, 17]
                            old_data[p: p + 2] = (old_parent, old_parent)
                            p += 2
                        """
                        if len(tree.vessel_map[bifurcation_vessel]['downstream']) > 0:
                            # new changes to add !!!!!!!!!!!!!!!!!!!!!!!
                            change_i.extend(tree.vessel_map[bifurcation_vessel]['downstream'])
                            change_j.extend([28] * len(tree.vessel_map[bifurcation_vessel]['downstream']))
                            tmp_new_data = (data[tree.vessel_map[bifurcation_vessel]['downstream'], 28] /
                                            data[bifurcation_vessel, 28]) * terminal_daughter_vessel[0, 28]
                            new_data.extend(tmp_new_data.tolist())
                            old_data.extend(data[tree.vessel_map[bifurcation_vessel]['downstream'], 28].tolist())
                            # new_vessels[tree.vessel_map[bifurcation_vessel]['downstream'], 28] /= new_vessels[
                            #    bifurcation_vessel, 28]
                            tmp_28[tree.vessel_map[bifurcation_vessel]['downstream']] /= data[
                                bifurcation_vessel, 28]
                            # new_vessels[tree.vessel_map[bifurcation_vessel]['downstream'], 28] *= \
                            # terminal_daughter_vessel[0, 28]
                            tmp_28[tree.vessel_map[bifurcation_vessel]['downstream']] *= terminal_daughter_vessel[0, 28]
                            change_i.extend(tree.vessel_map[bifurcation_vessel]['downstream'])
                            change_j.extend([26] * len(tree.vessel_map[bifurcation_vessel]['downstream']))
                            new_data.extend(
                                (data[tree.vessel_map[bifurcation_vessel]['downstream'], 26] + 1.0).tolist())
                            old_data.extend(data[tree.vessel_map[bifurcation_vessel]['downstream'], 26].tolist())
                            # new_vessels[tree.vessel_map[bifurcation_vessel]['downstream'], 26] += 1.0
                        # print('Bifurcation Vessel Upstream: ', new_vessel_map[bifurcation_vessel]['upstream'])
                        for k in tree.vessel_map[bifurcation_vessel]['upstream']:
                            #assert k != bifurcation_vessel, "reflexive insertion of bifurcation vessel"
                            #new_vessel_map[k]['downstream'].append(tree.data.shape[0])
                            #new_vessel_map[k]['downstream'].append(tree.data.shape[0] + 1)
                            new_vessel_map[k] = {'upstream': [], 'downstream': []}
                            new_vessel_map[k]['downstream'].extend([data.shape[0], data.shape[0]+1])
                            #new_vessel_map[k]['downstream'] = numpy.concatenate((new_vessel_map[k]['downstream'],
                            #                                                     numpy.array([tree.data.shape[0],
                            #                                                           tree.data.shape[0] + 1])))
                        if not numpy.any(numpy.isnan(terminal_daughter_vessel[0, 15:17])):
                            if not numpy.isnan(terminal_daughter_vessel[0, 15]):
                                change_i.append(int(terminal_daughter_vessel[0, 15]))
                                change_j.append(17)
                                new_data.append(data.shape[0] + 1)
                                old_data.append(data[int(terminal_daughter_vessel[0, 15]), 17])
                                #new_vessels[int(terminal_daughter_vessel[0, 15]), 17] = tree.data.shape[0] + 1
                            if not numpy.isnan(terminal_daughter_vessel[0, 16]):
                                change_i.append(int(terminal_daughter_vessel[0, 16]))
                                change_j.append(17)
                                new_data.append(data.shape[0] + 1)
                                old_data.append(data[int(terminal_daughter_vessel[0, 15]), 17])
                                #new_vessels[int(terminal_daughter_vessel[0, 16]), 17] = tree.data.shape[0] + 1
                        #for k in terminal_daughter_map[int(tree.data.shape[0] + 1)]['downstream']:
                        for k in tree.vessel_map[bifurcation_vessel]['downstream']:
                            # new_vessel_map[k]['upstream'].append(int(tree.data.shape[0] + 1))
                            new_vessel_map[k] = {'upstream': [], 'downstream': []}
                            new_vessel_map[k]['upstream'].append(int(data.shape[0] + 1))
                            # new_vessel_map[k]['upstream'] = numpy.concatenate((new_vessel_map[k]['upstream'],
                            #                                                   numpy.array([int(tree.data.shape[0] + 1)])))
                        #new_vessels[:, 21] = new_vessels[0, 21] * new_vessels[:, 28]
                        """
                        # --- chunk_3_6: radii for all existing vessels + parent row overwrite ---
                        tmp_radii = np.empty(N, dtype=data.dtype)
                        if tree.n_terminals < 10000:
                            np.multiply(tmp_28, root_radius, out=tmp_radii)
                        else:
                            ne_multiply(tmp_28, root_radius, tmp_radii)

                        rows = np.arange(N, dtype=np.intp)
                        change_i[p: p + N] = rows
                        change_j[p: p + N] = 21
                        new_data[p: p + N] = tmp_radii
                        old_data[p: p + N] = data[:, 21]
                        p += N

                        cols = np.arange(W, dtype=np.intp)
                        change_i[p: p + W] = bifurcation_vessel
                        change_j[p: p + W] = cols
                        new_data[p: p + W] = parent_vessel[0, :]
                        old_data[p: p + W] = data[bifurcation_vessel, :]
                        p += W
                        """
                        tmp_radii = np.zeros((data.shape[0], 1))
                        if tree.n_terminals < 10000:
                            #np.multiply(new_vessels[:, 28], new_vessels[0, 21], out=new_vessels[:, 21])
                            np.multiply(tmp_28, root_radius, out=tmp_radii[:, 0])
                        else:
                            #ne_multiply(new_vessels[:, 28], new_vessels[0, 21], new_vessels[:, 21])
                            ne_multiply(tmp_28, root_radius, tmp_radii[:, 0])
                            #scale_column_with_multiply(new_vessels, 28, new_vessels[0, 21], 21)
                            #multiply_columns(new_vessels)
                        #new_vessels[:, 21] = multiply_elements(new_vessels[:, 28], new_vessels[0, 21])
                        #ne.set_num_threads(ne.ncores)
                        #new_vessels[:, 21] = ne.evaluate('v28 * scalar', local_dict={'v28': new_vessels[:, 27],
                        #                                                            'scalar': new_vessels[0, 21]})
                        #if not np.all(np.isclose(tmp_28,new_vessels[:, 28])):
                        #    print('col 28 mismatch')
                        #idxs = np.arange(data.shape[0]).astype(int)
                        #change_i.extend(idxs.tolist())
                        change_i.extend(range(data.shape[0]))
                        change_j.extend([21]*data.shape[0])
                        new_data.extend(tmp_radii.flatten().tolist())
                        old_data.extend(data[:, 21].tolist())
                        #new_vessels[bifurcation_vessel, :] = parent_vessel
                        change_i.extend([bifurcation_vessel]*data.shape[1])
                        change_j.extend(np.arange(data.shape[1]).astype(int).tolist())
                        new_data.extend(parent_vessel[0, :].tolist())
                        old_data.extend(data[bifurcation_vessel, :].tolist())
                        appended_vessels = numpy.vstack([terminal_vessel, terminal_daughter_vessel])
                        #new_vessels = numpy.vstack([new_vessels, appended_vessels])
                        #new_vessels[-2, :] = terminal_vessel
                        #new_vessels[-1, :] = terminal_daughter_vessel
                        """
                        end_3_6 = perf_counter()
                        #tree.times['chunk_3_6'][-1] += end_3_6 - start_3_6
                        start_3_7 = perf_counter()
                        #print("Bifurcation Vessel: ", bifurcation_vessel)
                        #print("Connectivity: ", tree.connectivity_copy)
                        #connectivity[bifurcation_vessel, :] = np.nan_to_num(tree.data[bifurcation_vessel, 15:18], nan=-1.0).astype(int)
                        connectivity[bifurcation_vessel, :] = np.nan_to_num(parent_vessel[0, 15:18],
                                                                            nan=-1.0).astype(int)
                        if not numpy.isnan(terminal_daughter_vessel[0, 15]):
                            connectivity[int(terminal_daughter_vessel[0, 15]), -1] = data.shape[0] + 1
                        if not numpy.isnan(terminal_daughter_vessel[0, 16]):
                            connectivity[int(terminal_daughter_vessel[0, 16]), -1] = data.shape[0] + 1
                        connectivity = numpy.vstack((connectivity,
                                                     np.nan_to_num(terminal_vessel[:,15:18], nan=-1.0).astype(int).reshape(1,3),
                                                     np.nan_to_num(terminal_daughter_vessel[:, 15:18], nan=-1.0).astype(int)))
                        #tree.connectivity_copy[-2, :] = np.nan_to_num(terminal_vessel[:, 15:18], nan=-1.0).astype(int).reshape(1,3)
                        #tree.connectivity_copy[-1, :] = np.nan_to_num(terminal_daughter_vessel[:, 15:18], nan=-1.0).astype(int)
                        #tree.kdtm.start_update((new_vessels[:,0:3]+new_vessels[:,3:6])/2)
                        success = True
                        end_3_7 = perf_counter()
                        tree.times['chunk_3_7'][-1] += end_3_7 - start_3_7
                        end_3 = perf_counter()
                        tree.times['chunk_3'][-1] += end_3 - start_3
                        new_vessels = None
                        break
                        """
                        new_vessels[bifurcation_vessel, :] = parent_vessel
                        #appended_vessels = np.vstack([terminal_vessel, terminal_daughter_vessel])
                        #new_vessels = numpy.vstack([new_vessels, appended_vessels])
                        new_vessels[-2, :] = terminal_vessel
                        new_vessels[-1, :] = terminal_daughter_vessel
                        tree.connectivity_copy[bifurcation_vessel, :] = np.nan_to_num(new_vessels[bifurcation_vessel, 15:18], nan=-1.0).astype(int)
                        #tree.connectivity_copy[bifurcation_vessel, :] = np.nan_to_num(parent_vessel[0, 15:18], nan=-1.0).astype(int)
                        if not numpy.isnan(terminal_daughter_vessel[0, 15]):
                            tree.connectivity_copy[int(terminal_daughter_vessel[0, 15]), -1] = tree.data.shape[0] + 1
                        if not numpy.isnan(terminal_daughter_vessel[0, 16]):
                            tree.connectivity_copy[int(terminal_daughter_vessel[0, 16]), -1] = tree.data.shape[0] + 1
                        #tree.connectivity_copy = numpy.vstack((tree.connectivity_copy,
                        #                                       np.nan_to_num(terminal_vessel[:,15:18], nan=-1.0).astype(int).reshape(1,3),
                        #                                       np.nan_to_num(terminal_daughter_vessel[:, 15:18], nan=-1.0).astype(int)))
                        tree.connectivity_copy[-2, :] = np.nan_to_num(terminal_vessel[:,15:18], nan=-1.0).astype(int).reshape(1,3)
                        tree.connectivity_copy[-1, :] = np.nan_to_num(terminal_daughter_vessel[:, 15:18], nan=-1.0).astype(int)
                        #tree.kdtm.start_update((new_vessels[:, 0:3] + new_vessels[:, 3:6]) / 2)
                        end_3 = perf_counter()
                        tree.times['chunk_3'][-1] += end_3 - start_3
                        success = True
                        #print('Success')
                        break
                        """
                    if success:
                        break
                if not success:
                    threshold *= threshold_adjuster
    return new_vessels, added_vessels, new_vessel_map, history, lines, nonconvex_outside, (bifurcation_vessel, data.shape[0],data.shape[0] + 1), bifurcation_cell, connectivity, change_i, change_j, new_data, old_data
"""
@numba.jit(nopython=True, parallel=True, cache=True, fastmath=True, boundscheck=False)
def multiply_columns(new_vessels):
    scalar = new_vessels[0, 21]
    nrows = new_vessels.shape[0]
    for i in numba.prange(nrows):
        new_vessels[i, 21] = new_vessels[i, 28] * scalar

@numba.njit(parallel=True, fastmath=True)
def scale_column_with_multiply(array, col_index, scalar, out_index):
    column = array[:, col_index]
    out_column = array[:, out_index]
    np.multiply(column, scalar, out=out_column)
"""

def ne_multiply(column, scalar, out):
    ne.evaluate('column * scalar', out=out)

def ne_scale(radius, length, radius_exp, length_exp):
    scale = np.pi * ne.evaluate('sum(radius ** radius_exp * length ** length_exp)')
    return scale

def get_points(tree, n_points, **kwargs):
    """
    This function returns a set of points that are at least a distance
    of 'threshold' away from the tree. The points are generated in a
    specified region of the domain field (interior, exterior, or
    boundary).

    Parameters
    ----------
    tree : Tree
        The tree object that is used to generate the points.
    n_points : int
        The number of points to generate.
    kwargs : dict
        The keyword arguments that are used to specify the region
        of the domain field to generate the points.

    Returns
    -------
    points : numpy.ndarray
        The points that are generated.
    point_tree : scipy.spatial.cKDTree
        A cKDTree object that can be used for fast lookup
        of the returned points for further processing.
    """
    data = tree.data[:tree.segment_count, :]
    iteration = 0
    if len(tree.times['get_points_0']) == int(data.shape[0]-1) // 2 + 1:
        pass
    else:
        tree.times['get_points_0'].append(0.0)
        tree.times['get_points_1'].append(0.0)
        tree.times['get_points_2'].append(0.0)
        tree.times['get_points_3'].append(0.0)
    where = kwargs.get('where', 'interior')
    max_iterations = kwargs.get('max_iterations', 10)
    threshold = kwargs.get('threshold', tree.physical_clearance)
    volume_threshold = kwargs.get('volume_threshold', None)
    interior_range = kwargs.get('interior_range', [-1.0, 0.0])
    exterior_range = kwargs.get('exterior_range', [0.0, 1.0])
    search_tree = kwargs.get('search_tree', None)
    n_vessels = kwargs.get('n_vessels', min(data.shape[0], 10))
    n_heuristic = kwargs.get('n_heuristic', 500)
    use_random_int = kwargs.get('use_random_int', False)
    threshold_cuttoff = kwargs.get('n_random_int', 10000)
    if tree.n_terminals >= threshold_cuttoff:
        threshold = 0.0
    points = numpy.ones((n_points, 3), dtype=numpy.float64)*numpy.nan
    point_distances = numpy.ones((n_vessels, n_points), dtype=numpy.float64)*numpy.nan
    closest_vessel_idx = numpy.zeros((n_vessels, n_points), dtype=numpy.int64)
    mesh_cells = numpy.ones((n_points,), dtype=numpy.int64)*-1
    remaining_points = n_points
    #midpoints = (tree.data[:, 0:3] + tree.data[:, 3:6]) / 2
    #midpoints = tree.midpoints_copy
    midpoints = tree.midpoints
    if len(midpoints.shape) == 1:
        midpoints = midpoints.reshape(1, -1)
        #print("reshaping midpoints")
    #assert id(tree.hnsw_tree) == tree.hnsw_tree_id, "NOT THE SAME HNSW TREE"
    if search_tree is None:
        #search_ = cKDTree((tree.data[:, 0:3] + tree.data[:, 3:6]) / 2)
        pass
    else:
        #search_ = search_tree
        pass
    if tree.n_terminals < n_heuristic:
        point_distances = numpy.ones((data.shape[0], n_points), dtype=numpy.float64) * numpy.nan
        closest_vessel_idx = numpy.zeros((data.shape[0], n_points), dtype=numpy.int64)
        mesh_cells = numpy.ones((n_points,), dtype=numpy.int64) * -1
    while remaining_points > 0 and iteration < max_iterations:
        if where == 'interior':
            start = perf_counter()
            #print(f"Tree Convex: {tree.convex}")
            if not tree.convex and tree.n_terminals <= n_heuristic:
                #print("correct interior")
                tmp_points, cells = tree.domain.get_interior_points((2 * remaining_points), tree=midpoints, threshold=threshold,
                                                             volume_threshold=volume_threshold,
                                                             implicit_range=interior_range, use_random_int=use_random_int,
                                                             convex=tree.convex)
            else:
                #tmp_points, cells = tree.domain.get_interior_points((2 * remaining_points), tree=midpoints, threshold=threshold,
                #                                             volume_threshold=volume_threshold,
                #                                             implicit_range=interior_range, convex=tree.convex)
                #print("other interior")
                tmp_points, cells = tree.domain.get_interior_points((2 * remaining_points))
            end = perf_counter()
            #tree.times['get_points_0'][-1] += end - start
        elif where == 'exterior':
            tmp_points = tree.domain.get_exterior_points(n_points, exterior_range)
        elif where == 'boundary':
            tmp_points = tree.domain.get_boundary_points(n_points)
        else:
            raise ValueError("Invalid value for 'where'.")
        #print(f"Number of potential points: {len(tmp_points)}")
        #print(f"Number of NaN points: {numpy.sum(numpy.any(numpy.isnan(tmp_points),axis=1))}")
        #print(f"points: {tmp_points}")
        if tree.n_terminals >= n_heuristic:
            #distances, idx = search_.query(tmp_points, k=n_vessels)
            #distances, idx = tree.kdtm.query(tmp_points, k=n_vessels)
            start = perf_counter()
            distances, idx = tree.hnsw_tree.query(tmp_points, k=n_vessels)
            end = perf_counter()
            #tree.times['get_points_1'][-1] += end - start
            #idx = tree.rtree.query(tmp_points, k=n_vessels)
            start = perf_counter()
            if tree.n_terminals < threshold_cuttoff:
                AB = data[idx, 3:6] - data[idx, 0:3]
                AP = tmp_points[:, np.newaxis, :] - data[idx, 0:3]
                AB_dot_AB = np.sum(AB ** 2, axis=2)
                AP_dot_AB = np.sum(AP * AB, axis=2)
                with np.errstate(divide='ignore', invalid='ignore'):
                    tt = np.clip(np.true_divide(AP_dot_AB, AB_dot_AB), 0, 1)
                closest_points = data[idx, 0:3] + tt[..., np.newaxis] * AB
                distances = np.linalg.norm(tmp_points[:, np.newaxis, :] - closest_points, axis=2) - data[idx, 21]
                distances = distances.T
                min_dists = numpy.min(distances, axis=0)
                tmp_points = tmp_points[min_dists > threshold, :]
                resort = numpy.argsort(distances, axis=0)
                idx = idx.T
                idx = np.take_along_axis(idx, resort, axis=0)
            else:
                distances = distances.T
                min_dists = numpy.min(distances, axis=0)
                #mask = min_dists > threshold
                #tmp_points = tmp_points[mask, :]
                idx = idx.T
        else:
            start = perf_counter()
            #print(f"data.shape(): {data.shape}")
            AB = data[:, 3:6] - data[:, 0:3]
            AP = tmp_points[:, np.newaxis, :] - data[:, 0:3]
            AB_dot_AB = np.sum(AB ** 2, axis=1)
            AP_dot_AB = np.sum(AP * AB, axis=2)
            with np.errstate(divide='ignore', invalid='ignore'):
                tt = np.clip(np.true_divide(AP_dot_AB, AB_dot_AB), 0, 1)
            closest_points = data[:, 0:3] + tt[..., np.newaxis] * AB
            end = perf_counter()
            #tree.times['get_points_1'][-1] += end - start
            start = perf_counter()
            distances = np.linalg.norm(tmp_points[:, np.newaxis, :] - closest_points, axis=2)
            distances = distances.T
            min_dists = numpy.min(distances, axis=0)
            #plotter = pv.Plotter()
            #plotter.add_mesh(tree.domain.mesh, color='grey', opacity=0.2)
            #print(f"min_dists: {min_dists}")
            #plotter.add_mesh(tree.domain.mesh.extract_cells(cells), color='purple', opacity=0.2)
            #if len(tmp_points[min_dists < threshold, :]) > 0:
            #    plotter.add_points(tmp_points[min_dists < threshold, :], point_size=4, color='blue')
            tmp_points = tmp_points[min_dists > threshold, :]
            #if len(tmp_points) > 0:
            #    plotter.add_points(tmp_points, point_size=4, color='green')
            #plotter.add_points(midpoints, point_size=4, color='red')
            #print('threshold: {}'.format(threshold))
            #plotter.show()
            if tmp_points.shape[0] == 0:
                #print('get_points less than threshold')
                #continue
                pass
            idx = numpy.argsort(distances, axis=0)
        end = perf_counter()
        #tree.times['get_points_2'][-1] += end - start
        start = perf_counter()
        add_points = min(remaining_points, tmp_points.shape[0])
        points[n_points - remaining_points:n_points - remaining_points + add_points, :] = tmp_points[:add_points, :]
        if tree.n_terminals < threshold_cuttoff:
            point_distances[:, n_points - remaining_points:n_points - remaining_points + add_points] = distances[:, min_dists > threshold][:, :add_points]
            closest_vessel_idx[:, n_points - remaining_points:n_points - remaining_points + add_points] = idx[:, min_dists > threshold][:, :add_points]
            mesh_cells[n_points - remaining_points:n_points - remaining_points + add_points] = cells[min_dists > threshold][:add_points]
        else:
            point_distances[:, n_points - remaining_points:n_points - remaining_points + add_points] = distances[:, :add_points]
            closest_vessel_idx[:, n_points - remaining_points:n_points - remaining_points + add_points] = idx[:, :add_points]
            mesh_cells[n_points - remaining_points:n_points - remaining_points + add_points] = cells[:add_points]
        remaining_points -= add_points
        #print('remaining points: {}'.format(remaining_points))
        iteration += 1
        end = perf_counter()
        #tree.times['get_points_3'][-1] += end - start
    #point_distances = close_exact_points(tree.data, points)
    return points, point_distances, closest_vessel_idx, mesh_cells


def check_tree(tree, results):
    _new_data, _added_vessels, _new_vessel_map, _history, _lines = results
    _correct_data = deepcopy(_new_data)
    naive(tree, _correct_data, 0)
    naive_radius_scaling(_correct_data)
    naive_radius(_correct_data)
    if not numpy.all(numpy.isclose(_new_data, _correct_data, equal_nan=True)):
        #print("Tree has mismatched values compared to naive implementation.")
        return False, _new_data, _correct_data
    else:
        return True, _new_data, _correct_data

def naive(tree, data, start):
    """
    This function is used to check the values assigned in the current tree.
    """
    nu = tree.parameters.kinematic_viscosity * tree.parameters.fluid_density
    gamma = tree.parameters.murray_exponent
    if numpy.isnan(data[start, 15]) and numpy.isnan(data[start, 16]):
        data[start, 20] = np.linalg.norm(data[start, 3:6] - data[start, 0:3])
        data[start, 25] = (8*nu/np.pi)*data[start, 20]
        data[start, 23] = numpy.nan
        data[start, 24] = numpy.nan
        data[start, 27] = 0.0
        return (8*nu/np.pi)*data[start, 20]
    else:
        left = int(data[start, 15].item())
        right = int(data[start, 16].item())
        left_rr = naive(tree, data, left)
        right_rr = naive(tree, data, right)
        LR = ((data[left, 22] *
               left_rr) /
              (data[right, 22] *
               right_rr)) ** (1/4)
        lbif = (1 + LR ** (-gamma)) ** (-1/gamma)
        rbif = (1 + LR ** (gamma)) ** (-1/gamma)
        data[start, 20] = np.linalg.norm(data[start, 3:6] - data[start, 0:3])
        data[start, 25] = (8*nu/numpy.pi)*data[start, 20] + ((lbif ** 4 / left_rr) + (rbif ** 4 / right_rr)) ** -1
        data[start, 23] = lbif
        data[start, 24] = rbif
        data[start, 27] = lbif ** 2 * (data[left, 20]+data[left, 27]) + \
                          rbif ** 2 * (data[right, 20]+data[right, 27])
        return (8*nu/numpy.pi) * data[start, 20] + ((lbif ** 4 / left_rr) + (rbif ** 4 / right_rr)) ** -1

def naive_radius(data):
    """
    This function is used to check the values assigned in the current tree.
    """
    next_vessels = [int(data[0,15]), int(data[0,16])]
    while len(next_vessels) > 0:
        idx = next_vessels.pop(0)
        parent = int(data[idx, 17])
        if int(data[parent, 15]) == int(idx):
            data[idx, 21] = data[parent, 21] * data[parent, 23]
        elif int(data[parent, 16]) == int(idx):
            data[idx, 21] = data[parent, 21] * data[parent, 24]
        if not numpy.isnan(data[idx, 15]):
            next_vessels.append(int(data[idx, 15]))
        if not numpy.isnan(data[idx, 16]):
            next_vessels.append(int(data[idx, 16]))

def naive_radius_scaling(data):
    """
    This function is used to check the values assigned to the radius
    scaling coefficient in the current tree.
    """
    next_vessels = [int(data[0,15]), int(data[0,16])]
    while len(next_vessels) > 0:
        idx = next_vessels.pop(0)
        parent = int(data[idx, 17])
        if int(data[parent, 15]) == int(idx):
            data[idx, 28] = data[parent, 28] * data[parent, 23]
        elif int(data[parent, 16]) == int(idx):
            data[idx, 28] = data[parent, 28] * data[parent, 24]
        if not numpy.isnan(data[idx, 15]):
            next_vessels.append(int(data[idx, 15]))
        if not numpy.isnan(data[idx, 16]):
            next_vessels.append(int(data[idx, 16]))


def map_triad(tree, point, vessel):
    data = tree.data[:tree.segment_count, :]
    proximal = data[vessel, 0:3]
    distal = data[vessel, 3:6]
    terminal = point
    #def triad(x, proximal=proximal, distal=distal, terminal=terminal):
    #    s = x[0]
    #    t = x[1]
    #    if s > 1.0:
    #        s = 1.0
    #    elif s < 0.0:
    #        s = 0.0
    #    if t > 1.0:
    #        t = 1.0
    #    elif t < 0.0:
    #        t = 0.0
    #    x = proximal * (1 - t) * s + distal * (t * s) + terminal * (1 - s)
    #    return x
    def triad(x, proximal=proximal, distal=distal, terminal=terminal):
        #if len(x.shape) == 2:
        #    s = x[:, 0]
        #    t = x[:, 1]
        #    mask = s + t > 1
        #    s[mask] = 1 - s[mask]
        #    t[mask] = 1 - t[mask]
        #    x = proximal * (1 - s - t)[:, np.newaxis] + distal * s[:, np.newaxis] + terminal * t[:, np.newaxis]
        #else:
        s = x[0]
        t = x[1]
        if s + t > 1:
            s = 1 - s
            t = 1 - t
        x = proximal * (1 - s - t) + distal * s + terminal * t
        return x
    return triad


def map_clamped(tree, point, vessel):
    data = tree.data[:tree.segment_count, :]
    proximal = data[vessel, 0:3]
    distal = data[vessel, 3:6]
    terminal = point
    def line(x, proximal=proximal, distal=distal):
        s = x[0]
        if s > 1.0:
            s = 1.0
        elif s < 0.0:
            s = 0.0
        x = proximal * (1 - s) + distal * s
        return x
    return line

import warnings
#warnings.filterwarnings('error', category=RuntimeWarning)

def construct_optimizer(tree, point, vessel, **kwargs):
    """
     Construct the optimizer for the current tree configuration.
    """
    data = tree.data[:tree.segment_count, :]
    proximal = data[vessel, 0:3]
    distal = data[vessel, 3:6]
    terminal = point
    d_min = kwargs.get('d_min', data[vessel, 21]*4)
    interior_range = kwargs.get('interior_range', [-1.0, 0.0])
    tree_scale = deepcopy(numpy.pi * numpy.sum(data[:, 21] ** tree.parameters.radius_exponent *
                                      data[:, 20] ** tree.parameters.length_exponent))
    #tree_scale = tree.volume_scale
    vol_0 = np.linalg.norm(data[vessel, 0:3] - point) * np.pi * data[vessel, 21] ** 2
    vol_1 = np.linalg.norm(data[vessel, 3:6] - point) * np.pi * data[vessel, 21] ** 2
    vol_2 = data[vessel, 20] * np.pi * data[vessel, 21] ** 2
    tree_adj_scale = vol_0 + vol_1 + vol_2
    tree_scale = tree.volume_scale - vol_2
    penalty = kwargs.get('penalty', tree_adj_scale)
    triad = map_triad(tree, point, vessel)
    nonconvex_sampling = kwargs.get('nonconvex_sampling', 10)
    lines = numpy.zeros((3, data.shape[1]), dtype=numpy.float64)
    lines[0, 0:3] = proximal
    lines[0, 3:6] = distal
    lines[1, 0:3] = proximal
    lines[1, 3:6] = terminal
    lines[2, 0:3] = distal
    lines[2, 3:6] = terminal
    lines[:, 12:15] = (lines[:, 3:6] - lines[:, 0:3])/numpy.linalg.norm(lines[:, 3:6] - lines[:, 0:3]).reshape(-1, 1)
    lines[0, 21] = data[vessel, 21]
    parent_vessel = data[vessel, 17]
    if tree.clamped_root and vessel == 0:
        get_line_pt = map_clamped(tree, point, vessel)
        def cost(x, func=tree_cost_2, d_min=d_min, terminal=terminal,
                 murray_exponent=tree.parameters.murray_exponent, kinematic_viscosity=(tree.parameters.kinematic_viscosity*tree.parameters.fluid_density),
                 terminal_flow=tree.parameters.terminal_flow, root_pressure=tree.parameters.root_pressure,
                 terminal_pressure=tree.parameters.terminal_pressure, radius_exponent=tree.parameters.radius_exponent,
                 length_exponent=tree.parameters.length_exponent, get_line_pt=get_line_pt, lines=lines, penalty=penalty,
                 scale=tree_scale, connectivity=tree.connectivity):
            x = get_line_pt(x)
            dists = numpy.array([numpy.linalg.norm(lines[0, 0:3] - x),
                                 numpy.linalg.norm(lines[0, 3:6] - x),
                                 numpy.linalg.norm(lines[1, 3:6] - x)])
            #triad_penalty = numpy.max([0.0, -1.0 * numpy.min(dists - d_min)])/d_min * penalty
            #connectivity = numpy.nan_to_num(tree.data[:, 15:18], nan=-1.0).astype(int)
            results = func(x, data, terminal, connectivity,
                           vessel, murray_exponent, kinematic_viscosity,
                           terminal_flow, terminal_pressure, root_pressure,
                           radius_exponent, length_exponent)
            try:
                #value = np.tanh((np.clip(numpy.nan_to_num(results, nan=scale),0,scale) + triad_penalty) / scale)
                value = (((
                    np.clip(numpy.nan_to_num(results - scale, nan=2 * scale + penalty), 0, 2 * scale + penalty))) / (
                                    scale + penalty)) # used to have triad_penalty
            except RuntimeWarning as e:
                #print("RuntimeWarning caught:", e)
                #print("scale =", scale)
                #print("numerator =", (numpy.nan_to_num(results, nan=scale) + triad_penalty) )
                value = np.tanh((np.clip(numpy.nan_to_num(results, nan=scale),0,scale) + triad_penalty) / scale)
            return value
        def vol(x, func=tree_cost_2, d_min=d_min, terminal=terminal,
                 murray_exponent=tree.parameters.murray_exponent, kinematic_viscosity=(tree.parameters.kinematic_viscosity*tree.parameters.fluid_density),
                 terminal_flow=tree.parameters.terminal_flow, root_pressure=tree.parameters.root_pressure,
                 terminal_pressure=tree.parameters.terminal_pressure, radius_exponent=tree.parameters.radius_exponent,
                 length_exponent=tree.parameters.length_exponent, get_line_pt=get_line_pt, lines=lines, penalty=penalty,
                 scale=tree_scale, connectivity=tree.connectivity):
            x = get_line_pt(x)
            results = func(x, data, terminal, connectivity,
                           vessel, murray_exponent, kinematic_viscosity,
                           terminal_flow, terminal_pressure, root_pressure,
                           radius_exponent, length_exponent)
            return results
        return cost, get_line_pt, vol
    def cost(x, func=tree_cost_2, d_min=d_min, terminal=terminal,
             murray_exponent=tree.parameters.murray_exponent, kinematic_viscosity=(tree.parameters.kinematic_viscosity*tree.parameters.fluid_density),
             terminal_flow=tree.parameters.terminal_flow, root_pressure=tree.parameters.root_pressure,
             terminal_pressure=tree.parameters.terminal_pressure, radius_exponent=tree.parameters.radius_exponent,
             length_exponent=tree.parameters.length_exponent, triad=triad, lines=lines, penalty=penalty,
             scale=tree_scale,connectivity=tree.connectivity, parent_vessel=parent_vessel):
        x = triad(x)
        #dists = close_exact_point(lines, x)
        dists = numpy.array([numpy.linalg.norm(lines[0, 0:3] - x),
                             numpy.linalg.norm(lines[0, 3:6] - x),
                             numpy.linalg.norm(lines[1, 3:6] - x)])
        #vec1 = distal - x
        #vec2 = terminal - x
        #vec3 = proximal - x
        #vec1 = vec1/numpy.linalg.norm(vec1)
        #vec2 = vec2/numpy.linalg.norm(vec2)
        #vec3 = vec3/numpy.linalg.norm(vec3)
        #angle1 = numpy.arccos(numpy.dot(vec1, vec3))*(180/numpy.pi)
        #angle2 = numpy.arccos(numpy.dot(vec2, vec3))*(180/numpy.pi)
        #angle3 = numpy.arccos(numpy.dot(vec3, vec1))*(180/numpy.pi)
        #ADD Parent angles

        #if angle1 > 90 or angle2 > 90:
        #    angle_penalty = penalty
        #else:
        #    angle_penalty = 0.0
        #if not isinstance(parent_vessel,type(numpy.nan)):
        #    parent_vessel = int(parent_vessel)
        #    vec4 = data[parent_vessel, 3:6] - data[parent_vessel, 0:3]
        #    vec4 = vec4/numpy.linalg.norm(vec4)
        #    angle3 = numpy.arccos(numpy.dot(vec3, vec4))*(180/numpy.pi)
        #    if angle3 > 90:
        #        angle_penalty += penalty
        #angle_penalty = 0.0
        #triad_penalty = numpy.max([0.0, -1.0 * numpy.min(dists - d_min)])/d_min * penalty
        triad_penalty = 0.0
        d_min_dist = np.min(dists)
        #line = numpy.linspace(0, 1, 10).reshape(-1, 1)
        #proximal_line = np.sum(np.clip(tree.domain(proximal*line + x*(1-line)).flatten(), 0, 1))/len(line)
        #distal_line = np.sum(np.clip(tree.domain(distal*line + x*(1-line)).flatten(), 0, 1))/len(line)
        #terminal_line = np.sum(np.clip(tree.domain(terminal*line + x*(1-line)).flatten(), 0, 1))/len(line)
        #val = (proximal_line + distal_line + terminal_line)/3
        #domain_penalty = ((np.log(val + 0.001) - np.log(0.001))/(np.log(1.0+0.001) - np.log(0.001)))*2*penalty*scale
        domain_penalty = 0.0
        if numpy.isclose(numpy.min(dists),0.0):
            triad_penalty = 2*(penalty+scale)
            #return triad_penalty
        elif numpy.min(dists) < d_min:
            s = (d_min - d_min_dist)/d_min
            triad_penalty = 2*(penalty+scale)*s**2
            #return triad_penalty
        else:
            pass
        #[TODO] angle penalty
        #[TODO] require that resulting parent vessel is at least a certain length? remove buffer region around triad
        # points
        #triad_penalty = (numpy.max([0.0, -1.0 * numpy.min(dists - d_min)])/d_min)/(numpy.min(dists)/d_min)
        #connectivity = numpy.nan_to_num(tree.data[:, 15:18], nan=-1.0).astype(int)
        results = func(x, data, terminal, connectivity,
                       vessel, murray_exponent, kinematic_viscosity,
                       terminal_flow, terminal_pressure, root_pressure,
                       radius_exponent, length_exponent)
        #if numpy.isnan(results):
        #    results = 10.0 * (scale + penalty)
        #if not numpy.isfinite(results):
        #    results = 10.0*(scale + penalty)
        #else:
        #    results = np.clip(results - scale, 0.0, 2.0 * (scale + penalty))
        #    results = results/penalty
        #results = numpy.log1p(results)
        #results = np.pi*results[-2]**2*results[-1]
        #try:
        #    value = np.tanh((numpy.nan_to_num(results, nan=scale) + triad_penalty) / scale)
        #except RuntimeWarning as e:
        #    print("RuntimeWarning caught:", e)
        #    print("scale =", scale)
        #    print("numerator =", (numpy.nan_to_num(results, nan=scale) + triad_penalty))
        #assert results > tree_scale, '{} results < {} tree_scale'.format(results, tree_scale)
        #return (((np.clip(numpy.nan_to_num(results - scale, nan=2*scale+penalty), 0, 2*scale+penalty) + triad_penalty))/(scale+penalty))# + 1.0
        #return -1/np.clip(numpy.nan_to_num(results - scale, nan=2*scale+penalty), 0, 2*scale+penalty)
        #return -1 / np.clip(numpy.nan_to_num(results + triad_penalty + angle_penalty, nan=2 * scale + penalty), 0, 2 * scale + penalty)
        return -1 / (np.clip(numpy.nan_to_num(results, nan=2 * scale + penalty), 0,
                            2 * scale + penalty) + domain_penalty)
        """
        with numpy.errstate(over='raise', divide='raise', invalid='raise'):
            try:
                val = results + triad_penalty#numpy.nan_to_num(results, nan=2 * scale + penalty)/(scale + penalty)
                #val = (val / (scale + penalty))
                val = (2*val)**4
                #print(val)
            except FloatingPointError as e:
                print("Overflow/divide in cost; results =", results)
                print("scale =", scale, "penalty =", penalty)
                print("Bifurcation: ", x)
                print("Proximal: ", proximal)
                print("Distal: ", distal)
                print("Terminal: ", terminal)
                val = numpy.nan_to_num(results, nan=2 * scale + penalty)/(scale + penalty)
        """
        return val
        #return results
        #return results
        #return value
    def vol(x, func=tree_cost_2, d_min=d_min, terminal=terminal,
             murray_exponent=tree.parameters.murray_exponent, kinematic_viscosity=(tree.parameters.kinematic_viscosity*tree.parameters.fluid_density),
             terminal_flow=tree.parameters.terminal_flow, root_pressure=tree.parameters.root_pressure,
             terminal_pressure=tree.parameters.terminal_pressure, radius_exponent=tree.parameters.radius_exponent,
             length_exponent=tree.parameters.length_exponent, triad=triad, lines=lines, penalty=penalty,
             scale=tree_scale,connectivity=tree.connectivity):
        x = triad(x)
        results = func(x, data, terminal, connectivity,
                       vessel, murray_exponent, kinematic_viscosity,
                       terminal_flow, terminal_pressure, root_pressure,
                       radius_exponent, length_exponent)
        return results
    # OPTIMIZER FOR CLAMPED ROOTS
    return cost, triad, vol

def build_native_cost(point, vessel, tree):
    data = tree.data[:tree.segment_count, :]
    new_data = np.vstack((data, TreeData(), TreeData()))
