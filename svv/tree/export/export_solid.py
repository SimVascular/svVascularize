import numpy
import pyvista
import pymeshfix
from tqdm import trange
from scipy.interpolate import splprep, splev
from svv.utils.remeshing.remesh import remesh_surface
from svv.domain.routines.boolean import boolean

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


def generate_tube(polyline, hsize=None):
    """
    This function generates a tube from a given polyline representing a single vessel of a
    vascular object (tree or forest).

    Parameters
    ----------
    polyline : pyvista.PolyData
        A polyline object representing the centerline of a vessel.
    hsize : float
        The mesh element size for the surface mesh of the vessel.

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
    tube = remesh_surface(tube, hsiz=hsize)
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
    model = boolean(tubes[0], tubes[1], operation='union')
    model = model.compute_normals(auto_orient_normals=True)
    hsize = (min(min(lines[0]['radius']), min(lines[1]['radius']))*2*numpy.pi)/cap_resolution
    model = remesh_surface(model, hsiz=hsize)
    model = model.compute_normals(auto_orient_normals=True)
    if len(tubes) > 2:
        for i in range(2, len(tubes)):
            model = boolean(model, tubes[i], operation='union')
            model = model.compute_normals(auto_orient_normals=True)
            hsize = min(hsize, (min(lines[i]['radius'])*2*numpy.pi)/cap_resolution)
            model = remesh_surface(model, hsiz=hsize)
            model = model.compute_normals(auto_orient_normals=True)
    model.hsize = hsize
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
    # Remove poor quality elements and repair the mesh.
    cell_quality = model.compute_cell_quality(quality_measure='scaled_jacobian')
    keep = cell_quality.cell_data["CellQuality"] > 0.1
    if not numpy.all(keep):
        print("Removing poor quality elements from the mesh.")
        keep = numpy.argwhere(keep).flatten()
        non_manifold_model = model.extract_cells(keep)
        non_manifold_model = non_manifold_model.extract_surface()
        fix = pymeshfix.MeshFix(non_manifold_model)
        fix.repair(verbose=True)
        hsize = model.hsize
        model = fix.mesh.compute_normals(auto_orient_normals=True)
        model.hsize = hsize
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
