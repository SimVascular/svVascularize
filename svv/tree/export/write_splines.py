from scipy.interpolate import splprep, splev
import numpy as np
import numpy
from copy import deepcopy
import os

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

def write_splines(interp_xyzr, spline_sample_points=100, write_splines=True):
    tree_splines = []
    if write_splines:
        spline_file = open(os.getcwd() + os.sep + "tree_b_splines.txt", "w+")
    for vessel in range(len(interp_xyzr)):
        def vessel_spline(t, ctr=interp_xyzr[vessel]):
            return splev(t, ctr[0])
        tree_splines.append(deepcopy(vessel_spline))
        if write_splines:
            spline_file.write('Vessel: {}, Number of Points: {}\n\n'.format(vessel, spline_sample_points))
            t = np.linspace(0, 1, num=spline_sample_points)
            data = deepcopy(vessel_spline(t))
            for k in range(spline_sample_points):
                spline_file.write('{}, {}, {}, {}\n'.format(data[0][k], data[1][k], data[2][k], data[3][k]))
            spline_file.write('\n')
    if write_splines:
        spline_file.close()
    return tree_splines
