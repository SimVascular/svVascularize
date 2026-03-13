from svv.tree.export.write_splines import get_branches, get_points, get_radii, get_normals
from scipy.interpolate import splprep, splev
import numpy as np
import numpy
from copy import deepcopy
import os

def export_spline(tree_connections, extrude_roots=False):
    network_branches = []
    network_points = []
    network_radii = []
    network_normals = []
    network_terminals = []
    if extrude_roots:
        for i in range(len(tree_connections.connected_network)):
            direction = (tree_connections.connected_network[i].data[0, 3:6] - tree_connections.connected_network[i].data[0, 0:3])
            direction = direction / np.linalg.norm(direction)
            root_extension = tree_connections.connected_network[i].data[0, 21] * 4
            start = tree_connections.connected_network[i].data[0, 0:3].copy()
            for j in range(10):
                new_start = start - direction * root_extension * (j + 1)
                if tree_connections.forest.domain(new_start.reshape(1, 3)).flatten() > 0:
                    tree_connections.connected_network[i].data[0, 0:3] = new_start
                    break
    for i in range(len(tree_connections.connected_network)):
        branches = get_branches(tree_connections.connected_network[i].data)
        network_branches.append(branches)
        network_points.append(get_points(tree_connections.connected_network[i].data, branches))
        network_radii.append(get_radii(tree_connections.connected_network[i].data, branches))
        network_normals.append(get_normals(tree_connections.connected_network[i].data, branches))
        terminals = []
        for branch in branches:
            terminals.append(branch[-1])
        network_terminals.append(np.array(terminals))
    # Match the branch terminals to the connection assignments
    reordered_branches = []
    reordered_points = []
    reordered_radii = []
    reordered_normals = []
    for i in range(len(tree_connections.connected_network)):
        tmp_reordered_branches = []
        tmp_reordered_points = []
        tmp_reordered_radii = []
        tmp_reordered_normals = []
        for a in range(len(tree_connections.assignments[i])):
            ind = np.argwhere(network_terminals[i] == tree_connections.assignments[i][a]).flatten()[0]
            tmp_reordered_branches.append(network_branches[i][ind])
            tmp_reordered_points.append(network_points[i][ind])
            tmp_reordered_radii.append(network_radii[i][ind])
            tmp_reordered_normals.append(network_normals[i][ind])
        reordered_branches.append(tmp_reordered_branches)
        reordered_points.append(tmp_reordered_points)
        reordered_radii.append(tmp_reordered_radii)
        reordered_normals.append(tmp_reordered_normals)
    # For trees != 0 the branches, points, radii, and normals need to be reversed
    connection_vessels = deepcopy(tree_connections.vessels)
    for i in range(1, len(reordered_branches)):
        for j in range(len(reordered_branches[i])):
            reordered_branches[i][j] = list(reversed(reordered_branches[i][j]))
            reordered_points[i][j] = list(reversed(reordered_points[i][j]))
            reordered_radii[i][j] = list(reversed(reordered_radii[i][j]))
            reordered_normals[i][j] = list(reversed(reordered_normals[i][j]))
            connection_vessels[i][j] = np.flip(connection_vessels[i][j], axis=0)
            proximal_points = connection_vessels[i][j][:, 0:3].copy()
            distal_points = connection_vessels[i][j][:, 3:6].copy()
            connection_vessels[i][j][:, 0:3] = distal_points
            connection_vessels[i][j][:, 3:6] = proximal_points
    # Take intermediate points along the connecting vessels
    connection_points = []
    connection_radii = []
    connection_normals = []
    for i in range(len(connection_vessels)):
        tmp_tree_points = []
        tmp_tree_radii = []
        tmp_tree_normals = []
        for j in range(len(connection_vessels[i])):
            tmp_branch_points = []
            tmp_branch_radii = []
            tmp_branch_normals = []
            for k in range(connection_vessels[i][j].shape[0]):
                if i > 1:
                    tmp_branch_points.append(connection_vessels[i][j][k, 0:3].tolist())
                one_fourth = connection_vessels[i][j][k, 0:3] * (3 / 4) + connection_vessels[i][j][k, 3:6] * (1 / 4)
                tmp_branch_points.append(one_fourth.tolist())
                mid = connection_vessels[i][j][k, 0:3] * (1 / 2) + connection_vessels[i][j][k, 3:6] * (1 / 2)
                tmp_branch_points.append(mid.tolist())
                three_fourths = connection_vessels[i][j][k, 0:3] * (1 / 4) + connection_vessels[i][j][k, 3:6] * (3 / 4)
                tmp_branch_points.append(three_fourths.tolist())
                if i > 1:
                    tmp_branch_radii.append(connection_vessels[i][j][k, 6])
                tmp_branch_radii.append(connection_vessels[i][j][k, 6])
                tmp_branch_radii.append(connection_vessels[i][j][k, 6])
                tmp_branch_radii.append(connection_vessels[i][j][k, 6])
                normal = connection_vessels[i][j][k, 3:6] - connection_vessels[i][j][k, 0:3]
                normal = normal / np.linalg.norm(normal)
                if i > 1:
                    tmp_branch_normals.append(normal.tolist())
                tmp_branch_normals.append(normal.tolist())
                tmp_branch_normals.append(normal.tolist())
                tmp_branch_normals.append(normal.tolist())
            tmp_tree_points.append(tmp_branch_points)
            tmp_tree_radii.append(tmp_branch_radii)
            tmp_tree_normals.append(tmp_branch_normals)
        connection_points.append(tmp_tree_points)
        connection_radii.append(tmp_tree_radii)
        connection_normals.append(tmp_tree_normals)
    # Combine the reordered points, radii, and normals with the connection points, radii, and normals
    all_points = []
    all_radii = []
    all_normals = []
    # Because the first and second vessels are lofted together
    # to ensure smoothness, the first vessel is not included
    for i in range(1, len(tree_connections.connected_network)):
        network_vessel_points = []
        network_vessel_radii = []
        network_vessel_normals = []
        for j in range(len(connection_points[i])):
            vessel = []
            radii = []
            normals = []
            if i == 1:
                vessel.extend(reordered_points[0][j])
                vessel.extend(connection_points[0][j])
                vessel.extend(connection_points[i][j])
                vessel.extend(reordered_points[i][j])
                radii.extend(reordered_radii[0][j])
                radii.extend(connection_radii[0][j])
                radii.extend(connection_radii[i][j])
                radii.extend(reordered_radii[i][j])
                normals.extend(reordered_normals[0][j])
                normals.extend(connection_normals[0][j])
                normals.extend(connection_normals[i][j])
                normals.extend(reordered_normals[i][j])
            else:
                vessel.extend(connection_points[i][j])
                vessel.extend(reordered_points[i][j])
                radii.extend(connection_radii[i][j])
                radii.extend(reordered_radii[i][j])
                normals.extend(connection_normals[i][j])
                normals.extend(reordered_normals[i][j])
            network_vessel_points.append(vessel)
            network_vessel_radii.append(radii)
            network_vessel_normals.append(normals)
        all_points.append(network_vessel_points)
        all_radii.append(network_vessel_radii)
        all_normals.append(network_vessel_normals)
    # Now we have complete sets of points, radii, and normals
    # defining the solids for each vessel
    # Create the solid
    interp_xyz = []
    interp_radii = []
    interp_normals = []
    for i in range(len(all_points)):
        network_xyz = []
        network_r = []
        network_n = []
        network_xyzr = []
        for j in range(len(all_points[i])):
            p = np.array(all_points[i][j]).T
            r = np.array(all_radii[i][j]).T
            n = np.array(all_normals[i][j]).T
            if p.shape[1] == 2:
                network_xyz.append(splprep(p, k=1, s=0))
                rr = numpy.vstack((network_xyz[-1][1], r))
                network_r.append(splprep(rr, k=1, s=0))
                xyzr = numpy.vstack((p, r))
                network_xyzr.append(splprep(xyzr, k=1, s=0))
                # interp_n.append(splprep(n, k=1, s=0))
            elif p.shape[1] == 3:
                network_xyz.append(splprep(p, k=2, s=0))
                rr = numpy.vstack((network_xyz[-1][1], r))
                network_r.append(splprep(rr, k=1, s=0))
                xyzr = numpy.vstack((p, r))
                network_xyzr.append(splprep(xyzr, k=2, s=0))
                # interp_n.append(splprep(n, k=2, s=0))
            else:
                network_xyz.append(splprep(p, s=0))
                rr = numpy.vstack((network_xyz[-1][1], r))
                network_r.append(splprep(rr, k=1, s=0))
                xyzr = numpy.vstack((p, r))
                network_xyzr.append(splprep(xyzr, s=0))
                # interp_n.append(splprep(n, s=0))
        interp_xyz.append(network_xyz)
        interp_radii.append(network_r)
        interp_normals.append(network_n)
    return interp_xyz, interp_radii, interp_normals, all_points, all_radii, all_normals


def write_splines(ALL_POINTS, ALL_RADII, spline_sample_points=100, seperate=False, write_splines=True, outdir=None,
                  name_prefix=None):
    ALL_SPLINES = []
    if write_splines and outdir is None:
        outdir = os.getcwd()
    for network in range(len(ALL_POINTS)):
        network_splines = []
        if write_splines:
            if name_prefix is None:
                name = "network_{}_b_splines.txt".format(network)
            else:
                name = "{}_network_{}_b_splines.txt".format(name_prefix, network)
            spline_path = outdir + os.sep + name
            with open(spline_path, "w+", encoding="utf-8") as spline_file:
                for vessel in range(len(ALL_POINTS[network])):
                    pt_array = np.array(ALL_POINTS[network][vessel])
                    r_array = np.array(ALL_RADII[network][vessel]).reshape(-1, 1)
                    pt_r_combined = deepcopy(np.hstack((pt_array, r_array)).T)
                    vessel_ctr = splprep(pt_r_combined, s=0)

                    def vessel_spline(t, ctr=vessel_ctr):
                        return splev(t, ctr[0])

                    network_splines.append(deepcopy(vessel_spline))
                    spline_file.write('Vessel: {}, Number of Points: {}\n\n'.format(vessel, spline_sample_points))
                    t = np.linspace(0, 1, num=spline_sample_points)
                    data = deepcopy(vessel_spline(t))
                    for k in range(spline_sample_points):
                        label = 1 if k > spline_sample_points // 2 else 0
                        if seperate:
                            spline_file.write(
                                '{}, {}, {}, {}, {}\n'.format(
                                    data[0][k], data[1][k], data[2][k], data[3][k], label
                                )
                            )
                        else:
                            spline_file.write(
                                '{}, {}, {}, {}\n'.format(data[0][k], data[1][k], data[2][k], data[3][k])
                            )
                    spline_file.write('\n')
        else:
            for vessel in range(len(ALL_POINTS[network])):
                pt_array = np.array(ALL_POINTS[network][vessel])
                r_array = np.array(ALL_RADII[network][vessel]).reshape(-1, 1)
                pt_r_combined = deepcopy(np.hstack((pt_array, r_array)).T)
                vessel_ctr = splprep(pt_r_combined, s=0)

                def vessel_spline(t, ctr=vessel_ctr):
                    return splev(t, ctr[0])

                network_splines.append(deepcopy(vessel_spline))
        ALL_SPLINES.append(network_splines)
    return ALL_SPLINES
