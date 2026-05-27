import numpy
import pyvista as pv
from svv.forest.connect.base_connection import BaseConnection


def _build_tree_collision_segments(tree_data, exclude_idx):
    """Build collision segment array for a tree, excluding specified vessel and its children."""
    n = tree_data.shape[0]
    keep = numpy.ones(n, dtype=bool)
    vessel = tree_data[exclude_idx, :]

    # Exclude the connected vessel itself
    if not numpy.isnan(vessel[17]):
        keep[exclude_idx] = False
    # Exclude daughters if they exist
    if not numpy.isnan(vessel[15]):
        keep[int(vessel[15])] = False
    if not numpy.isnan(vessel[16]):
        keep[int(vessel[16])] = False

    idx = numpy.nonzero(keep)[0]
    if idx.size == 0:
        return numpy.zeros((0, 7), dtype=float)

    tmp = numpy.empty((idx.size, 7), dtype=float)
    tmp[:, 0:3] = tree_data[idx, 0:3]
    tmp[:, 3:6] = tree_data[idx, 3:6]
    tmp[:, 6] = tree_data[idx, 21]
    return tmp


class VesselConnection:
    def __init__(self, forest, network_id, tree_0, tree_1, idx, jdx,
                 ctrl_function=None, clamp_first=True, clamp_second=True,
                 point_0=None, point_1=None, curve_type="Bezier", collision_vessels=None):
        self.forest = forest
        self.tree_0 = tree_0
        self.tree_1 = tree_1
        self.idx = idx
        self.jdx = jdx
        vessel_0 = forest.networks[network_id][tree_0].data[idx, :]
        vessel_1 = forest.networks[network_id][tree_1].data[jdx, :]
        self.proximal_0 = vessel_0[0:3]
        self.distal_0 = vessel_0[3:6]
        self.proximal_1 = vessel_1[0:3]
        self.distal_1 = vessel_1[3:6]
        self.radius_0 = vessel_0[21]
        self.radius_1 = vessel_1[21]
        min_distance = max(self.radius_0, self.radius_1) * 0.5
        conn = BaseConnection(vessel_0[0:3], vessel_0[3:6], vessel_1[0:3], vessel_1[3:6], vessel_0[21], vessel_1[21],
                              domain=forest.domain, ctrlpt_function=ctrl_function, clamp_first=clamp_first,
                              clamp_second=clamp_second, point_0=point_0, point_1=point_1, min_distance=min_distance,
                              curve_type=curve_type)
        if collision_vessels is None:
            collision_list = []
            # Build collision segments for both trees (no deepcopy needed - tmp arrays are freshly allocated)
            collision_list.append(_build_tree_collision_segments(
                forest.networks[network_id][tree_0].data, idx))
            collision_list.append(_build_tree_collision_segments(
                forest.networks[network_id][tree_1].data, jdx))

            # Add segments from other networks
            for i in range(forest.n_networks):
                for j in range(forest.n_trees_per_network[i]):
                    if i == network_id and (j == tree_0 or j == tree_1):
                        continue
                    data = forest.networks[i][j].data
                    n = data.shape[0]
                    tmp = numpy.empty((n, 7), dtype=float)
                    tmp[:, 0:3] = data[:, 0:3]
                    tmp[:, 3:6] = data[:, 3:6]
                    tmp[:, 6] = data[:, 21]
                    collision_list.append(tmp)

            # Filter empty arrays and stack
            collision_list = [a for a in collision_list if a.shape[0] > 0]
            if collision_list:
                conn.set_collision_vessels(numpy.vstack(collision_list))
        else:
            collision_vessels = numpy.asarray(collision_vessels, dtype=float)
            if collision_vessels.ndim == 2 and collision_vessels.shape[1] >= 6 and collision_vessels.shape[0] > 0:
                if collision_vessels.shape[1] == 6:
                    tmp = numpy.zeros((collision_vessels.shape[0], 7), dtype=float)
                    tmp[:, 0:6] = collision_vessels[:, 0:6]
                    collision_vessels = tmp
                conn.set_collision_vessels(collision_vessels)
        conn.set_physical_clearance(self.forest.networks[network_id][tree_0].domain_clearance)
        bounds = numpy.zeros((3, 2))
        bounds[:, 0] = numpy.min(forest.domain.points, axis=0).T
        bounds[:, 1] = numpy.max(forest.domain.points, axis=0).T
        conn.clamp_first = clamp_first
        conn.clamp_second = clamp_second
        self.connection = conn
        self.curve = None
        self.result = None
        self.vessels_1 = None
        self.vessels_2 = None
        self.vessel_1_meshes = None
        self.vessel_2_meshes = None
        self.seperate = None
        self.plotting_vessels = None

    def solve(self, *args, **kwargs):
        result, curve, objective, constraints = self.connection.solve(*args, **kwargs)
        self.result = result
        self.curve = curve
        self.objective = objective
        self.constraints = constraints

    def build_vessels(self, num, seperate=True, build_meshes=True):
        t = numpy.linspace(0, 1, num)
        pts = self.curve.evaluate(t)
        if seperate:
            sep = (pts.shape[0]-1)//2
            vessels_1 = numpy.zeros((sep,7))
            vessels_2 = numpy.zeros((pts.shape[0]-1-sep,7))
            vessels_1_pts = pts[:sep+1,:]
            vessels_2_pts = numpy.flip(pts[sep:,:],axis=0)
            vessels_1[:, 0:3] = vessels_1_pts[:-1,:]
            vessels_1[:, 3:6] = vessels_1_pts[1:,:]
            vessels_2[:, 0:3] = vessels_2_pts[:-1,:]
            vessels_2[:, 3:6] = vessels_2_pts[1:,:]
            vessels_1[:, 6] = self.radius_0
            vessels_2[:, 6] = self.radius_1
        else:
            vessels_1 = numpy.zeros((pts.shape[0]-1,7))
            vessels_1[:,0:3] = pts[:-1,:]
            vessels_1[:,3:6] = pts[1:,:]
            vessels_1[:,6] = self.radius_0
            pts_reverse = numpy.flip(pts,axis=0)
            vessels_2 = numpy.zeros((pts.shape[0]-1,7))
            vessels_2[:,0:3] = pts_reverse[:-1,:]
            vessels_2[:,3:6] = pts_reverse[1:,:]
            vessels_2[:,6] = self.radius_1
        self.vessels_1 = vessels_1
        self.vessels_2 = vessels_2
        self.seperate = seperate
        self.vessel_1_meshes = []
        if build_meshes:
            for i in range(vessels_1.shape[0]):
                center = (self.vessels_1[i, 0:3] + self.vessels_1[i, 3:6]) / 2
                direction = self.vessels_1[i, 3:6] - self.vessels_1[i, 0:3]
                length = numpy.linalg.norm(direction)
                direction = direction / length
                radius = self.vessels_1[i, 6]
                cylinder = pv.Cylinder(center=center, direction=direction, radius=radius, height=length)
                self.vessel_1_meshes.append(cylinder)
            self.vessel_2_meshes = []
            for i in range(self.vessels_2.shape[0]):
                center = (self.vessels_2[i, 0:3] + self.vessels_2[i, 3:6]) / 2
                direction = self.vessels_2[i, 3:6] - self.vessels_2[i, 0:3]
                length = numpy.linalg.norm(direction)
                direction = direction / length
                radius = self.vessels_2[i, 6]
                cylinder = pv.Cylinder(center=center, direction=direction, radius=radius, height=length)
                self.vessel_2_meshes.append(cylinder)
        return vessels_1, vessels_2

    def show(self):
        plotter = pv.Plotter()
        if self.seperate:
            for i in range(self.vessels_1.shape[0]):
                center = (self.vessels_1[i,0:3]+self.vessels_1[i,3:6])/2
                direction = self.vessels_1[i,3:6] - self.vessels_1[i,0:3]
                length = numpy.linalg.norm(direction)
                direction = direction/length
                radius = self.vessels_1[i,6]
                cylinder = pv.Cylinder(center=center,direction=direction,radius=radius,height=length)
                plotter.add_mesh(cylinder,color='red')
            for i in range(self.vessels_2.shape[0]):
                center = (self.vessels_2[i,0:3]+self.vessels_2[i,3:6])/2
                direction = self.vessels_2[i,3:6] - self.vessels_2[i,0:3]
                length = numpy.linalg.norm(direction)
                direction = direction/length
                radius = self.vessels_2[i,6]
                cylinder = pv.Cylinder(center=center,direction=direction,radius=radius,height=length)
                plotter.add_mesh(cylinder,color='blue')
        else:
            for i in range(self.vessels_1.shape[0]):
                center = (self.vessels_1[i,0:3]+self.vessels_1[i,3:6])/2
                direction = self.vessels_1[i,3:6] - self.vessels_1[i,0:3]
                length = numpy.linalg.norm(direction)
                direction = direction/length
                radius = self.vessels_1[i,6]
                cylinder = pv.Cylinder(center=center,direction=direction,radius=radius,height=length)
                plotter.add_mesh(cylinder,color='red')
        if self.plotting_vessels is not None:
            for i in range(self.plotting_vessels.shape[0]):
                center = (self.plotting_vessels[i,0:3]+self.plotting_vessels[i,3:6])/2
                direction = self.plotting_vessels[i,3:6] - self.plotting_vessels[i,0:3]
                length = numpy.linalg.norm(direction)
                direction = direction/length
                radius = self.plotting_vessels[i,6]
                cylinder = pv.Cylinder(center=center,direction=direction,radius=radius,height=length)
                plotter.add_mesh(cylinder,color='black')
        if numpy.all(self.proximal_0 != self.distal_0):
            center = (self.proximal_0+self.distal_0)/2
            direction = (self.distal_0-self.proximal_0)
            length = numpy.linalg.norm(direction)
            direction = direction/length
            radius = self.radius_0
            cylinder = pv.Cylinder(center=center,direction=direction,radius=radius,height=length)
            plotter.add_mesh(cylinder,color='yellow')
        if numpy.all(self.proximal_1 != self.distal_1):
            center = (self.proximal_1+self.distal_1)/2
            direction = (self.distal_1-self.proximal_1)
            length = numpy.linalg.norm(direction)
            direction = direction/length
            radius = self.radius_1
            cylinder = pv.Cylinder(center=center,direction=direction,radius=radius,height=length)
            plotter.add_mesh(cylinder,color='yellow')
        return plotter
