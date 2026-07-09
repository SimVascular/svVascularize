import numpy as np
import pyvista as pv
from pyvista import examples
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

from svv.utils.spatial.c_distance import minimum_segment_distance, minimum_self_segment_distance
from svv.forest.connect.curve import Curve

def random_point_in_mesh(mesh: pv.PolyData, max_tries=1000):
    """
    Returns a single random point strictly inside a watertight mesh
    using rejection sampling within its bounding box.
    Raises a ValueError if it fails to find a point after `max_tries`.
    """
    # Get bounding box of the mesh
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds

    # Sample a random point in the bounding box
    x = np.random.uniform(xmin, xmax, max_tries)
    y = np.random.uniform(ymin, ymax, max_tries)
    z = np.random.uniform(zmin, zmax, max_tries)
    candidate = np.array([x, y, z]).T  # shape (1, 3)

    # Convert to PyVista object
    candidate_points = pv.PolyData(candidate)

    # Check if point is inside using `select_enclosed_points`
    enclosed = candidate_points.select_enclosed_points(mesh)
    if np.any(enclosed["SelectedPoints"] == 1):  # 1 means inside
        return candidate_points.extract_points(enclosed['SelectedPoints'].view(bool))  # return as a 1D array [x, y, z]
    else:
        raise ValueError("Could not find a random interior point within max_tries.")


def random_line_segment_in_mesh(mesh: pv.PolyData,
                                resolution=10,
                                max_tries=1000):
    """
    Returns two endpoints (p1, p2) that lie fully inside `mesh` (watertight).
    The entire line from p1 -> p2 is checked to ensure all sampled points
    are inside. Uses your updated random_point_in_mesh() function to pick
    endpoints from multiple inside points at once.

    Parameters
    ----------
    mesh : pv.PolyData
        A watertight PyVista mesh.
    resolution : int
        Number of points along the line segment for the inside check.
    max_tries : int
        Maximum number of attempts to find a fully enclosed line segment.

    Returns
    -------
    (p1, p2) : tuple of np.ndarray
        Each is a 3D coordinate array representing one endpoint of the segment.

    Raises
    ------
    ValueError
        If no fully enclosed line segment could be found in `max_tries` attempts.
    """
    for attempt in range(max_tries):
        # -----------------------------------------------------------
        # 1) Get a set of random points inside the mesh for endpoint 1
        # -----------------------------------------------------------
        try:
            inside_points_1 = random_point_in_mesh(mesh, max_tries=max_tries)
            # 'inside_points_1' is a pv.PolyData with one or more points
            n_inside_1 = inside_points_1.n_points
            # Randomly choose one point among them
            idx1 = np.random.randint(n_inside_1)
            p1 = inside_points_1.points[idx1]
        except ValueError:
            # Could not find a single inside point within max_tries
            continue

        # -----------------------------------------------------------
        # 2) Get a set of random points inside the mesh for endpoint 2
        # -----------------------------------------------------------
        try:
            inside_points_2 = random_point_in_mesh(mesh, max_tries=max_tries)
            n_inside_2 = inside_points_2.n_points
            idx2 = np.random.randint(n_inside_2)
            p2 = inside_points_2.points[idx2]
        except ValueError:
            # Could not find a single inside point within max_tries
            continue

        # If p1 == p2 (or extremely close), skip
        if np.allclose(p1, p2, atol=1e-12):
            continue

        # -----------------------------------------------------------
        # 3) Discretize the line between p1 and p2
        # -----------------------------------------------------------
        t_values = np.linspace(0, 1, resolution)
        line_points_arr = (1.0 - t_values)[:, None] * p1 + t_values[:, None] * p2
        line_points = pv.PolyData(line_points_arr)

        # -----------------------------------------------------------
        # 4) Check if *all* discretized points are inside the mesh
        # -----------------------------------------------------------
        enclosed = line_points.select_enclosed_points(mesh)
        inside_mask = enclosed["SelectedPoints"]  # array of 0 or 1

        if np.all(inside_mask == 1):
            # Found a line segment fully inside
            return p1, p2

    # If we exhaust all tries without finding a fully inside segment, raise
    raise ValueError("Failed to find a line segment fully inside the mesh within max_tries.")
def random_line_segments_in_mesh(mesh: pv.PolyData,
                                 n_segments=5,
                                 resolution=10,
                                 max_tries=2000):
    """
    Generates and returns `n_segments` random line segments
    lying fully inside a watertight mesh.

    Returns
    -------
    segments : list of tuples
        Each tuple is (p1, p2), where both p1 and p2 are
        3-element np.array of coordinates.
    """
    segments = []
    for _ in range(n_segments):
        p1, p2 = random_line_segment_in_mesh(mesh, resolution, max_tries)
        segments.append((p1, p2))
    return segments

class BaseConnection:
    """
    A base connection class that computes an objective function and
    various geometric constraints for an optimized curve connecting
    four points (A->B and C->D) subject to optional clamp constraints.
    """
    def __init__(
        self,
        point_00,
        point_01,
        point_10,
        point_11,
        radius_0,
        radius_1,
        domain=None,
        ctrlpt_function=None,
        clamp_first=True,
        clamp_second=True,
        point_0=None,
        point_1=None,
        min_distance=1e-4,
        clearance=0.0,
        curve_type='Bezier'
    ):
        # Radii
        self.radius_0 = radius_0
        self.radius_1 = radius_1

        # Assign A, B, C, D
        # If point_0 is given, use that for A & B; otherwise, use point_00, point_01.
        if point_0 is not None:
            self.A = point_0
            self.B = point_1
        else:
            self.A = point_00
            self.B = point_01

        # If point_1 is given, use that for C & D; otherwise, use point_10, point_11.
        if point_1 is not None:
            self.C = point_1
            self.D = point_1
        else:
            self.C = point_10
            self.D = point_11

        # Store line segments, domain, etc.
        self.other_line_segments = []
        self.min_distance = min_distance
        self.ctrlpt_function = ctrlpt_function
        self.clamp_first = clamp_first
        self.clamp_second = clamp_second
        self.domain = domain
        self.clearance = clearance
        self.curve_type = curve_type

        # Domain bounding box
        if domain is not None:
            mins = np.min(domain.points, axis=0) * 2
            maxs = np.max(domain.points, axis=0) * 2
            self.x_min, self.x_max = mins[0], maxs[0]
            self.y_min, self.y_max = mins[1], maxs[1]
            self.z_min, self.z_max = mins[2], maxs[2]
        else:
            self.x_min, self.x_max = 0, 1
            self.y_min, self.y_max = 0, 1
            self.z_min, self.z_max = 0, 1

        # Keep track of curves for optional debugging/plotting.
        # In production we only retain the final curve to avoid
        # unbounded memory growth during optimization.
        self.history_curves = []

    def _build_control_points(self, ctrlpts_flat, mid, n_pts):
        """
        Construct the full array of control points from the flattened
        optimization variables, taking into account any clamping.
        """
        # Precompute direction unit vectors if clamps are active
        if self.clamp_first:
            BA_dir = (self.B - self.A)
            BA_dir /= np.linalg.norm(BA_dir)
        if self.clamp_second:
            DC_dir = (self.D - self.C)
            DC_dir /= np.linalg.norm(DC_dir)

        # P0 is a mix between A and B; similarly for P3 (mix between C and D)
        P0 = self.A * (1 - ctrlpts_flat[0]) + self.B * ctrlpts_flat[0]
        if self.clamp_first:
            P1 = P0 + ctrlpts_flat[1] * BA_dir

        P3 = self.C * (1 - ctrlpts_flat[-1]) + self.D * ctrlpts_flat[-1]
        if self.clamp_second:
            P2 = P3 + ctrlpts_flat[-2] * DC_dir

        # Midpoints from the 'mid' function
        mid_points = mid(np.linspace(0.25, 0.75, n_pts))

        # Decide how many in the middle (depending on clamps)
        if self.clamp_first and self.clamp_second:
            ctrlpts_middle = mid_points + ctrlpts_flat[2:-2].reshape(-1, 3)
            all_points = np.vstack([P0, P1, ctrlpts_middle, P2, P3])

        elif self.clamp_first:
            ctrlpts_middle = mid_points + ctrlpts_flat[2:-1].reshape(-1, 3)
            all_points = np.vstack([P0, P1, ctrlpts_middle, P3])

        elif self.clamp_second:
            ctrlpts_middle = mid_points + ctrlpts_flat[1:-2].reshape(-1, 3)
            all_points = np.vstack([P0, ctrlpts_middle, P2, P3])

        else:
            ctrlpts_middle = mid_points + ctrlpts_flat[1:-1].reshape(-1, 3)
            all_points = np.vstack([P0, ctrlpts_middle, P3])

        return all_points

    def create_objective(self, t_num, mid, n_pts):
        """
        Returns the objective function (callable) that takes in a
        flattened array of control points and returns a scalar.
        """
        def objective(control_points_flat):
            # Build the control points and the corresponding curve
            control_points = self._build_control_points(control_points_flat, mid, n_pts)
            curve = Curve(control_points, curve_type=self.curve_type)

            # Only track the most recent curve for optional debugging/plotting.
            # This avoids unbounded history growth during optimization.
            self.history_curves = [curve]

            # Example objective: difference between curve length and some nominal length
            # length = np.linalg.norm(self.B - self.A) + np.linalg.norm(self.D - self.C)
            # or you can do something else:
            length = np.linalg.norm(self.D - self.B)
            return max(0.0, curve.arc_length(num_points=t_num) - length)

        return objective

    def create_constraints(self,radius1,radius2,other_line_segments,t_num,mid,n_pts):
        """
        Returns a list of constraint dictionaries to be used in a
        typical optimizer (e.g., scipy.optimize).

        Uses a shared cache so control points, curve, and evaluations
        are computed once per unique optimizer input and shared across
        all constraint functions.
        """
        # Shared cache: avoids rebuilding control points and curve
        # for every constraint in the same optimizer iteration.
        _cache = {'key': None, 'ctrl_pts': None, 'curve': None,
                  'curve_pts_t': None, 'curve_pts_roc': None}

        # Pre-compute constant values
        min_radius_needed = 2 * max(radius1, radius2)
        max_radius = max(self.radius_0, self.radius_1)
        t_values_curve = np.linspace(0, 1, t_num)
        t_values_roc = np.linspace(0, 1, 50)  # Reduced from 100 for curvature
        t_values_boundary = np.linspace(0, 1, 40)  # Reduced from 100 for boundary
        has_collision_segments = (len(self.other_line_segments) > 0 if
                                  hasattr(self.other_line_segments, '__len__') else False)

        # Pre-compute collision radii offset if there are collision segments
        if has_collision_segments:
            _collision_radii = self.other_line_segments[:, 6].reshape(-1, 1) + max_radius + self.min_distance

        def _get_cached(ctrlpts_flat):
            """Return (control_points, curve) from cache or recompute."""
            key = ctrlpts_flat.tobytes()
            if _cache['key'] != key:
                _cache['key'] = key
                _cache['ctrl_pts'] = self._build_control_points(ctrlpts_flat, mid, n_pts)
                _cache['curve'] = Curve(_cache['ctrl_pts'], curve_type=self.curve_type)
                _cache['curve_pts_t'] = None
                _cache['curve_pts_roc'] = None
            return _cache['ctrl_pts'], _cache['curve']

        def _get_curve_points(ctrlpts_flat, variant='t'):
            """Return cached curve points for the given evaluation set."""
            _, curve = _get_cached(ctrlpts_flat)
            cache_key = 'curve_pts_' + variant
            if _cache[cache_key] is None:
                if variant == 't':
                    _cache[cache_key] = curve.evaluate(t_values_curve)
                elif variant == 'roc':
                    _cache[cache_key] = None  # roc uses its own t values
            return _cache[cache_key]

        def curvature_constraint(ctrlpts_flat):
            _, curve = _get_cached(ctrlpts_flat)
            roc_values = curve.roc(t_values_roc)
            return np.min(roc_values[1:-1]) - min_radius_needed

        def non_coincidence_constraint(ctrlpts_flat):
            control_points, _ = _get_cached(ctrlpts_flat)
            distances = pdist(control_points)
            return np.min(distances) - 1e-4

        def curve_min_distance_constraint(ctrlpts_flat):
            if not has_collision_segments:
                return 1.0

            curve_points = _get_curve_points(ctrlpts_flat, 't')
            if curve_points is None:
                _, curve = _get_cached(ctrlpts_flat)
                curve_points = curve.evaluate(t_values_curve)
                _cache['curve_pts_t'] = curve_points

            # Build segments from discretized curve
            segments = np.empty((curve_points.shape[0] - 1, 6))
            segments[:, :3] = curve_points[:-1]
            segments[:, 3:] = curve_points[1:]

            dist_matrix = minimum_segment_distance(self.other_line_segments[:, :6], segments)
            dist_main = np.min(dist_matrix - _collision_radii)
            return dist_main

        def boundary_constraint(ctrlpts_flat):
            if self.domain is None:
                return 1.0

            _, curve = _get_cached(ctrlpts_flat)
            curve_points = curve.evaluate(t_values_boundary)

            if callable(self.domain):
                values = self.domain(curve_points)
                return -(np.max(values) + self.clearance)
            elif isinstance(self.domain, pv.PolyData):
                temp_points = pv.PolyData(curve_points)
                enclosed_result = temp_points.select_enclosed_points(self.domain, tolerance=0.0)
                inside_mask = enclosed_result['SelectedPoints']
                return 1.0 if np.all(inside_mask) else -1.0
            else:
                return 1.0

        def self_collision(ctrlpts_flat):
            curve_points = _get_curve_points(ctrlpts_flat, 't')
            if curve_points is None:
                _, curve = _get_cached(ctrlpts_flat)
                curve_points = curve.evaluate(t_values_curve)
                _cache['curve_pts_t'] = curve_points
            segments = np.empty((t_num - 1, 6))
            segments[:, :3] = curve_points[:-1]
            segments[:, 3:6] = curve_points[1:]
            dist_main = minimum_self_segment_distance(segments) - max_radius - self.min_distance
            return dist_main

        return [
            {'type': 'ineq', 'fun': curvature_constraint},
            {'type': 'ineq', 'fun': curve_min_distance_constraint},
            {'type': 'ineq', 'fun': non_coincidence_constraint},
            {'type': 'ineq', 'fun': boundary_constraint},
        ]

    def solve(self, n_mid_pts, t_num=20):
        """
        Solves for an optimized curve connecting A->B and C->D using
        an objective function and constraints. Clamping can be applied
        to the start/end tangents if desired (clamp_first/clamp_second).

        Parameters
        ----------
        n_mid_pts : int
            Number of 'middle' control points used in the curve construction.
            This was originally args[0].
        t_num : int
            Number of points used for discretizing/evaluating the curve
            for the objective and constraints.

        Returns
        -------
        result : OptimizeResult
            The result returned by scipy.optimize.minimize (includes x, success, etc.).
        curve : Curve
            The final Curve object constructed from the optimized control points.
        """
        # 1) Build an interpolation ('mid') function from B->D
        #    if no custom ctrlpt_function is provided.
        pts_bd = np.vstack((self.B, self.D))
        xpts = interp1d([0, 1], pts_bd[:, 0], kind='linear')
        ypts = interp1d([0, 1], pts_bd[:, 1], kind='linear')
        zpts = interp1d([0, 1], pts_bd[:, 2], kind='linear')

        if self.ctrlpt_function is None or isinstance(self.ctrlpt_function, list):
            # By default, just linearly interpolate between B and D
            def mid(t):
                return np.array([xpts(t), ypts(t), zpts(t)]).T
        else:
            # Use a custom function (presumably expects t in [0,1])
            mid = self.ctrlpt_function

        # 2) Build the objective and constraints
        objective = self.create_objective(t_num, mid, n_mid_pts)
        constraints = self.create_constraints(
            self.radius_0,
            self.radius_1,
            self.other_line_segments,
            t_num,
            mid,
            n_mid_pts
        )

        # 3) Create an initial guess for the optimization variables
        #    - Format: [ clamp_first_alpha, clamp_first_dist, (3*n_mid_pts) mid devs..., clamp_second_dist, clamp_second_alpha ]
        #    - The shape is n_mid_pts * 3 (x,y,z per middle point) + 4
        #      (two scalars for clamp_first, two scalars for clamp_second).
        initial_control_points = np.zeros(n_mid_pts * 3 + 4)
        # Some example initial values for the clamps:
        initial_control_points[0] = 0.85  # clamp_first_alpha
        initial_control_points[1] = 1.0  # clamp_first_dist
        initial_control_points[-2] = 1.0  # clamp_second_dist
        initial_control_points[-1] = 0.85  # clamp_second_alpha

        # 4) Construct bounds for each parameter
        length = np.linalg.norm(self.B - self.D)
        lb = [-length] * initial_control_points.size
        ub = [length] * initial_control_points.size

        # Example clamp bounds
        lb[0], ub[0] = 0.25, 1.0  # clamp_first_alpha ∈ [0.5, 1]
        lb[1], ub[1] = 0.1 * length, 10 * length  # clamp_first_dist  ∈ [0.1*L, 10*L]
        lb[-1], ub[-1] = 0.25, 1.0  # clamp_second_alpha ∈ [0.5, 1]
        lb[-2], ub[-2] = 0.1 * length, 10 * length  # clamp_second_dist ∈ [0.1*L, 10*L]

        # 5) Remove clamp-related parameters if not used
        #    (i.e., if clamp_first=False and/or clamp_second=False).
        if self.clamp_first and self.clamp_second:
            pass  # keep all four clamp variables
        elif self.clamp_first:
            # Remove the second-to-last element (clamp_second_dist)
            # because we are not using the second clamp's tangent distance
            initial_control_points = np.delete(initial_control_points, -2)
            del lb[-2], ub[-2]
        elif self.clamp_second:
            # Remove the second element (clamp_first_dist)
            # because we are not using the first clamp's tangent distance
            initial_control_points = np.delete(initial_control_points, 1)
            del lb[1], ub[1]
        else:
            # Remove both clamp distances (second-to-last and the second element)
            initial_control_points = np.delete(initial_control_points, [-2, 1])
            del lb[-2], ub[-2]
            del lb[1], ub[1]

        bounds = list(zip(lb, ub))

        # 6) Optimize
        result = minimize(objective,initial_control_points,method='SLSQP',constraints=constraints,bounds=bounds)

        # 7) Reconstruct the final control points from the optimized result
        #    - We'll replicate the logic used in _build_control_points or
        #      expand it here in-line.

        opt_x = result.x  # For clarity

        if self.clamp_first:
            BA = (self.B - self.A)
            BA /= np.linalg.norm(BA)
        if self.clamp_second:
            DC = (self.D - self.C)
            DC /= np.linalg.norm(DC)

        # P0: blend of A and B by alpha
        P0 = self.A * (1 - opt_x[0]) + self.B * opt_x[0]
        # If clamp_first, we have a distance param for the tangent
        if self.clamp_first:
            P1 = P0 + opt_x[1] * BA

        # P3: blend of C and D by alpha
        P3 = self.C * (1 - opt_x[-1]) + self.D * opt_x[-1]
        # If clamp_second, we have a distance param for that tangent
        if self.clamp_second:
            P2 = P3 + opt_x[-2] * DC

        # Middle control points
        mid_points = mid(np.linspace(0.25, 0.75, n_mid_pts))

        # Indices for the middle portion depend on clamps being present
        if self.clamp_first and self.clamp_second:
            middle_flat = opt_x[2:-2]
        elif self.clamp_first:
            middle_flat = opt_x[2:-1]
        elif self.clamp_second:
            middle_flat = opt_x[1:-2]
        else:
            middle_flat = opt_x[1:-1]

        middle_deviations = middle_flat.reshape(-1, 3)
        control_points = mid_points + middle_deviations

        # Stack everything together
        if self.clamp_first and self.clamp_second:
            control_points = np.vstack([P0, P1, control_points, P2, P3])
        elif self.clamp_first:
            control_points = np.vstack([P0, P1, control_points, P3])
        elif self.clamp_second:
            control_points = np.vstack([P0, control_points, P2, P3])
        else:
            control_points = np.vstack([P0, control_points, P3])

        # 8) Build final curve and return. Overwrite history_curves so
        # that only the final optimized curve is retained.
        curve = Curve(control_points, curve_type=self.curve_type)
        self.history_curves = [curve]
        return result, curve, objective, constraints

    def set_collision_vessels(self, collision_vessels):
        self.other_line_segments = collision_vessels

    def set_physical_clearance(self, clearance):
        self.clearance = clearance

    def plot(self, n_points=50, plot_as_tube=True):
        """
        Plots the final optimized connection curve (if it exists) and any
        additional line segments stored in `self.other_line_segments` as cylinders.

        Parameters
        ----------
        n_points : int, optional
            Number of discrete points to use when sampling the final curve
            for visualization.
        plot_as_tube : bool, optional
            If True, the optimized curve is also displayed as a tube with a
            small radius. Otherwise, it's drawn as a simple line.
        """
        # 1) Check if a final solution (curve) is available.
        if not self.history_curves:
            print("No solution found yet. Please run solve() first.")
            return
        # 2) Get the final curve from the history and discretize it
        final_curve = self.history_curves[-1]
        t_vals = np.linspace(0, 1, n_points)
        curve_points = final_curve.evaluate(t_vals)
        # 3) Initialize a PyVista plotter
        plotter = pv.Plotter()
        if isinstance(self.domain, pv.PolyData):
            plotter.add_mesh(self.domain, opacity=0.25)
        # 4) Plot the final curve
        if plot_as_tube:
            # Create a PolyData line, then use .tube(...) to make it a tubular mesh
            vessels = np.zeros((curve_points.shape[0]-1, 7))
            vessels[:, 0:3] = curve_points[:-1, :]
            vessels[:, 3:6] = curve_points[1:, :]
            vessels[:, 6] = max(self.radius_0, self.radius_1)
            meshes = []
            for i in range(vessels.shape[0]):
                center = (vessels[i, 0:3] + vessels[i, 3:6])/2
                direction = (vessels[i, 3:6] - vessels[i, 0:3])
                length = np.linalg.norm(direction)
                direction = direction/length
                cyl = pv.Cylinder(radius = (self.radius_0 + self.radius_1)/2, center = center, direction = direction,
                                  height=length, capping=True)
                meshes.append(cyl)
            tube = pv.merge(meshes)
            plotter.add_mesh(tube, color='blue', label='Optimized Curve')
            #plot first vessel
            center = (vessels[0, 0:3] + self.A)/2
            direction = (vessels[0, 0:3] - self.A)
            length = np.linalg.norm(direction)
            direction = direction/length
            cyl = pv.Cylinder(radius=(self.radius_0 + self.radius_1) / 2, center=center, direction=direction,
                              height=length, capping=True)
            plotter.add_mesh(cyl, color='green', label='Start')
            center = (vessels[-1, 3:6] + self.C)/2
            direction = (self.C - vessels[-1, 3:6])
            length = np.linalg.norm(direction)
            direction = direction/length
            cyl = pv.Cylinder(radius=(self.radius_0 + self.radius_1) / 2, center=center, direction=direction,
                              height=length, capping=True)
            plotter.add_mesh(cyl, color='green', label='End')

        else:
            # Just plot as a simple line
            curve_polydata = pv.lines_from_points(curve_points)
            plotter.add_mesh(curve_polydata, color='blue', line_width=4, label='Optimized Curve')

        # 5) Plot other line segments as cylinders
        #    We assume each segment is [x1, y1, z1, x2, y2, z2, radius].
        #    If the radius is missing, we default to a small number.
        if len(self.other_line_segments) > 0:
            for i, seg in enumerate(self.other_line_segments):
                start = np.array(seg[0:3])
                end = np.array(seg[3:6])

                # If segment has a 7th element, treat that as radius
                if len(seg) >= 7:
                    radius = seg[6]
                else:
                    radius = 0.01  # Fallback radius

                # Compute cylinder parameters:
                direction = end - start
                height = np.linalg.norm(direction)
                # If start and end coincide, skip plotting or treat as a small segment
                if np.isclose(height, 0):
                    continue
                direction /= height  # Normalize for PyVista's cylinder

                center = 0.5 * (start + end)  # Midpoint in world space
                cylinder = pv.Cylinder(
                    center=center,
                    direction=direction,
                    radius=radius,
                    height=height,
                    resolution=24  # number of radial subdivisions (smoothness)
                )

                plotter.add_mesh(cylinder, color='red', label=f'Segment {i + 1}')

        # 6) Optional: add a legend if you want to show labels in the plot
        #    Note that PyVista only displays unique labels; duplicates are merged.
        plotter.add_legend()

        # 7) Show the interactive PyVista window
        plotter.show()

def test(curve_type="Bezier"):
    cow = examples.download_cow()
    segments = np.array(random_line_segments_in_mesh(cow,n_segments=2)).reshape((4,3))
    other_segments = np.array(random_line_segments_in_mesh(cow, n_segments=20)).reshape((20,6))
    vessels = np.zeros((20,7))
    vessels[:, 0:6] = other_segments
    vessels[:, -1] = 0.05
    conn = BaseConnection(segments[0],segments[1],segments[2],segments[3],0.05,0.05,domain=cow,curve_type=curve_type)
    conn.set_collision_vessels(vessels)
    return conn


def cylinders_collide_any(cylinders_A: np.ndarray, cylinders_B: np.ndarray) -> bool:
    """
    Checks if any pair of cylinders (one from `cylinders_A`, one from `cylinders_B`)
    intersects (collides).

    The inputs `cylinders_A` and `cylinders_B` are (N,7) and (M,7) arrays respectively, where:
      - columns 0..5: endpoints of the cylinder's central axis
      - column 6: radius

    Returns:
      True if at least one pair of cylinders is colliding; otherwise False.
    """
    # 1) Extract just the first 6 columns for the line segments
    segments_A = cylinders_A[:, :6]  # shape (N, 6)
    segments_B = cylinders_B[:, :6]  # shape (M, 6)
    # 2) Extract the radii
    radii_A = cylinders_A[:, 6]  # shape (N,)
    radii_B = cylinders_B[:, 6]  # shape (M,)
    # 3) Compute the pairwise distances between line segments (N x M).
    #    You should already have a function like `minimum_segment_distance(segA, segB)`
    #    that returns a (N,M) array of distances.
    dist_array = minimum_segment_distance(segments_A, segments_B)  # shape (N, M)
    # 4) Compute sum of radii for each pair.
    #    We can expand dims to broadcast properly: (N,1) + (1,M) => (N,M)
    sum_of_radii = radii_A[:, None] + radii_B[None, :]
    # 5) Compare distances vs. sum_of_radii. A collision occurs if distance < sum_of_radii.
    #    So if `np.any(dist < sum_of_radii)` is True => at least one collision.
    collisions_mask = dist_array < sum_of_radii
    return np.any(collisions_mask)

def cylinders_collide_any_naive(cylinders_A: np.ndarray, cylinders_B: np.ndarray):
    """
    Checks if any pair of cylinders (one from `cylinders_A`, one from `cylinders_B`)
    collides, by computing the minimal distance between each pair of line segments
    and comparing it to the sum of their radii.

    Parameters
    ----------
    cylinders_A : np.ndarray
        Shape (N,7). Columns [x1, y1, z1, x2, y2, z2, radius].
    cylinders_B : np.ndarray
        Shape (M,7). Columns [x1, y1, z1, x2, y2, z2, radius].

    Returns
    -------
    bool
        True if at least one pair collides; False otherwise.
    """
    # For each cylinder in A, for each cylinder in B:
    # 1) Get the line segments Aseg, Bseg
    # 2) Compute min distance between those line segments
    # 3) Compare to sum_of_radii
    # If distance < sum_of_radii => collision => return True.
    # If we exit loops with no collisions => return False.
    dists = []
    for a in cylinders_A:
        # A's endpoints
        #Ax1, Ay1, Az1, Ax2, Ay2, Az2, Aradius = a
        Ax1, Ay1, Az1, Ax2, Ay2, Az2 = a
        sub_dists = []
        for b in cylinders_B:
            # B's endpoints
            #Bx1, By1, Bz1, Bx2, By2, Bz2, Bradius = b
            Bx1, By1, Bz1, Bx2, By2, Bz2 = b
            # sum of radii
            #sum_radii = Aradius + Bradius
            # If segment distance < sum_radii => collision
            dist = segment_segment_distance_3d(
                Ax1, Ay1, Az1, Ax2, Ay2, Az2,
                Bx1, By1, Bz1, Bx2, By2, Bz2
            )
            #if dist < sum_radii:
            sub_dists.append(dist)
        dists.append(sub_dists)
    return np.array(dists)


def segment_segment_distance_3d(
    Ax1, Ay1, Az1, Ax2, Ay2, Az2,
    Bx1, By1, Bz1, Bx2, By2, Bz2
) -> float:
    """
    Computes the minimal Euclidean distance between two 3D line segments:
      Segment A: (Ax1,Ay1,Az1) -> (Ax2,Ay2,Az2)
      Segment B: (Bx1,By1,Bz1) -> (Bx2,By2,Bz2)

    Returns
    -------
    float
        The minimal distance between the segments in 3D space.
    """
    # Direction vectors
    ABx = Ax2 - Ax1
    ABy = Ay2 - Ay1
    ABz = Az2 - Az1
    CDx = Bx2 - Bx1
    CDy = By2 - By1
    CDz = Bz2 - Bz1
    # Vector between first endpoints
    # CA = A1 - B1
    CAx = Ax1 - Bx1
    CAy = Ay1 - By1
    CAz = Az1 - Bz1
    # Dot products
    AB_AB = ABx*ABx + ABy*ABy + ABz*ABz    # |AB|^2
    CD_CD = CDx*CDx + CDy*CDy + CDz*CDz    # |CD|^2
    AB_CD = ABx*CDx + ABy*CDy + ABz*CDz    # AB . CD
    CA_AB = CAx*ABx + CAy*ABy + CAz*ABz    # CA . AB
    CA_CD = CAx*CDx + CAy*CDy + CAz*CDz    # CA . CD
    EPS = 1e-14
    # Degenerate checks
    if AB_AB < EPS and CD_CD < EPS:
        # Both segments are points
        dx = Ax1 - Bx1
        dy = Ay1 - By1
        dz = Az1 - Bz1
        return np.sqrt(dx*dx + dy*dy + dz*dz)
    elif AB_AB < EPS:
        # Segment A is a point, distance from that point to segment B
        return point_segment_distance_3d(Ax1, Ay1, Az1, Bx1, By1, Bz1, Bx2, By2, Bz2)
    elif CD_CD < EPS:
        # Segment B is a point
        return point_segment_distance_3d(Bx1, By1, Bz1, Ax1, Ay1, Az1, Ax2, Ay2, Az2)
    # If both are non-degenerate
    denominator = AB_AB * CD_CD - AB_CD * AB_CD
    if abs(denominator) < EPS:
        # The segments are parallel (or nearly so).
        # We can use a fallback approach: check each endpoint vs. the other segment
        # and pick the min among the four point-seg distances.
        dA1 = point_segment_distance_3d(Ax1, Ay1, Az1, Bx1, By1, Bz1, Bx2, By2, Bz2)
        dA2 = point_segment_distance_3d(Ax2, Ay2, Az2, Bx1, By1, Bz1, Bx2, By2, Bz2)
        dB1 = point_segment_distance_3d(Bx1, By1, Bz1, Ax1, Ay1, Az1, Ax2, Ay2, Az2)
        dB2 = point_segment_distance_3d(Bx2, By2, Bz2, Ax1, Ay1, Az1, Ax2, Ay2, Az2)
        return min(dA1, dA2, dB1, dB2)
    else:
        # Skew or intersecting lines, use the param formula
        t = (AB_CD*CA_CD - CA_AB*CD_CD) / denominator
        s = (AB_AB*CA_CD - AB_CD*CA_AB) / denominator
        # Clamp t, s within [0..1]
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        if s < 0.0:
            s = 0.0
        elif s > 1.0:
            s = 1.0
        # Closest points
        # P1 = A1 + t * AB
        P1x = Ax1 + t*ABx
        P1y = Ay1 + t*ABy
        P1z = Az1 + t*ABz
        # P2 = B1 + s * CD
        P2x = Bx1 + s*CDx
        P2y = By1 + s*CDy
        P2z = Bz1 + s*CDz
        dx = P1x - P2x
        dy = P1y - P2y
        dz = P1z - P2z
        return np.sqrt(dx*dx + dy*dy + dz*dz)


def point_segment_distance_3d(
    Px, Py, Pz,
    Sx1, Sy1, Sz1,
    Sx2, Sy2, Sz2
) -> float:
    """
    Distance from point (Px,Py,Pz) to the segment [Sx1,Sy1,Sz1 -> Sx2,Sy2,Sz2].
    """
    vx = Sx2 - Sx1
    vy = Sy2 - Sy1
    vz = Sz2 - Sz1
    wx = Px - Sx1
    wy = Py - Sy1
    wz = Pz - Sz1
    seg_len_sq = vx*vx + vy*vy + vz*vz
    if seg_len_sq < 1e-14:
        # Degenerate => just distance from (Sx1,Sy1,Sz1)
        dx = Px - Sx1
        dy = Py - Sy1
        dz = Pz - Sz1
        return np.sqrt(dx*dx + dy*dy + dz*dz)
    proj = (wx*vx + wy*vy + wz*vz) / seg_len_sq
    if proj < 0.0:
        proj = 0.0
    elif proj > 1.0:
        proj = 1.0
    cx = Sx1 + proj*vx
    cy = Sy1 + proj*vy
    cz = Sz1 + proj*vz
    dx = Px - cx
    dy = Py - cy
    dz = Pz - cz
    return np.sqrt(dx*dx + dy*dy + dz*dz)


def create_cylinder_mesh(p0: np.ndarray, p1: np.ndarray, radius: float) -> pv.PolyData:
    """
    Create a PyVista mesh (PolyData) of a cylinder whose centerline
    goes from p0 to p1, with the specified radius.
    If the segment length is near zero, return a sphere as a fallback.
    """
    # Direction vector
    v = p1 - p0
    length = np.linalg.norm(v)

    # Handle degenerate segment => fallback as a sphere at p0
    if length < 1e-12:
        return pv.Sphere(radius=radius, center=p0)

    direction = v / length
    midpoint = 0.5 * (p0 + p1)

    cylinder = pv.Cylinder(
        center=midpoint,
        direction=direction,
        height=length,
        radius=radius
    )
    return cylinder


def batch_cylinder_intersection(data_0: np.ndarray,
                                data_1: np.ndarray) -> np.ndarray:
    """
    Build cylinder meshes for each row in data_0 and data_1,
    and check pairwise intersections via boolean operations.

    Args:
        data_0, data_1: Arrays of shape (N,7) and (M,7).
                        Each row is (x0, y0, z0, x1, y1, z1, radius).

    Returns:
        A boolean array `intersect[N, M]` where True means cylinders intersect.
    """
    n = data_0.shape[0]
    m = data_1.shape[0]

    # Pre-build the cylinder meshes for data_0
    cyls_0 = []
    for i in range(n):
        x0, y0, z0, x1, y1, z1, r0 = data_0[i]
        cyls_0.append(create_cylinder_mesh(
            np.array([x0, y0, z0]),
            np.array([x1, y1, z1]),
            r0
        ))

    # Pre-build the cylinder meshes for data_1
    cyls_1 = []
    for j in range(m):
        x0, y0, z0, x1, y1, z1, r1 = data_1[j]
        cyls_1.append(create_cylinder_mesh(
            np.array([x0, y0, z0]),
            np.array([x1, y1, z1]),
            r1
        ))

    # Prepare the output matrix
    intersect = np.zeros((n, m), dtype=bool)

    # Pairwise boolean intersection check
    for i in range(n):
        for j in range(m):
            # Attempt a boolean cut or intersection
            # If the result has any cells, they intersect
            res = cyls_0[i].boolean_cut(cyls_1[j])
            intersect[i, j] = (res.n_cells > 0)

    return intersect
