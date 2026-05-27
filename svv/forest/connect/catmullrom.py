import numpy as np
from abc import ABC

from svv.forest.connect.base import BaseCurve


class CatmullRomCurve(BaseCurve, ABC):
    """
    A concrete implementation of a Catmull–Rom spline inheriting from the Curve base class.

    This class interpolates a sequence of control points with a C^1-continuous spline.
    The param domain for the base class is [0,1], mapped to piecewise segments internally.

    Parameters
    ----------
    control_points : (N, d) array-like
        Sequence of points to interpolate. d can be 2 or 3 (for 2D/3D).
    closed : bool, optional
        If True, treat the curve as closed (the last point connects to the first).
        Defaults to False (an open spline).
    """

    def __init__(self, control_points, closed=False):
        self.control_points = np.array(control_points, dtype=float)
        if self.control_points.ndim != 2:
            raise ValueError("control_points must be of shape (N, d).")

        self._closed = closed
        self._num_pts = len(self.control_points)
        if self._num_pts < 2:
            raise ValueError("Catmull–Rom needs at least 2 points.")

    @property
    def dimension(self):
        """Return the dimension (2 or 3) of the control points."""
        return self.control_points.shape[1]

    def evaluate(self, t_values):
        """
        Evaluate the Catmull–Rom spline at param values t in [0,1].

        Uses vectorized operations to evaluate all t values at once per segment.

        Parameters
        ----------
        t_values : float or array-like
            Parametric values in [0, 1].

        Returns
        -------
        np.ndarray
            Array of shape (len(t_values), dimension) with the evaluated points.
        """
        t_values = np.atleast_1d(t_values).astype(float)

        # Number of segments
        n_segs = self._num_pts if self._closed else (self._num_pts - 1)
        if n_segs < 1:
            raise ValueError("Not enough segments to evaluate.")

        # Clamp and map to segment space
        t_clamped = np.clip(t_values, 0.0, 1.0)
        scaled = t_clamped * n_segs
        seg_idx = np.floor(scaled).astype(int)
        seg_idx = np.clip(seg_idx, 0, n_segs - 1)
        local_u = scaled - seg_idx

        # Process each unique segment to batch evaluations
        out = np.empty((len(t_values), self.dimension))
        unique_segs = np.unique(seg_idx)

        for s in unique_segs:
            mask = seg_idx == s
            u_vals = local_u[mask]

            p0 = self._get_ctrl_point(s - 1)
            p1 = self._get_ctrl_point(s)
            p2 = self._get_ctrl_point(s + 1)
            p3 = self._get_ctrl_point(s + 2)

            # Vectorized Catmull-Rom evaluation for all u values in this segment
            u2 = u_vals * u_vals
            u3 = u2 * u_vals
            a = -p0 + p2
            b = 2 * p0 - 5 * p1 + 4 * p2 - p3
            c = -p0 + 3 * p1 - 3 * p2 + p3
            out[mask] = 0.5 * (2 * p1 + np.outer(u_vals, a) + np.outer(u2, b) + np.outer(u3, c))

        return out

    def derivative(self, t_values, order=1):
        """
        Compute the nth derivative of the Catmull–Rom spline at t in [0,1].

        For a piecewise cubic:
          - 1st derivative => quadratic polynomial
          - 2nd derivative => linear polynomial
          - 3rd derivative => constant
          - 4th derivative => 0

        We map t => segment local parameter u and use chain rule:
          d^k P / dt^k = (num_segments)^k * d^k P / du^k
        """
        if order < 1:
            raise ValueError("derivative order must be >= 1")

        t_values = np.atleast_1d(t_values).astype(float)
        n_segs = self._num_pts if self._closed else (self._num_pts - 1)
        if n_segs < 1:
            raise ValueError("Not enough segments to evaluate derivative.")

        out = np.zeros((len(t_values), self.dimension))
        for idx, t in enumerate(t_values):
            # Clamp t => [0,1]
            t = max(0.0, min(1.0, t))

            scaled = t * n_segs
            i = int(np.floor(scaled))
            if i == n_segs:
                i = n_segs - 1
            u = scaled - i

            p0 = self._get_ctrl_point(i - 1)
            p1 = self._get_ctrl_point(i)
            p2 = self._get_ctrl_point(i + 1)
            p3 = self._get_ctrl_point(i + 2)

            # derivative wrt u
            dP_du = self._catmull_rom_segment_derivative_order(p0, p1, p2, p3, u, order)
            # chain rule => (d^k P / dt^k) = (num_segments)^k * (d^k P / du^k)
            factor = (n_segs ** order)
            out[idx] = dP_du * factor

        return out

    def roc(self, t_values):
        """
        Compute the radius of curvature, ROC(t) = |v|^3 / |v x a|.
        - v = first derivative wrt t
        - a = second derivative wrt t
        """
        t_values = np.atleast_1d(t_values)

        # First derivative (velocity) and second derivative (acceleration)
        v = self.derivative(t_values, order=1)  # shape: (N, d)
        a = self.derivative(t_values, order=2)  # shape: (N, d)

        # Magnitudes of v
        v_mags = np.linalg.norm(v, axis=1)

        if self.dimension == 2:
            # 2D "cross product" => scalar z-component: (v_x, v_y) x (a_x, a_y) = v_x*a_y - v_y*a_x
            cross_vals = v[:, 0] * a[:, 1] - v[:, 1] * a[:, 0]
            cross_mags = np.abs(cross_vals)
        elif self.dimension == 3:
            # 3D cross => shape (N,3)
            cross_vecs = np.cross(v, a)
            cross_mags = np.linalg.norm(cross_vecs, axis=1)
        else:
            raise NotImplementedError("Curvature only implemented for 2D or 3D Catmull–Rom.")

        epsilon = 1e-12
        roc_vals = (v_mags**3) / (cross_mags + epsilon)
        return roc_vals

    def torsion(self, t_values):
        """
        Torsion: τ(t) = ((v x a) · b) / |v x a|^2
          where:
            v = first derivative
            a = second derivative
            b = third derivative

        Returns a 1D array of torsion values for each t.
        """
        if self.dimension == 2:
            # In 2D, torsion is typically zero (curve is planar).
            t_values = np.atleast_1d(t_values)
            return np.zeros_like(t_values, dtype=float)

        if self.dimension != 3:
            raise NotImplementedError("Torsion only implemented for 2D (returns 0) or 3D Catmull–Rom.")

        t_values = np.atleast_1d(t_values)

        # First, second, and third derivatives
        v = self.derivative(t_values, order=1)  # shape (N,3)
        a = self.derivative(t_values, order=2)  # shape (N,3)
        b = self.derivative(t_values, order=3)  # shape (N,3)

        # cross_va => shape (N,3)
        cross_va = np.cross(v, a)
        # Numerator => (cross(v,a) · b)
        numerator = np.einsum('ij,ij->i', cross_va, b)  # dot product row-wise

        # Denominator => |v x a|^2
        denominator = np.einsum('ij,ij->i', cross_va, cross_va)

        epsilon = 1e-12
        torsion_vals = numerator / (denominator + epsilon)

        return torsion_vals

    def arc_length(self, t_start=0.0, t_end=1.0, num_points=100):
        """
        Approximate arc length over [t_start, t_end] using piecewise linear approximation.

        Parameters
        ----------
        t_start : float
            Start parameter in [0, 1].
        t_end : float
            End parameter in [0, 1].
        num_points : int
            Number of sample points for piecewise approximation.

        Returns
        -------
        float
            Approximate arc length of the Catmull-Rom spline from t_start to t_end.
        """
        if t_start < 0:
            t_start = 0
        if t_end > 1:
            t_end = 1
        if t_end < t_start:
            raise ValueError("t_end must be >= t_start in arc_length.")

        # Sample parametric values
        t_values = np.linspace(t_start, t_end, num_points)
        points = self.evaluate(t_values)  # shape => (num_points, dimension)

        # Sum distances between consecutive sample points
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        return np.sum(segment_lengths)

    def transform(self, matrix):
        """
        Apply a homogeneous transformation matrix to the control points.

        Parameters
        ----------
        matrix : np.ndarray
            - If dimension=2, a 3x3 homogeneous transform is expected.
            - If dimension=3, a 4x4 homogeneous transform is expected.

        Raises
        ------
        ValueError
            If the matrix shape doesn't match the expected (3,3) or (4,4).
        """
        d = self.dimension

        if d == 2:
            if matrix.shape != (3, 3):
                raise ValueError(
                    "For a 2D Catmull–Rom spline, transform must be a 3x3 homogeneous matrix."
                )
            # Convert (N,2) -> (N,3) by appending a column of 1s
            ones = np.ones((self._num_pts, 1), dtype=float)
            homogeneous_pts = np.hstack([self.control_points, ones])  # shape (N,3)

            # Apply transform: new_pts = (N,3) @ (3,3).T => shape (N,3)
            # or use matrix @ homogeneous_pts.T, but let's keep row-major convention
            transformed = homogeneous_pts @ matrix.T

            # Drop the last column to get back to R^2
            self.control_points = transformed[:, :2]

        elif d == 3:
            if matrix.shape != (4, 4):
                raise ValueError(
                    "For a 3D Catmull–Rom spline, transform must be a 4x4 homogeneous matrix."
                )
            # Convert (N,3) -> (N,4)
            ones = np.ones((self._num_pts, 1), dtype=float)
            homogeneous_pts = np.hstack([self.control_points, ones])  # shape (N,4)

            # Apply transform
            transformed = homogeneous_pts @ matrix.T

            # Drop the last coordinate to get back to R^3
            self.control_points = transformed[:, :3]

        else:
            raise NotImplementedError(
                f"Transform not implemented for dimension={d}. Only 2D or 3D supported."
            )

    def _get_ctrl_point(self, i):
        """
        Fetch control point i with boundary conditions.
        If closed, wrap around. If open, clamp at ends.
        """
        if self._closed:
            # Wrap index modulo _num_pts
            return self.control_points[i % self._num_pts]
        else:
            # Clamp index to [0, _num_pts-1]
            i_clamped = max(0, min(self._num_pts - 1, i))
            return self.control_points[i_clamped]

    @staticmethod
    def _catmull_rom_segment(p0, p1, p2, p3, u):
        """
        Compute a point on the Catmull–Rom segment for local parameter u in [0,1].
        Uses the standard Catmull–Rom blending with tension = 0.5.

        The formula (with alpha=0.5) is often written as:
          P(u) = 0.5 * [ 2*P1
                         + (-P0 + P2)*u
                         + ( 2P0 - 5P1 + 4P2 - P3)*u^2
                         + (-P0 + 3P1 - 3P2 + P3)*u^3 ]
        """
        # Convert to 0.5 factor for clarity
        u2 = u * u
        u3 = u2 * u

        # Coefficients for the cubic polynomial
        a = -p0 + p2
        b = 2 * p0 - 5 * p1 + 4 * p2 - p3
        c = -p0 + 3 * p1 - 3 * p2 + p3

        return 0.5 * (
                2 * p1
                + a * u
                + b * u2
                + c * u3
        )

    @staticmethod
    def _catmull_rom_segment_derivative_order(p0, p1, p2, p3, u, order):
        """
        Return the `order`-th derivative w.r.t. u of the Catmull–Rom segment.

        For the cubic polynomial:
          P(u)   = 0.5 ( alpha0 + alpha1 u + alpha2 u^2 + alpha3 u^3 )
        => P'(u) = 0.5 ( alpha1 + 2 alpha2 u + 3 alpha3 u^2 )
        => P''(u)= 0.5 ( 2 alpha2 + 6 alpha3 u )
        => P'''(u)= 0.5 ( 6 alpha3 )    # constant
        => P''''(u)= 0

        So we implement up to the 3rd derivative. For order>3 => zero vector.
        """
        # Precompute polynomial coefficients alpha0..alpha3
        # Here, alpha0 = 2 p1
        #       alpha1 = -p0 + p2
        #       alpha2 = 2p0 - 5p1 + 4p2 - p3
        #       alpha3 = -p0 + 3p1 - 3p2 + p3
        a = -p0 + p2
        b = 2*p0 - 5*p1 + 4*p2 - p3
        c = -p0 + 3*p1 - 3*p2 + p3
        # alpha0 = 2 p1
        # alpha1 = a
        # alpha2 = b
        # alpha3 = c

        # For convenience, define them as vectors
        alpha0 = 2 * p1
        alpha1 = a
        alpha2 = b
        alpha3 = c

        # factor = 0.5 for all
        half = 0.5

        if order == 1:
            # P'(u) = 0.5 [ alpha1 + 2 alpha2 u + 3 alpha3 u^2 ]
            u2 = u * u
            return half * (alpha1 + 2*alpha2*u + 3*alpha3*u2)
        elif order == 2:
            # P''(u) = 0.5 [ 2 alpha2 + 6 alpha3 u ]
            return half * (2*alpha2 + 6*alpha3*u)
        elif order == 3:
            # P'''(u) = 0.5 [ 6 alpha3 ]
            return half * (6*alpha3)
        else:
            # 4th derivative and above => 0
            return np.zeros_like(p0)  # shape=(dimension,)
