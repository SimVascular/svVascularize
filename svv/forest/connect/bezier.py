from abc import ABC

import numpy as np
from svv.forest.connect.base import BaseCurve


class BezierCurve(BaseCurve, ABC):
    """
    A concrete implementation of Curve for Bézier curves in 2D or 3D space.

    Attributes
    ----------
    control_points : np.ndarray
        Array of shape (n+1, d), where n is the polynomial degree of the Bézier curve,
        and d = 2 or 3 for 2D or 3D curves.
    """

    def __init__(self, control_points):
        """
        Initialize the BezierCurve with control points.

        Parameters
        ----------
        control_points : array-like
            A 2D or 3D array/list of shape (n+1, d), where n+1 is the number
            of control points, and d is either 2 or 3.
        """
        self.control_points = np.array(control_points, dtype=float)
        if self.control_points.ndim != 2:
            raise ValueError("control_points must be a 2D array of shape (n+1, d).")

    def _de_casteljau(self, points, t):
        """
        Compute a single point on the Bézier curve using De Casteljau's algorithm.

        Parameters
        ----------
        points : np.ndarray
            Control points (or derivative control points) to be interpolated,
            shape (m, d).
        t : float
            Parametric value in the range [0, 1].

        Returns
        -------
        np.ndarray
            A single point on the Bézier curve at the given parametric value,
            shape (d,).
        """
        pts = points.copy()
        while len(pts) > 1:
            pts = (1 - t) * pts[:-1] + t * pts[1:]
        return pts[0]

    def evaluate(self, t_values):
        """
        Evaluate the Bézier curve at the given array of parametric values.

        Uses vectorized De Casteljau's algorithm to evaluate all points at once.

        Parameters
        ----------
        t_values : array-like
            Parametric values in the range [0, 1].

        Returns
        -------
        np.ndarray
            Points on the Bézier curve at the specified parametric values.
            Shape: (len(t_values), d).
        """
        t_values = np.atleast_1d(t_values)
        # Vectorized De Casteljau: process all t values simultaneously
        # pts shape: (n_ctrl_pts, len(t_values), d)
        pts = np.broadcast_to(
            self.control_points[:, None, :],
            (self.control_points.shape[0], len(t_values), self.control_points.shape[1])
        ).copy()
        t = t_values[None, :, None]  # (1, len(t_values), 1)
        n = pts.shape[0]
        for i in range(n - 1):
            pts[:n - 1 - i] = (1 - t) * pts[:n - 1 - i] + t * pts[1:n - i]
        return pts[0]  # (len(t_values), d)

    def derivative(self, t_values, order=1):
        """
        Compute the nth derivative of the Bézier curve at specified parametric values.

        Uses vectorized evaluation for all t values simultaneously.

        Parameters
        ----------
        t_values : array-like
            Parametric values in the range [0, 1].
        order : int, optional
            The order of the derivative (1 for first derivative, 2 for second, etc.).

        Returns
        -------
        np.ndarray
            The derivative values at each parametric value, shape (len(t_values), d).
        """
        n = len(self.control_points) - 1
        derivative_points = self.control_points.copy()

        # Iteratively compute up to the requested derivative order
        for _ in range(order):
            derivative_points = n * (derivative_points[1:] - derivative_points[:-1])
            n -= 1
            if n < 0:
                derivative_points = np.zeros((1, derivative_points.shape[1]))
                break

        t_values = np.atleast_1d(t_values)
        # Vectorized De Casteljau on derivative control points
        m = derivative_points.shape[0]
        if m == 1:
            return np.broadcast_to(derivative_points[0], (len(t_values), derivative_points.shape[1])).copy()
        pts = np.broadcast_to(
            derivative_points[:, None, :],
            (m, len(t_values), derivative_points.shape[1])
        ).copy()
        t = t_values[None, :, None]
        for i in range(m - 1):
            pts[:m - 1 - i] = (1 - t) * pts[:m - 1 - i] + t * pts[1:m - i]
        return pts[0]

    def roc(self, t_values):
        """
        Compute the radius of curvature of the Bézier curve at specified parametric values.

        Parameters
        ----------
        t_values : array-like
            Parametric values in the range [0, 1].

        Returns
        -------
        np.ndarray
            Radius of curvature values along the Bézier curve.
            Shape: (len(t_values),).

        Notes
        -----
        Formula: R = |v|^3 / |v x a|, where
            - v is the first derivative (velocity),
            - a is the second derivative (acceleration).
        """
        t_values = np.atleast_1d(t_values)
        v = self.derivative(t_values, order=1)  # shape (N, d)
        a = self.derivative(t_values, order=2)  # shape (N, d)

        v_mags = np.linalg.norm(v, axis=1)

        if self.dimension == 2:
            # Cross product in 2D is a scalar: v_x a_y - v_y a_x
            cross_va = v[:, 0] * a[:, 1] - v[:, 1] * a[:, 0]
            cross_mags = np.abs(cross_va)
        else:
            # 3D cross product => shape (N, 3)
            cross_va = np.cross(v, a)
            cross_mags = np.linalg.norm(cross_va, axis=1)

        epsilon = 1e-12
        roc_values = v_mags ** 3 / (cross_mags + epsilon)
        return roc_values

    def torsion(self, t_values):
        """
        Compute the torsion of the Bézier curve at specified parametric values.

        Parameters
        ----------
        t_values : array-like
            Parametric values in the range [0, 1].

        Returns
        -------
        np.ndarray
            Torsion values at each parametric value.
            Shape: (len(t_values),).

        Notes
        -----
        Torsion is generally meaningful in 3D. In 2D, it's typically zero.
        Formula: τ = ((v x a) · b) / |v x a|^2, where
            - v: first derivative
            - a: second derivative
            - b: third derivative
        """
        t_values = np.atleast_1d(t_values)

        # If 2D, short-circuit to zero
        if self.dimension == 2:
            return np.zeros_like(t_values, dtype=float)

        # 3D case
        v = self.derivative(t_values, order=1)
        a = self.derivative(t_values, order=2)
        b = self.derivative(t_values, order=3)

        cross_va = np.cross(v, a)  # shape (N,3)
        numerator = np.einsum('ij,ij->i', cross_va, b)  # dot((v x a), b)
        denominator = np.einsum('ij,ij->i', cross_va, cross_va)  # |v x a|^2

        with np.errstate(divide='ignore', invalid='ignore'):
            torsion_values = numerator / denominator
            torsion_values = np.nan_to_num(torsion_values)

        return torsion_values

    def arc_length(self, t_start=0, t_end=1, num_points=100):
        """
        Compute the approximate arc length of the Bézier curve between two parametric values.

        Parameters
        ----------
        t_start : float, optional
            Start parameter in [0, 1]. Default is 0.
        t_end : float, optional
            End parameter in [0, 1]. Default is 1.
        num_points : int, optional
            Number of sample points for piecewise linear approximation. Default is 100.

        Returns
        -------
        float
            Approximate arc length over [t_start, t_end].
        """
        if not (0 <= t_start < t_end <= 1):
            raise ValueError("t_start and t_end must satisfy 0 <= t_start < t_end <= 1.")

        # Sample parametric values
        t_values = np.linspace(t_start, t_end, num_points)

        # Evaluate points on the curve
        points = self.evaluate(t_values)

        # Sum distances between consecutive sample points
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        arc_len = np.sum(segment_lengths)

        return arc_len

    @property
    def dimension(self):
        return self.control_points.shape[1]
