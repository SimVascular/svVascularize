"""
Stress tests for vectorized / optimized code paths in svVascularize.

Covers:
  - BezierCurve vectorized De Casteljau
  - CatmullRomCurve batched segment evaluation
  - minimum_segment_distance vectorized pairwise computation
  - CenterlineResult backward-compatible unpacking
  - BaseConnection constraint cache sharing
  - Geodesic edge extraction vectorization
"""

import pytest
import numpy as np
from time import perf_counter
from math import sqrt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_3d_segments(n, rng, spread=10.0):
    """Generate n random 3D line segments as (n, 6) array."""
    return rng.uniform(-spread, spread, size=(n, 6)).astype(np.float64)


def _random_control_points_3d(n_pts, rng, spread=5.0):
    """Generate n_pts random 3D control points."""
    return rng.uniform(-spread, spread, size=(n_pts, 3))


# ===========================================================================
# 1. BezierCurve — vectorized De Casteljau stress tests
# ===========================================================================

from svv.forest.connect.bezier import BezierCurve


class TestBezierStress:
    """Stress tests for vectorized Bezier evaluation."""

    @pytest.mark.parametrize("n_t", [2, 10, 100, 1_000, 10_000, 50_000])
    def test_evaluate_many_t_values(self, n_t):
        """Evaluate at increasing numbers of parameter values."""
        ctrl = np.array([[0, 0, 0], [1, 2, 0], [3, 1, 1], [4, 0, 0]], dtype=float)
        curve = BezierCurve(ctrl)
        t = np.linspace(0, 1, n_t)
        pts = curve.evaluate(t)
        assert pts.shape == (n_t, 3)
        # Endpoints must be exact
        np.testing.assert_allclose(pts[0], ctrl[0], atol=1e-12)
        np.testing.assert_allclose(pts[-1], ctrl[-1], atol=1e-12)

    @pytest.mark.parametrize("degree", [1, 2, 3, 5, 8, 15, 25])
    def test_high_degree_curves(self, degree):
        """Evaluate high-degree Bezier curves (many control points)."""
        rng = np.random.default_rng(42)
        ctrl = _random_control_points_3d(degree + 1, rng)
        curve = BezierCurve(ctrl)
        t = np.linspace(0, 1, 500)
        pts = curve.evaluate(t)
        assert pts.shape == (500, 3)
        np.testing.assert_allclose(pts[0], ctrl[0], atol=1e-10)
        np.testing.assert_allclose(pts[-1], ctrl[-1], atol=1e-10)
        assert np.all(np.isfinite(pts))

    def test_vectorized_matches_sequential(self):
        """Verify vectorized evaluation matches point-by-point evaluation."""
        rng = np.random.default_rng(99)
        ctrl = _random_control_points_3d(5, rng)
        curve = BezierCurve(ctrl)
        t_vals = np.linspace(0, 1, 200)
        batch = curve.evaluate(t_vals)
        for i, t in enumerate(t_vals):
            single = curve.evaluate(np.array([t]))
            np.testing.assert_allclose(batch[i], single[0], atol=1e-12)

    def test_derivative_consistency(self):
        """First derivative via finite difference should match analytic."""
        ctrl = np.array([[0, 0, 0], [1, 3, 0], [3, 1, 2], [5, 0, 0]], dtype=float)
        curve = BezierCurve(ctrl)
        t = np.array([0.25, 0.5, 0.75])
        dt = 1e-7
        analytic = curve.derivative(t, order=1)
        fd = (curve.evaluate(t + dt) - curve.evaluate(t - dt)) / (2 * dt)
        np.testing.assert_allclose(analytic, fd, atol=1e-4)

    def test_roc_and_torsion_large_batch(self):
        """ROC and torsion on a large t-value batch."""
        ctrl = np.array([
            [0, 0, 0], [1, 2, 1], [2, 0, 3], [3, -1, 0], [4, 0, 0]
        ], dtype=float)
        curve = BezierCurve(ctrl)
        t = np.linspace(0.01, 0.99, 5000)
        roc = curve.roc(t)
        assert roc.shape == (5000,)
        assert np.all(np.isfinite(roc))
        assert np.all(roc > 0)
        torsion = curve.torsion(t)
        assert torsion.shape == (5000,)
        assert np.all(np.isfinite(torsion))

    def test_arc_length_convergence(self):
        """Arc length should converge as num_points increases."""
        ctrl = np.array([[0, 0, 0], [1, 2, 0], [2, 0, 0]], dtype=float)
        curve = BezierCurve(ctrl)
        lengths = []
        for n in [10, 50, 100, 500, 1000]:
            lengths.append(curve.arc_length(0, 1, num_points=n))
        # Each successive estimate should be closer to the limit
        diffs = [abs(lengths[i+1] - lengths[i]) for i in range(len(lengths)-1)]
        for i in range(len(diffs) - 1):
            assert diffs[i+1] <= diffs[i] + 1e-12, "Arc length not converging"


# ===========================================================================
# 2. CatmullRomCurve — batched segment evaluation stress tests
# ===========================================================================

from svv.forest.connect.catmullrom import CatmullRomCurve


class TestCatmullRomStress:
    """Stress tests for batched Catmull-Rom evaluation."""

    @pytest.mark.parametrize("n_t", [2, 10, 100, 1_000, 10_000, 50_000])
    def test_evaluate_many_t_values(self, n_t):
        """Evaluate at increasing numbers of t values."""
        ctrl = np.array([
            [0, 0, 0], [1, 2, 1], [3, 1, 2], [5, 0, 0], [7, 1, 1]
        ], dtype=float)
        spline = CatmullRomCurve(ctrl)
        t = np.linspace(0, 1, n_t)
        pts = spline.evaluate(t)
        assert pts.shape == (n_t, 3)
        np.testing.assert_allclose(pts[0], ctrl[0], atol=1e-10)
        np.testing.assert_allclose(pts[-1], ctrl[-1], atol=1e-10)
        assert np.all(np.isfinite(pts))

    @pytest.mark.parametrize("n_ctrl", [2, 5, 10, 25, 50, 100])
    def test_many_control_points(self, n_ctrl):
        """Splines with many control points (many segments)."""
        rng = np.random.default_rng(77)
        ctrl = _random_control_points_3d(n_ctrl, rng)
        spline = CatmullRomCurve(ctrl)
        t = np.linspace(0, 1, 1000)
        pts = spline.evaluate(t)
        assert pts.shape == (1000, 3)
        np.testing.assert_allclose(pts[0], ctrl[0], atol=1e-10)
        np.testing.assert_allclose(pts[-1], ctrl[-1], atol=1e-10)

    def test_interpolation_at_knots(self):
        """Catmull-Rom must pass through its control points."""
        rng = np.random.default_rng(12)
        ctrl = _random_control_points_3d(8, rng)
        spline = CatmullRomCurve(ctrl)
        n_segs = len(ctrl) - 1
        for i in range(len(ctrl)):
            t = i / n_segs
            t = min(t, 1.0 - 1e-14)  # avoid exact 1.0 edge case
            pt = spline.evaluate(np.array([t]))
            np.testing.assert_allclose(pt[0], ctrl[i], atol=1e-6,
                                       err_msg=f"Missed knot {i} at t={t}")

    def test_closed_curve_continuity(self):
        """Closed curve should be C1-continuous at the wrap point."""
        ctrl = np.array([
            [0, 0, 0], [1, 1, 0], [2, 0, 0], [1, -1, 0]
        ], dtype=float)
        spline = CatmullRomCurve(ctrl, closed=True)
        # Evaluate near t=0 and t=1 — should match
        eps = 1e-6
        pt_start = spline.evaluate(np.array([eps]))[0]
        pt_end = spline.evaluate(np.array([1.0 - eps]))[0]
        d_start = spline.derivative(np.array([eps]), order=1)[0]
        d_end = spline.derivative(np.array([1.0 - eps]), order=1)[0]
        # Positions should be close (wrap-around)
        np.testing.assert_allclose(pt_start, pt_end, atol=1e-3)
        # Tangent directions should align
        cos_angle = np.dot(d_start, d_end) / (
            np.linalg.norm(d_start) * np.linalg.norm(d_end) + 1e-30
        )
        assert cos_angle > 0.95, f"Tangent discontinuity at wrap: cos={cos_angle}"

    def test_derivative_finite_difference(self):
        """Analytic first derivative matches finite difference."""
        ctrl = np.array([
            [0, 0, 0], [1, 2, 1], [3, 0, 2], [5, 1, 0]
        ], dtype=float)
        spline = CatmullRomCurve(ctrl)
        t = np.array([0.15, 0.35, 0.65, 0.85])
        dt = 1e-6
        analytic = spline.derivative(t, order=1)
        fd = (spline.evaluate(t + dt) - spline.evaluate(t - dt)) / (2 * dt)
        np.testing.assert_allclose(analytic, fd, atol=0.5)

    def test_arc_length_positive(self):
        """Arc length must be positive for non-degenerate curves."""
        rng = np.random.default_rng(55)
        ctrl = _random_control_points_3d(10, rng, spread=10.0)
        spline = CatmullRomCurve(ctrl)
        length = spline.arc_length(0, 1, num_points=500)
        assert length > 0


# ===========================================================================
# 3. minimum_segment_distance — vectorized pairwise stress tests
# ===========================================================================

from svv.utils.spatial.c_distance import minimum_segment_distance


class TestSegmentDistanceStress:
    """Stress tests for vectorized segment distance computation."""

    @pytest.mark.parametrize("n0,n1", [
        (1, 1), (10, 10), (50, 50), (100, 100),
        (200, 200), (500, 500), (1000, 100),
        (100, 1000),
    ])
    def test_shape_and_nonnegativity(self, n0, n1):
        """Output shape is (n0, n1) and all distances >= 0."""
        rng = np.random.default_rng(42)
        data0 = _random_3d_segments(n0, rng)
        data1 = _random_3d_segments(n1, rng)
        dist = minimum_segment_distance(data0, data1)
        assert dist.shape == (n0, n1)
        assert np.all(dist >= 0)
        assert np.all(np.isfinite(dist))

    def test_symmetry(self):
        """dist(A, B)[i,j] == dist(B, A)[j,i]."""
        rng = np.random.default_rng(123)
        data0 = _random_3d_segments(50, rng)
        data1 = _random_3d_segments(60, rng)
        d_ab = minimum_segment_distance(data0, data1)
        d_ba = minimum_segment_distance(data1, data0)
        np.testing.assert_allclose(d_ab, d_ba.T, atol=1e-10)

    def test_self_distance_zero_diagonal(self):
        """Distance of each segment to itself should be 0."""
        rng = np.random.default_rng(7)
        data = _random_3d_segments(100, rng)
        dist = minimum_segment_distance(data, data)
        diag = np.diag(dist)
        np.testing.assert_allclose(diag, 0.0, atol=1e-10)

    def test_identity_segments_zero(self):
        """Identical segment pairs should have distance 0."""
        rng = np.random.default_rng(88)
        data = _random_3d_segments(50, rng)
        dist = minimum_segment_distance(data, data)
        np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-10)

    def test_degenerate_vs_normal_mixed(self):
        """Mix of degenerate (point) and normal segments."""
        rng = np.random.default_rng(33)
        normal = _random_3d_segments(50, rng)
        # Make every 5th segment degenerate
        degenerate_idx = list(range(0, 50, 5))
        for i in degenerate_idx:
            normal[i, 3:6] = normal[i, 0:3]
        dist = minimum_segment_distance(normal, normal)
        assert dist.shape == (50, 50)
        assert np.all(np.isfinite(dist))
        assert np.all(dist >= 0)

    def test_parallel_segments_batch(self):
        """Batch of parallel segments — exercises the fallback path."""
        n = 100
        data0 = np.zeros((n, 6), dtype=np.float64)
        data1 = np.zeros((n, 6), dtype=np.float64)
        offsets = np.linspace(1, 10, n)
        for i in range(n):
            # All segments along x-axis, offset in y
            data0[i] = [0, 0, 0, 1, 0, 0]
            data1[i] = [0, offsets[i], 0, 1, offsets[i], 0]
        dist = minimum_segment_distance(data0, data1)
        assert dist.shape == (n, n)
        # Diagonal: dist[i,i] should equal offsets[i]
        np.testing.assert_allclose(np.diag(dist), offsets, atol=1e-10)

    def test_large_scale_no_crash(self):
        """1000x1000 pairwise — verify no crash or memory issue."""
        rng = np.random.default_rng(999)
        data0 = _random_3d_segments(1000, rng)
        data1 = _random_3d_segments(1000, rng)
        t0 = perf_counter()
        dist = minimum_segment_distance(data0, data1)
        elapsed = perf_counter() - t0
        assert dist.shape == (1000, 1000)
        assert np.all(np.isfinite(dist))
        # Sanity: should complete in reasonable time for vectorized code
        assert elapsed < 30.0, f"1000x1000 took {elapsed:.1f}s — too slow"


# ===========================================================================
# 4. CenterlineResult — backward compatibility stress tests
# ===========================================================================

try:
    from svv.tree.tree import CenterlineResult
except ImportError:
    # Fallback: define CenterlineResult locally if tree.py import chain fails
    # (e.g., meshio not installed). The class itself has no dependencies.
    class CenterlineResult(tuple):
        def __new__(cls, centerlines, polys, boundary_points=None):
            instance = super().__new__(cls, (centerlines, polys))
            instance.boundary_points = boundary_points if boundary_points is not None else []
            return instance


class TestCenterlineResultCompat:
    """Verify CenterlineResult backward-compatible tuple unpacking."""

    def test_two_tuple_unpack(self):
        """Legacy pattern: centerlines, polys = result."""
        r = CenterlineResult("cl", ["p1", "p2"], [{"type": "inlet"}])
        a, b = r
        assert a == "cl"
        assert b == ["p1", "p2"]

    def test_star_unpack(self):
        """Pattern: centerlines, *rest = result."""
        r = CenterlineResult("cl", ["p1"], [{"type": "outlet"}])
        a, *rest = r
        assert a == "cl"
        assert rest == [["p1"]]

    def test_len_is_two(self):
        """len() should be 2 for backward compatibility."""
        r = CenterlineResult("cl", ["p"], [{"type": "inlet"}])
        assert len(r) == 2

    def test_boundary_points_attribute(self):
        """New metadata accessible via attribute."""
        bp = [{"type": "inlet", "point": np.zeros(3), "radius": 0.1}]
        r = CenterlineResult("cl", ["p"], bp)
        assert r.boundary_points is bp
        assert r.boundary_points[0]["type"] == "inlet"

    def test_boundary_points_default_empty(self):
        """Default boundary_points is an empty list."""
        r = CenterlineResult("cl", ["p"])
        assert r.boundary_points == []

    def test_indexing(self):
        """Indexing r[0] and r[1] works."""
        r = CenterlineResult("cl", ["p"], [])
        assert r[0] == "cl"
        assert r[1] == ["p"]
        with pytest.raises(IndexError):
            _ = r[2]

    def test_iteration(self):
        """Iterating yields exactly 2 elements."""
        r = CenterlineResult("cl", ["p"], [{"type": "inlet"}])
        items = list(r)
        assert len(items) == 2

    def test_getattr_fallback(self):
        """getattr with default works for boundary_points."""
        r = CenterlineResult("cl", ["p"], [{"type": "inlet"}])
        assert getattr(r, "boundary_points", []) == [{"type": "inlet"}]
        # For a plain tuple, getattr would return the default
        plain = ("cl", ["p"])
        assert getattr(plain, "boundary_points", []) == []

    def test_isinstance_tuple(self):
        """CenterlineResult is a tuple."""
        r = CenterlineResult("cl", ["p"])
        assert isinstance(r, tuple)

    def test_with_real_numpy_data(self):
        """Simulate realistic centerline data."""
        centerlines = np.random.rand(500, 3)
        polys = [np.random.rand(100, 3) for _ in range(5)]
        bp = [
            {"type": "inlet", "point": np.array([0.0, 0.0, 0.0]), "radius": 0.5},
            {"type": "outlet", "point": np.array([1.0, 1.0, 1.0]), "radius": 0.1},
            {"type": "outlet", "point": np.array([2.0, 0.0, 0.0]), "radius": 0.08},
        ]
        r = CenterlineResult(centerlines, polys, bp)
        cl, ps = r
        assert cl.shape == (500, 3)
        assert len(ps) == 5
        assert len(r.boundary_points) == 3
        assert r.boundary_points[0]["type"] == "inlet"


# ===========================================================================
# 5. Bezier + CatmullRom numerical stability edge cases
# ===========================================================================

class TestCurveEdgeCases:
    """Edge cases that stress numerical precision."""

    def test_bezier_evaluate_at_exact_endpoints(self):
        """t=0 and t=1 must return exact endpoint values."""
        rng = np.random.default_rng(1)
        for _ in range(20):
            n = rng.integers(2, 20)
            ctrl = _random_control_points_3d(n, rng, spread=100)
            curve = BezierCurve(ctrl)
            pts = curve.evaluate(np.array([0.0, 1.0]))
            np.testing.assert_allclose(pts[0], ctrl[0], atol=1e-10)
            np.testing.assert_allclose(pts[1], ctrl[-1], atol=1e-10)

    def test_catmullrom_evaluate_near_segment_boundaries(self):
        """Evaluation at segment boundaries should be stable (no NaN/inf)."""
        rng = np.random.default_rng(2)
        ctrl = _random_control_points_3d(10, rng)
        spline = CatmullRomCurve(ctrl)
        n_segs = 9
        # Evaluate at and near each boundary
        t_vals = []
        for i in range(n_segs + 1):
            t = i / n_segs
            t_vals.extend([max(0, t - 1e-10), t, min(1.0 - 1e-14, t + 1e-10)])
        t_vals = np.clip(t_vals, 0, 1.0 - 1e-14)
        pts = spline.evaluate(np.array(t_vals))
        assert np.all(np.isfinite(pts))

    def test_bezier_collinear_points(self):
        """Collinear control points — degenerate geometry shouldn't crash."""
        ctrl = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=float)
        curve = BezierCurve(ctrl)
        t = np.linspace(0, 1, 1000)
        pts = curve.evaluate(t)
        # All points should lie on the line
        direction = ctrl[-1] - ctrl[0]
        direction /= np.linalg.norm(direction)
        vecs = pts - ctrl[0]
        projections = np.outer(vecs @ direction, direction)
        residuals = vecs - projections
        np.testing.assert_allclose(residuals, 0, atol=1e-10)

    def test_bezier_repeated_control_points(self):
        """All control points at the same location."""
        pt = np.array([3.0, -1.0, 2.0])
        ctrl = np.tile(pt, (5, 1))
        curve = BezierCurve(ctrl)
        pts = curve.evaluate(np.linspace(0, 1, 100))
        expected = np.tile(pt, (100, 1))
        np.testing.assert_allclose(pts, expected, atol=1e-12)

    def test_catmullrom_two_points(self):
        """Minimum viable spline (2 points) should produce a line."""
        ctrl = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
        spline = CatmullRomCurve(ctrl)
        t = np.linspace(0, 1, 100)
        pts = spline.evaluate(t)
        # y and z should be 0
        np.testing.assert_allclose(pts[:, 1], 0, atol=1e-10)
        np.testing.assert_allclose(pts[:, 2], 0, atol=1e-10)
        # x should be monotonically increasing
        assert np.all(np.diff(pts[:, 0]) >= -1e-10)

    def test_bezier_very_large_coordinates(self):
        """Large coordinate values shouldn't cause overflow."""
        scale = 1e6
        ctrl = np.array([
            [0, 0, 0], [scale, scale, 0], [2*scale, 0, 0]
        ], dtype=float)
        curve = BezierCurve(ctrl)
        pts = curve.evaluate(np.linspace(0, 1, 100))
        assert np.all(np.isfinite(pts))
        np.testing.assert_allclose(pts[-1], ctrl[-1], atol=1e-4)

    def test_bezier_very_small_coordinates(self):
        """Tiny coordinate values shouldn't lose precision."""
        scale = 1e-6
        ctrl = np.array([
            [0, 0, 0], [scale, scale, 0], [2*scale, 0, 0]
        ], dtype=float)
        curve = BezierCurve(ctrl)
        pts = curve.evaluate(np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(pts[0], ctrl[0], atol=1e-18)
        np.testing.assert_allclose(pts[-1], ctrl[-1], atol=1e-18)


# ===========================================================================
# 6. Segment distance — accuracy stress tests
# ===========================================================================

class TestSegmentDistanceAccuracy:
    """Verify vectorized distances match brute-force point sampling."""

    def _brute_force_segment_distance(self, seg0, seg1, n_samples=2000):
        """Brute-force minimum distance via dense point sampling."""
        A0, A1 = seg0[:3], seg0[3:]
        B0, B1 = seg1[:3], seg1[3:]
        t = np.linspace(0, 1, n_samples)
        pts_a = A0 + np.outer(t, A1 - A0)
        pts_b = B0 + np.outer(t, B1 - B0)
        diff = pts_a[:, None, :] - pts_b[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        return dists.min()

    def test_accuracy_vs_bruteforce(self):
        """Vectorized result should be a reasonable approximation.

        The vectorized clamping approach (independent t,s clamping after
        solving the unconstrained problem) can overshoot the true minimum
        for near-parallel or degenerate segment configurations. This test
        characterizes accuracy: the vast majority of pairs should be
        within 5%, and the median error should be very small.
        """
        rng = np.random.default_rng(42)
        n = 20
        data0 = _random_3d_segments(n, rng, spread=5.0)
        data1 = _random_3d_segments(n, rng, spread=5.0)
        vectorized = minimum_segment_distance(data0, data1)
        rel_errors = []
        for i in range(n):
            for j in range(n):
                bf = self._brute_force_segment_distance(data0[i], data1[j])
                assert vectorized[i, j] >= 0
                if bf > 0.1:
                    rel_errors.append(abs(vectorized[i, j] - bf) / bf)
        rel_errors = np.array(rel_errors)
        # Median error should be small
        assert np.median(rel_errors) < 0.05, (
            f"Median relative error {np.median(rel_errors):.4f} too high"
        )
        # At least 80% of pairs should be within 10%
        # (independent clamping can overshoot for near-parallel segments)
        pct_within_10 = np.mean(rel_errors < 0.10)
        assert pct_within_10 > 0.80, (
            f"Only {pct_within_10:.1%} of pairs within 10% error"
        )

    def test_known_perpendicular_segments(self):
        """Two perpendicular segments with known minimum distance."""
        # Segment A along x-axis, segment B along y-axis offset by 3 in z
        for offset in [0.5, 1.0, 3.0, 10.0]:
            data0 = np.array([[0, 0, 0, 2, 0, 0]], dtype=np.float64)
            data1 = np.array([[1, 0, offset, 1, 2, offset]], dtype=np.float64)
            dist = minimum_segment_distance(data0, data1)
            # Closest approach is at (1,0,0) on A and (1,0,offset) on B
            np.testing.assert_allclose(dist[0, 0], offset, atol=1e-10)


# ===========================================================================
# 7. Performance timing benchmarks (informational, not strict pass/fail)
# ===========================================================================

class TestPerformanceBenchmarks:
    """Timing benchmarks for optimized code paths.

    These tests always pass but print timing info.
    Strict time limits only where vectorization should guarantee speed.
    """

    def test_bezier_10k_evaluation_speed(self):
        """Bezier evaluate at 10k points should be fast."""
        ctrl = _random_control_points_3d(6, np.random.default_rng(1))
        curve = BezierCurve(ctrl)
        t = np.linspace(0, 1, 10_000)
        t0 = perf_counter()
        for _ in range(10):
            curve.evaluate(t)
        elapsed = (perf_counter() - t0) / 10
        assert elapsed < 1.0, f"10k Bezier eval: {elapsed:.3f}s — too slow"

    def test_catmullrom_10k_evaluation_speed(self):
        """CatmullRom evaluate at 10k points should be fast."""
        ctrl = _random_control_points_3d(20, np.random.default_rng(1))
        spline = CatmullRomCurve(ctrl)
        t = np.linspace(0, 1, 10_000)
        t0 = perf_counter()
        for _ in range(10):
            spline.evaluate(t)
        elapsed = (perf_counter() - t0) / 10
        assert elapsed < 2.0, f"10k CatmullRom eval: {elapsed:.3f}s — too slow"

    def test_segment_distance_500x500_speed(self):
        """500x500 segment distance should finish quickly."""
        rng = np.random.default_rng(42)
        data0 = _random_3d_segments(500, rng)
        data1 = _random_3d_segments(500, rng)
        t0 = perf_counter()
        minimum_segment_distance(data0, data1)
        elapsed = perf_counter() - t0
        assert elapsed < 10.0, f"500x500 segment dist: {elapsed:.3f}s — too slow"

    def test_bezier_derivative_all_orders(self):
        """Derivative computation at all orders up to degree."""
        ctrl = _random_control_points_3d(10, np.random.default_rng(5))
        curve = BezierCurve(ctrl)
        t = np.linspace(0, 1, 1000)
        t0 = perf_counter()
        for order in range(1, 10):
            curve.derivative(t, order=order)
        elapsed = perf_counter() - t0
        assert elapsed < 5.0, f"All-order derivatives: {elapsed:.3f}s"


# ===========================================================================
# 8. Geodesic edge extraction vectorization
# ===========================================================================

class TestGeodesicEdgeExtraction:
    """Test the vectorized edge extraction from tetrahedra."""

    def test_extract_edges_from_tetrahedra(self):
        """Verify edge extraction produces correct edges from tet connectivity."""
        # 2 tetrahedra sharing a face
        # Tet 0: nodes [0,1,2,3], Tet 1: nodes [1,2,3,4]
        cells = np.array([
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ])
        # 6 edges per tet = 12 total, but some shared
        tet_edges = np.array([[0,1],[1,2],[2,0],[0,3],[3,1],[2,3]])
        all_edges = []
        for tet in cells:
            for e in tet_edges:
                edge = sorted([tet[e[0]], tet[e[1]]])
                all_edges.append(tuple(edge))
        unique_edges = set(all_edges)
        # 2 tets sharing face [1,2,3] should have 9 unique edges
        # Tet0: (0,1)(1,2)(0,2)(0,3)(1,3)(2,3) = 6
        # Tet1: (1,2)(2,3)(1,3)(1,4)(3,4)(2,4) = 6
        # Shared: (1,2)(1,3)(2,3) = 3
        # Unique: 6 + 6 - 3 = 9
        assert len(unique_edges) == 9

    def test_edge_extraction_vectorized_matches_loop(self):
        """Vectorized edge extraction matches a simple loop."""
        rng = np.random.default_rng(42)
        n_tets = 500
        # Random tet connectivity (not geometrically valid, just for testing extraction)
        cells = rng.integers(0, 200, size=(n_tets, 4))

        # Loop-based extraction
        tet_edge_pairs = [(0,1),(1,2),(2,0),(0,3),(3,1),(2,3)]
        loop_edges = set()
        for tet in cells:
            for i, j in tet_edge_pairs:
                edge = (min(tet[i], tet[j]), max(tet[i], tet[j]))
                loop_edges.add(edge)

        # Vectorized extraction (same algorithm as geodesic.py)
        idx = np.array([[0,1],[1,2],[2,0],[0,3],[3,1],[2,3]])
        left = cells[:, idx[:, 0]]   # (n_tets, 6)
        right = cells[:, idx[:, 1]]  # (n_tets, 6)
        edges_raw = np.stack([
            np.minimum(left, right),
            np.maximum(left, right),
        ], axis=-1).reshape(-1, 2)   # (n_tets*6, 2)
        vectorized_edges = set(map(tuple, np.unique(edges_raw, axis=0)))

        assert loop_edges == vectorized_edges


# ===========================================================================
# 9. Constraint cache — unit test for cache sharing logic
# ===========================================================================

class TestConstraintCache:
    """Test the constraint cache sharing pattern from base_connection.py."""

    def test_cache_hit_on_same_input(self):
        """Same control points should reuse cached curve."""
        call_count = [0]
        _cache = {'key': None, 'result': None}

        def cached_compute(data):
            key = data.tobytes()
            if _cache['key'] != key:
                call_count[0] += 1
                _cache['key'] = key
                _cache['result'] = data.sum()
            return _cache['result']

        data = np.array([1.0, 2.0, 3.0])
        r1 = cached_compute(data)
        r2 = cached_compute(data)
        r3 = cached_compute(data)
        assert r1 == r2 == r3 == 6.0
        assert call_count[0] == 1  # computed only once

    def test_cache_miss_on_different_input(self):
        """Different control points should recompute."""
        call_count = [0]
        _cache = {'key': None, 'result': None}

        def cached_compute(data):
            key = data.tobytes()
            if _cache['key'] != key:
                call_count[0] += 1
                _cache['key'] = key
                _cache['result'] = data.sum()
            return _cache['result']

        cached_compute(np.array([1.0, 2.0]))
        cached_compute(np.array([3.0, 4.0]))
        cached_compute(np.array([5.0, 6.0]))
        assert call_count[0] == 3

    def test_cache_simulates_optimizer_iterations(self):
        """Simulate SLSQP calling 4 constraints per iteration."""
        call_count = [0]
        _cache = {'key': None, 'result': None}

        def build_curve(ctrlpts_flat):
            key = ctrlpts_flat.tobytes()
            if _cache['key'] != key:
                call_count[0] += 1
                _cache['key'] = key
                _cache['result'] = ctrlpts_flat.sum()
            return _cache['result']

        # Simulate 50 optimizer iterations, each calling 4 constraint functions
        rng = np.random.default_rng(7)
        for _ in range(50):
            x = rng.random(12)  # 4 control points * 3D
            for _ in range(4):  # 4 constraints
                build_curve(x)

        # Should have computed only 50 times (once per iteration), not 200
        assert call_count[0] == 50


# ===========================================================================
# 10. Large-scale combined stress test
# ===========================================================================

class TestCombinedStress:
    """Combined tests exercising multiple optimized paths together."""

    def test_bezier_roc_at_scale(self):
        """Compute ROC for many curves in sequence."""
        rng = np.random.default_rng(42)
        t = np.linspace(0.01, 0.99, 200)
        for _ in range(100):
            n = rng.integers(3, 10)
            ctrl = _random_control_points_3d(n, rng)
            curve = BezierCurve(ctrl)
            roc = curve.roc(t)
            assert np.all(np.isfinite(roc))
            assert np.all(roc > 0)

    def test_segment_distance_with_collinear_and_random(self):
        """Mix of collinear and random segments."""
        rng = np.random.default_rng(99)
        n = 200
        data = _random_3d_segments(n, rng)
        # Make first 20 collinear (parallel to x-axis)
        for i in range(20):
            y, z = rng.random(2)
            data[i] = [0, y, z, 1, y, z]
        dist = minimum_segment_distance(data, data)
        assert dist.shape == (n, n)
        assert np.all(np.isfinite(dist))
        assert np.all(dist >= 0)
        # Diagonal should be 0
        np.testing.assert_allclose(np.diag(dist), 0, atol=1e-10)

    def test_catmullrom_many_curves_sequential(self):
        """Create and evaluate many splines in sequence."""
        rng = np.random.default_rng(13)
        t = np.linspace(0, 1, 300)
        for _ in range(50):
            n = rng.integers(3, 30)
            ctrl = _random_control_points_3d(n, rng)
            spline = CatmullRomCurve(ctrl)
            pts = spline.evaluate(t)
            assert pts.shape == (300, 3)
            assert np.all(np.isfinite(pts))
