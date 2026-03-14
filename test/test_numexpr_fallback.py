import numpy as np

from svv.tree.branch import bifurcation as bifurcation_mod


def test_numexpr_helpers_fall_back_to_numpy(monkeypatch):
    monkeypatch.setattr(bifurcation_mod, "_NUMEXPR", None)

    column = np.array([1.0, 2.0, 3.0], dtype=float)
    out = np.zeros_like(column)
    bifurcation_mod.ne_multiply(column, 2.5, out)
    np.testing.assert_allclose(out, column * 2.5)

    radius = np.array([1.0, 2.0], dtype=float)
    length = np.array([3.0, 4.0], dtype=float)
    scale = bifurcation_mod.ne_scale(radius, length, 2.0, 1.0)
    expected = np.pi * np.sum(radius ** 2.0 * length)
    np.testing.assert_allclose(scale, expected)
