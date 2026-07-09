import importlib
import sys
import types

import numpy as np
import pytest
import pyvista as pv


def _closed_tetra_surface():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [3, 0, 2, 1],
            [3, 0, 1, 3],
            [3, 1, 2, 3],
            [3, 2, 0, 3],
        ],
        dtype=np.int64,
    ).ravel()
    return pv.PolyData(points, faces)


@pytest.fixture()
def tetra_mod(monkeypatch):
    fake_tetgen = types.ModuleType("tetgen")
    fake_tetgen.TetGen = object
    monkeypatch.setitem(sys.modules, "tetgen", fake_tetgen)

    fake_pymeshfix = types.ModuleType("pymeshfix")
    fake_pymeshfix.MeshFix = object
    monkeypatch.setitem(sys.modules, "pymeshfix", fake_pymeshfix)

    fake_remesh = types.ModuleType("svv.utils.remeshing.remesh")
    fake_remesh.remesh_surface = lambda mesh, **kwargs: mesh
    monkeypatch.setitem(sys.modules, "svv.utils.remeshing.remesh", fake_remesh)

    import svv.utils.remeshing as remeshing_pkg

    monkeypatch.setattr(remeshing_pkg, "remesh", fake_remesh, raising=False)
    sys.modules.pop("svv.domain.routines.tetrahedralize", None)
    module = importlib.import_module("svv.domain.routines.tetrahedralize")
    yield module
    sys.modules.pop("svv.domain.routines.tetrahedralize", None)


def test_tetrahedralize_retries_with_uniform_remesh(tetra_mod, monkeypatch):
    surface = _closed_tetra_surface()
    remeshed_surface = _closed_tetra_surface()
    nodes = surface.points.copy()
    elems = np.array([[0, 1, 2, 3]], dtype=np.int64)
    worker_calls = []
    remesh_calls = []

    def fake_worker(surface_arg, tet_args, tet_kwargs, worker_script, python_exe):
        worker_calls.append((surface_arg, tet_args, dict(tet_kwargs), worker_script, python_exe))
        if len(worker_calls) == 1:
            raise RuntimeError("Delaunay criterion failed")
        return nodes, elems

    def fake_remesh(surface_arg, *, subdivisions, clusters, clean_tolerance):
        remesh_calls.append(
            {
                "surface": surface_arg,
                "subdivisions": subdivisions,
                "clusters": clusters,
                "clean_tolerance": clean_tolerance,
            }
        )
        return remeshed_surface

    monkeypatch.setattr(tetra_mod, "_tetgen_worker_tetrahedralize", fake_worker)
    monkeypatch.setattr(tetra_mod, "uniform_remesh_surface", fake_remesh)

    grid, out_nodes, out_elems = tetra_mod.tetrahedralize(
        surface,
        order=1,
        nobisect=True,
        remesh_on_failure=True,
        remesh_subdivisions=2,
        remesh_clusters=7,
        remesh_clean_tolerance=1e-4,
    )

    assert grid.n_cells == 1
    np.testing.assert_allclose(out_nodes, nodes)
    np.testing.assert_array_equal(out_elems, elems)
    assert len(worker_calls) == 2
    assert worker_calls[0][0] is surface
    assert worker_calls[1][0] is remeshed_surface
    assert worker_calls[0][2]["order"] == 1
    assert worker_calls[0][2]["nobisect"] is True
    assert remesh_calls == [
        {
            "surface": surface,
            "subdivisions": 2,
            "clusters": 7,
            "clean_tolerance": 1e-4,
        }
    ]


def test_tetrahedralize_can_disable_remesh_retry(tetra_mod, monkeypatch):
    surface = _closed_tetra_surface()
    remesh_called = False

    def fake_worker(surface_arg, tet_args, tet_kwargs, worker_script, python_exe):
        raise RuntimeError("TetGen failed before retry")

    def fake_remesh(*args, **kwargs):
        nonlocal remesh_called
        remesh_called = True
        return surface

    monkeypatch.setattr(tetra_mod, "_tetgen_worker_tetrahedralize", fake_worker)
    monkeypatch.setattr(tetra_mod, "uniform_remesh_surface", fake_remesh)

    with pytest.raises(RuntimeError, match="TetGen failed before retry"):
        tetra_mod.tetrahedralize(surface, remesh_on_failure=False)

    assert remesh_called is False


def test_uniform_remesh_surface_uses_pyacvd_options(tetra_mod, monkeypatch):
    surface = _closed_tetra_surface()
    calls = {}

    class FakeClustering:
        def __init__(self, mesh):
            calls["mesh"] = mesh
            self.mesh = mesh

        def subdivide(self, subdivisions):
            calls["subdivide"] = subdivisions

        def cluster(self, clusters):
            calls["cluster"] = clusters

        def create_mesh(self):
            return self.mesh

    fake_pyacvd = types.ModuleType("pyacvd")
    fake_pyacvd.Clustering = FakeClustering
    monkeypatch.setitem(sys.modules, "pyacvd", fake_pyacvd)

    out = tetra_mod.uniform_remesh_surface(
        surface,
        subdivisions=2,
        clusters=11,
        clean_tolerance=1e-6,
    )

    assert out.is_all_triangles
    assert calls["mesh"].is_all_triangles
    assert calls["subdivide"] == 2
    assert calls["cluster"] == 11
