import sys
import types

import pytest

class _MeshFixStub:
    def __init__(self, mesh, *args, **kwargs):
        self.mesh = mesh

    def repair(self, *args, **kwargs):
        return self.mesh


class _TetGenStub:
    def __init__(self, *args, **kwargs):
        pass


sys.modules.setdefault("pymeshfix", types.SimpleNamespace(MeshFix=_MeshFixStub))
sys.modules.setdefault("tetgen", types.SimpleNamespace(TetGen=_TetGenStub))
sys.modules.setdefault("meshio", types.SimpleNamespace())
sys.modules.setdefault("trimesh", types.SimpleNamespace(Trimesh=object))

import numpy as np
import pyvista as pv

from svv.simulation.simulation import Simulation
from svv.tree.data.data import TreeData
from svv.tree.tree import Tree


class _StaticDomain:
    def __init__(self, boundary):
        self.boundary = boundary



def _make_tree():
    tree = Tree()
    data = TreeData((2, 31))
    data[:] = np.nan

    data[0, 0:3] = [0.0, 0.0, 0.0]
    data[0, 3:6] = [1.0, 0.0, 0.0]
    data[1, 0:3] = [1.0, 0.0, 0.0]
    data[1, 3:6] = [2.0, 0.0, 0.0]
    data[:, 12:15] = [1.0, 0.0, 0.0]
    data[:, 20] = 1.0
    data[:, 21] = 0.1
    data[:, 22] = 0.05
    data[0, 15] = 1
    data[1, 17] = 0
    data[0, 26] = 0
    data[1, 26] = 1

    tree.data = data
    boundary = pv.Sphere(radius=3.0, theta_resolution=12, phi_resolution=12).triangulate()
    tree.domain = _StaticDomain(boundary)

    solid = pv.Sphere(radius=0.2, center=(0.5, 0.0, 0.0), theta_resolution=12, phi_resolution=12).triangulate()
    solid.cell_data["hsize"] = np.full(solid.n_cells, 0.1)
    tree.export_solid = lambda watertight=True, _solid=solid: _solid.copy(deep=True)
    return tree



def _make_fake_grid():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    cells = np.array([4, 0, 1, 2, 3], dtype=np.int64)
    celltypes = np.array([pv.CellType.TETRA], dtype=np.uint8)
    return pv.UnstructuredGrid(cells, celltypes, points)



def test_build_meshes_spline_point_constrained_uses_tree_splines(monkeypatch):
    import svv.simulation.simulation as sim_mod

    tree = _make_tree()
    sim = Simulation(tree)
    captured = {}

    monkeypatch.setattr(sim_mod, "remesh_surface", lambda mesh, **kwargs: mesh)
    monkeypatch.setattr(sim_mod, "boolean", lambda a, b, **kwargs: a)

    def fake_build(self, tissue_domain, *, constraint_data, **kwargs):
        captured["constraint_data"] = constraint_data
        grid = _make_fake_grid()
        elems = np.array([[0, 1, 2, 3]], dtype=np.int64)
        return grid, grid.points.copy(), elems, {"source": "fake", "point_count": constraint_data["points"].shape[0]}

    monkeypatch.setattr(Simulation, "_build_constrained_tissue_volume_mesh", fake_build)

    sim.build_meshes(
        fluid=False,
        tissue=True,
        tissue_mesh_type="spline_point_constrained",
        tissue_constraint_spline_sample_points=8,
    )

    assert "constraint_data" in captured
    constraint_data = captured["constraint_data"]
    assert constraint_data["points"].shape[0] > 0
    assert constraint_data["lines"].shape[0] > 0
    assert np.isfinite(constraint_data["radius"]).all()
    assert constraint_data["points"][:, 0].min() >= 0.0
    assert constraint_data["points"][:, 0].max() <= 2.0
    assert sim.tissue_constraint_metadata == [{"source": "fake", "point_count": constraint_data["points"].shape[0]}]
    assert len(sim.tissue_domain_volume_meshes) == 1
    assert sim.tissue_domain_volume_meshes[0].n_cells == 1



def test_build_meshes_spline_point_constrained_skips_boolean_tree_tissue(monkeypatch):
    import svv.simulation.simulation as sim_mod

    tree = _make_tree()
    tree.export_solid = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("export_solid should not be called"))
    sim = Simulation(tree)
    captured = {}

    monkeypatch.setattr(
        sim_mod,
        "remesh_surface",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("remesh_surface should not be called")),
    )
    monkeypatch.setattr(
        sim_mod,
        "boolean",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("boolean should not be called")),
    )

    def fake_build(self, tissue_domain, *, constraint_data, **kwargs):
        captured["tissue_domain"] = tissue_domain.copy(deep=True)
        grid = _make_fake_grid()
        elems = np.array([[0, 1, 2, 3]], dtype=np.int64)
        return grid, grid.points.copy(), elems, {"source": "fake", "point_count": constraint_data["points"].shape[0]}

    monkeypatch.setattr(Simulation, "_build_constrained_tissue_volume_mesh", fake_build)

    sim.build_meshes(
        fluid=False,
        tissue=True,
        tissue_mesh_type="spline_point_constrained",
        tissue_constraint_spline_sample_points=8,
    )

    assert np.allclose(captured["tissue_domain"].points, tree.domain.boundary.points)
    assert sim.fluid_domain_surface_meshes == []
    assert sim.fluid_domain_volume_meshes == []
    assert len(sim.tissue_domain_volume_meshes) == 1



def test_build_constrained_tissue_volume_mesh_does_not_meshfix_manifold_surface(monkeypatch):
    import svv.simulation.simulation as sim_mod

    tree = _make_tree()
    sim = Simulation(tree)
    constraint_data = {
        "points": np.array([[0.0, 0.0, 0.0]], dtype=float),
        "lines": np.empty((0, 2), dtype=np.int64),
        "spline_id": np.array([0], dtype=np.int32),
        "spline_order": np.array([0], dtype=np.int32),
        "radius": np.array([0.1], dtype=float),
    }

    monkeypatch.setattr(
        sim_mod,
        "tetrahedralize_with_prescribed_points",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        sim_mod.pymeshfix,
        "MeshFix",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("MeshFix should not be called")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        sim._build_constrained_tissue_volume_mesh(
            tree.domain.boundary,
            constraint_data=constraint_data,
        )



def test_build_meshes_spline_point_constrained_with_fluid_skips_boolean_and_root_extension(monkeypatch):
    import svv.simulation.simulation as sim_mod

    tree = _make_tree()
    sim = Simulation(tree)
    captured = {}
    original_root = tree.data[0, 0:3].copy()

    monkeypatch.setattr(sim_mod, "remesh_surface", lambda mesh, **kwargs: mesh)
    monkeypatch.setattr(
        sim_mod,
        "boolean",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("boolean should not be called")),
    )

    def fake_tetrahedralize(mesh, switches=None):
        grid = _make_fake_grid()
        elems = np.array([[0, 1, 2, 3]], dtype=np.int64)
        return grid, grid.points.copy(), elems

    def fake_build(self, tissue_domain, *, constraint_data, **kwargs):
        captured["tissue_domain"] = tissue_domain.copy(deep=True)
        grid = _make_fake_grid()
        elems = np.array([[0, 1, 2, 3]], dtype=np.int64)
        return grid, grid.points.copy(), elems, {"source": "fake", "point_count": constraint_data["points"].shape[0]}

    monkeypatch.setattr(sim_mod, "tetrahedralize", fake_tetrahedralize)
    monkeypatch.setattr(Simulation, "_build_constrained_tissue_volume_mesh", fake_build)

    sim.build_meshes(
        fluid=True,
        tissue=True,
        boundary_layer=False,
        tissue_mesh_type="spline_point_constrained",
        tissue_constraint_spline_sample_points=8,
    )

    assert np.allclose(tree.data[0, 0:3], original_root)
    assert np.allclose(captured["tissue_domain"].points, tree.domain.boundary.points)
    assert len(sim.fluid_domain_volume_meshes) == 1
    assert len(sim.tissue_domain_volume_meshes) == 1
