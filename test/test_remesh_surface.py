import numpy as np
import pytest


def _fake_run_mmg(tool, args, stdout=None, stderr=None, cwd=None):
    import meshio
    from pathlib import Path

    assert tool == "mmgs"
    assert cwd is not None
    cwd_path = Path(cwd)

    in_mesh = cwd_path / str(args[0])
    text = in_mesh.read_text(errors="ignore")
    # Regression check: MMGS input must contain triangles (e.g., pv.Cube() starts as quads).
    assert "Triangles" in text

    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    triangles = np.array([[0, 1, 2]], dtype=int)
    mesh = meshio.Mesh(points, cells=[("triangle", triangles)])
    meshio.write(str(cwd_path / "tmp.o.mesh"), mesh, file_format="medit")
    return cwd_path / "mmgs_O3"


def test_remesh_surface_triangulates_quads(monkeypatch, tmp_path):
    import pyvista as pv
    from svv.utils.remeshing import remesh as remesh_mod

    monkeypatch.setattr(remesh_mod, "run_mmg", _fake_run_mmg)
    monkeypatch.chdir(tmp_path)

    out = remesh_mod.remesh_surface(pv.Cube(), autofix=False, verbosity=0)
    assert out.is_all_triangles


def test_remesh_surface_required_triangles_requires_tri_mesh():
    import pyvista as pv
    from svv.utils.remeshing import remesh as remesh_mod

    with pytest.raises(ValueError, match="requires the input surface to be all triangles"):
        remesh_mod.remesh_surface(pv.Cube(), required_triangles=[1], autofix=False, verbosity=0)


def test_remesh_surface_deletes_in_sol_on_success(monkeypatch, tmp_path):
    import pyvista as pv
    from svv.utils.remeshing import remesh as remesh_mod

    monkeypatch.setattr(remesh_mod, "run_mmg", _fake_run_mmg)
    monkeypatch.chdir(tmp_path)

    (tmp_path / "in.sol").write_text("dummy sizing function\n")
    remesh_mod.remesh_surface(pv.Cube(), autofix=False, verbosity=0)
    assert not (tmp_path / "in.sol").exists()

