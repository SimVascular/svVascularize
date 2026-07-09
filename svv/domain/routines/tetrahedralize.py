import tetgen
import pymeshfix
import subprocess
import tempfile
import os
import sys
from tqdm.auto import tqdm
from itertools import cycle
from time import sleep
import time
import numpy as np
import pyvista as pv
from svv.utils.remeshing import remesh
import shutil
import json

filepath = os.path.abspath(__file__)
dirpath = os.path.dirname(filepath)

def format_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"


def _spinner_cycle():
    """
    Return a spinner cycle that is safe for the current stdout encoding.

    On Windows runners the default code page may not support the braille
    characters used by common Unicode spinners, which can raise a
    UnicodeEncodeError when writing to sys.stdout. To avoid this, we
    fall back to a simple ASCII spinner if the encoding cannot handle
    the Unicode characters.
    """
    ascii_spinner = ["-", "\\", "|", "/"]

    encoding = getattr(sys.stdout, "encoding", None)
    if not encoding:
        return cycle(ascii_spinner)

    fancy_spinner = ["⠋", "⠙", "⠹", "⠸", "⠼",
                     "⠴", "⠦", "⠧", "⠇", "⠏"]
    try:
        "".join(fancy_spinner).encode(encoding)
    except Exception:
        return cycle(ascii_spinner)

    return cycle(fancy_spinner)

def triangulate(curve, verbose=False, **kwargs):
    """
    Triangulate a curve using VTK.

    Parameters
    ----------
    curve : Pyvista.PolyData PolyLine object
        The boundary curve within which the triangulation will
        be performed.
    verbose : bool
        A flag to indicate if mesh fixing should be verbose.
    kwargs : dict
        A dictionary of keyword arguments to be passed to VTK.

    Returns
    -------
    mesh : PyMesh mesh object
        A triangular mesh representing the triangulated region bounded by
        the curve.
    nodes : ndarray
        An array of node coordinates for the triangular mesh.
    vertices : ndarray
        An array of vertex indices for the triangular mesh.
    """
    mesh = curve.delaunay_2d(**kwargs)
    mesh = remesh.remesh_surface(mesh)
    nodes = mesh.points
    vertices = mesh.faces.reshape(-1, 4)[:, 1:]
    return mesh, nodes, vertices

def _run_tetgen(surface_mesh):
    tgen = tetgen.TetGen(surface_mesh)
    nodes, elems = tgen.tetrahedralize(verbose=0)
    return nodes, elems

def _tetrahedralize_subprocess(surface, *tet_args,
                                worker_script=None,
                                python_exe=None,
                                **tet_kwargs):
    """Subprocess fallback for TetGen crash isolation."""
    if worker_script is None:
        worker_script = dirpath + os.sep + "tetgen_worker.py"
    if python_exe is None:
        python_exe = sys.executable

    tmp_root = None
    if os.name == "nt":
        for env_var in ("TEMP", "TMP"):
            candidate = os.environ.get(env_var)
            if candidate and os.path.isdir(candidate):
                tmp_root = candidate
                break

    with tempfile.TemporaryDirectory(dir=tmp_root) as tmpdir:
        surface_path = os.path.join(tmpdir, "surface.vtp")
        out_path = os.path.join(tmpdir, "tet.npz")
        config_path = os.path.join(tmpdir, "config.json")

        cfg = {
            "args": list(tet_args),
            "kwargs": tet_kwargs,
        }
        with open(config_path, "w") as f:
            json.dump(cfg, f)

        surface.save(surface_path)

        cmd = [python_exe, worker_script, surface_path, out_path, config_path]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        show_spinner = sys.stdout.isatty()
        if show_spinner:
            spinner = _spinner_cycle()
            start_time = time.time()

            sys.stdout.write("TetGen meshing| ")
            sys.stdout.flush()

            while proc.poll() is None:
                elapsed = time.time() - start_time
                elapsed_str = format_elapsed(elapsed)

                spin_char = next(spinner)
                left = f"TetGen meshing| {spin_char}"

                try:
                    width = shutil.get_terminal_size(fallback=(80, 20)).columns
                except Exception:
                    width = 80

                min_gap = 1
                total_len = len(left) + min_gap + len(elapsed_str)
                if total_len <= width:
                    spaces = width - len(left) - len(elapsed_str)
                else:
                    spaces = min_gap

                line = f"{left}{' ' * spaces}{elapsed_str}"
                sys.stdout.write("\r" + line)
                sys.stdout.flush()

                time.sleep(0.1)

            sys.stdout.write("\n")
            sys.stdout.flush()
        else:
            proc.wait()

        stdout, stderr = proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(
                f"TetGen worker failed with code {proc.returncode}\n"
                f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
            )

        with np.load(out_path) as data:
            nodes = data["nodes"]
            elems = data["elems"]

    return nodes, elems

def tetrahedralize(surface: pv.PolyData,
                   *tet_args,
                   worker_script: str = dirpath+os.sep+"tetgen_worker.py",
                   python_exe: str = sys.executable,
                   **tet_kwargs):
    """
    Tetrahedralize a surface mesh using TetGen.

    Parameters
    ----------
    surface_mesh : PyMesh mesh object
        The surface mesh to tetrahedralize.
    verbose : bool
        A flag to indicate if mesh fixing should be verbose.
    kwargs : dict
        A dictionary of keyword arguments to be passed to TetGen.

    Returns
    -------
    mesh : PyMesh mesh object
        An unstructured grid mesh representing the tetrahedralized
        volume enclosed by the surface mesh manifold.
    """
    tet_kwargs.setdefault("verbose", 0)

    # Try in-process TetGen first (avoids subprocess overhead: save/load/spawn)
    try:
        nodes, elems = _run_tetgen(surface)
    except Exception:
        # Fall back to subprocess worker for crash isolation
        nodes, elems = _tetrahedralize_subprocess(
            surface, *tet_args,
            worker_script=worker_script,
            python_exe=python_exe,
            **tet_kwargs
        )

    if elems.min() == 1:
        elems = elems - 1

    n_cells, n_vertices_per_cell = elems.shape
    cells = np.hstack(
        [
            np.full((n_cells, 1), n_vertices_per_cell, dtype=np.int64),
            elems.astype(np.int64),
        ]
    ).ravel()
    if n_vertices_per_cell == 4:
        celltypes = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
    elif n_vertices_per_cell == 10:
        celltypes = np.full(n_cells, pv.CellType.QUADRATIC_TETRA, dtype=np.uint8)
    else:
        raise ValueError(f"Unexpected number of vertices per cell: {n_vertices_per_cell}")

    grid = pv.UnstructuredGrid(cells, celltypes, nodes)

    return grid, nodes, elems
