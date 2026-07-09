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

def uniform_remesh_surface(surface: pv.PolyData,
                           *,
                           subdivisions: int = 3,
                           clusters: int = 20000,
                           clean_tolerance: float = 1e-5) -> pv.PolyData:
    """
    Generate a uniform, isotropic triangle surface for TetGen retry attempts.

    PyACVD is imported lazily so callers that do not need the retry path do not
    pay the import cost until TetGen actually fails.
    """
    try:
        import pyacvd
    except ImportError as exc:
        raise RuntimeError(
            "PyACVD is required for TetGen uniform remeshing fallback. "
            "Install pyacvd or call tetrahedralize(..., remesh_on_failure=False)."
        ) from exc

    if subdivisions < 0:
        raise ValueError("subdivisions must be non-negative")
    if clusters <= 0:
        raise ValueError("clusters must be positive")

    if not isinstance(surface, pv.PolyData):
        surface = surface.extract_surface()
    base_mesh = pv.PolyData(surface.points, surface.faces)
    if clean_tolerance is not None:
        base_mesh = base_mesh.clean(tolerance=clean_tolerance)
    if not base_mesh.is_all_triangles:
        base_mesh = base_mesh.triangulate()

    if base_mesh.n_cells == 0:
        raise ValueError("Cannot remesh an empty surface")

    clustering = pyacvd.Clustering(base_mesh)
    if subdivisions:
        clustering.subdivide(int(subdivisions))
    clustering.cluster(int(clusters))
    remeshed = clustering.create_mesh()
    if clean_tolerance is not None:
        remeshed = remeshed.clean(tolerance=clean_tolerance)
    if not remeshed.is_all_triangles:
        remeshed = remeshed.triangulate()
    return remeshed


def _tetgen_worker_tetrahedralize(surface: pv.PolyData,
                                  tet_args,
                                  tet_kwargs,
                                  worker_script: str,
                                  python_exe: str):
    # On Windows, `tempfile` honors TMPDIR, which may be set to a POSIX-style
    # path such as '/tmp' and is not a valid directory there. Prefer the
    # standard TEMP/TMP locations when available to avoid spurious
    # "[WinError 267] The directory name is invalid" errors.
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

        # Save the surface mesh so the worker can read it
        surface.save(surface_path)

        # Command: call the worker script as a separate Python process
        cmd = [python_exe, worker_script, surface_path, out_path, config_path]

        # Start the worker process
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,   # decode to strings
        )

        show_spinner = sys.stdout.isatty()
        if show_spinner:
            spinner = _spinner_cycle()
            start_time = time.time()

            # Print label once
            sys.stdout.write("TetGen meshing| ")
            sys.stdout.flush()

            # Live spinner loop
            while proc.poll() is None:
                # Compute elapsed time
                elapsed = time.time() - start_time
                elapsed_str = format_elapsed(elapsed)

                # Build left side message
                spin_char = next(spinner)
                left = f"TetGen meshing| {spin_char}"

                # Get terminal width (fallback if IDE doesn't report it)
                try:
                    width = shutil.get_terminal_size(fallback=(80, 20)).columns
                except Exception:
                    width = 80

                # Compute spacing so elapsed time is right-aligned
                # We'll always keep at least one space between left and right
                min_gap = 1
                total_len = len(left) + min_gap + len(elapsed_str)
                if total_len <= width:
                    spaces = width - len(left) - len(elapsed_str)
                else:
                    # If line is longer than terminal, don't try to be clever; just put a single space
                    spaces = min_gap

                line = f"{left}{' ' * spaces}{elapsed_str}"

                # '\r' to return to the start of the same line and overwrite
                sys.stdout.write("\r" + line)
                sys.stdout.flush()

                time.sleep(0.1)

            # Finish line
            sys.stdout.write("\n")
            sys.stdout.flush()
        else:
            # Non-interactive environment (e.g., CI): just wait for the
            # worker process to finish without a live spinner to avoid
            # any potential overhead from frequent stdout updates.
            proc.wait()

        # Collect output (so the pipes don't hang)
        stdout, stderr = proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(
                f"TetGen worker failed with code {proc.returncode}\n"
                f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
            )

        # Load results and ensure the file handle is closed before the
        # temporary directory is cleaned up (important on Windows).
        with np.load(out_path) as data:
            nodes = data["nodes"]
            elems = data["elems"]

    return nodes, elems


def _tetgen_grid_from_arrays(nodes, elems):
    """
    Convert TetGen node/connectivity arrays into a PyVista unstructured grid.
    """
    nodes = np.asarray(nodes)
    elems = np.asarray(elems)
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


def tetrahedralize(surface: pv.PolyData,
                   *tet_args,
                   worker_script: str = dirpath+os.sep+"tetgen_worker.py",
                   python_exe: str = sys.executable,
                   remesh_on_failure: bool = True,
                   remesh_subdivisions: int = 3,
                   remesh_clusters: int = 20000,
                   remesh_clean_tolerance: float = 1e-5,
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
    remesh_on_failure : bool
        If True, retry TetGen once using a PyACVD uniform isotropic remesh
        when the original surface fails to tetrahedralize.
    remesh_subdivisions : int
        Number of PyACVD subdivision passes used by the retry path.
    remesh_clusters : int
        Number of PyACVD clusters used by the retry path.
    remesh_clean_tolerance : float
        PyVista clean tolerance applied before and after PyACVD remeshing.

    Returns
    -------
    mesh : PyMesh mesh object
        An unstructured grid mesh representing the tetrahedralized
        volume enclosed by the surface mesh manifold.
    """
    tet_kwargs.setdefault("verbose", 0)

    try:
        nodes, elems = _tetgen_worker_tetrahedralize(
            surface, tet_args, tet_kwargs, worker_script, python_exe
        )
    except RuntimeError as original_error:
        if not remesh_on_failure:
            raise
        try:
            remeshed_surface = uniform_remesh_surface(
                surface,
                subdivisions=remesh_subdivisions,
                clusters=remesh_clusters,
                clean_tolerance=remesh_clean_tolerance,
            )
        except Exception as remesh_error:
            raise RuntimeError(
                "TetGen failed and uniform surface remeshing fallback failed.\n\n"
                f"Original TetGen error:\n{original_error}\n\n"
                f"Remeshing error:\n{remesh_error}"
            ) from remesh_error

        try:
            nodes, elems = _tetgen_worker_tetrahedralize(
                remeshed_surface, tet_args, tet_kwargs, worker_script, python_exe
            )
        except RuntimeError as retry_error:
            raise RuntimeError(
                "TetGen failed after PyACVD uniform surface remeshing fallback.\n\n"
                f"Original TetGen error:\n{original_error}\n\n"
                f"Retry TetGen error:\n{retry_error}"
            ) from retry_error

    return _tetgen_grid_from_arrays(nodes, elems)
