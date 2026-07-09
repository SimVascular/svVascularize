import os
import json
from typing import Optional, Union

import numpy as np
from scipy.spatial import cKDTree

try:
    import pyvista as pv  # Optional for boundary/mesh restore
except Exception:  # pragma: no cover
    pv = None


DMN_FORMAT = "svv.domain/1.0"


def ensure_dmn_path(path: Union[str, os.PathLike]) -> str:
    """
    Normalize an output path for .dmn persistence.

    - Accepts ``str`` or ``PathLike`` objects.
    - Ensures the returned path ends with a lowercase ``.dmn`` extension.
    - Collapses legacy ``.dmn.npz`` filenames (from NumPy ``savez`` behavior)
      to ``.dmn``.
    """
    path_str = os.fsdecode(os.fspath(path))
    lower = path_str.lower()

    # Legacy: np.savez_compressed("name.dmn", ...) writes "name.dmn.npz"
    if lower.endswith(".npz") and lower[:-4].endswith(".dmn"):
        path_str = path_str[:-4]
        lower = lower[:-4]

    if lower.endswith(".dmn"):
        return path_str[:-4] + ".dmn"
    return path_str + ".dmn"


def _ensure_ext(path: Union[str, os.PathLike]) -> str:
    return ensure_dmn_path(path)


def _compute_firsts_from_pts(PTS: np.ndarray) -> np.ndarray:
    """
    Derive representative points ("firsts") for each patch from padded PTS.
    PTS shape: (n_patches, max_pts, 1, 1, 1, d)
    """
    n_patches = PTS.shape[0]
    d = PTS.shape[-1]
    firsts = np.zeros((n_patches, d), dtype=float)
    for i in range(n_patches):
        pts_i = PTS[i, :, 0, 0, 0, :]
        mask = np.any(~np.isnan(pts_i), axis=1)
        if not np.any(mask):
            # Fallback to zeros if empty row (should not happen for solved domains)
            firsts[i] = np.zeros((d,), dtype=float)
        else:
            firsts[i] = pts_i[np.argmax(mask)]
    return firsts


def write_dmn(domain, path: Union[str, os.PathLike], include_boundary: bool = False, include_mesh: bool = False,
              include_patch_normals: bool = True) -> str:
    """
    Serialize a Domain instance into a .dmn file.

    The .dmn container is a compressed NumPy archive written with a .dmn
    extension. It stores the precomputed arrays required for fast evaluation
    and enough metadata to rebuild the Domain state on load.

    Parameters
    ----------
    domain : svv.domain.domain.Domain
        The Domain to serialize. Must have been created/solved/built so that
        A/B/C/D/PTS arrays exist and a function_tree is available.
    path : str
        Output filename. If no ".dmn" extension is provided, it will be added.
    include_boundary : bool, optional
        Include boundary (points + faces or lines) to restore visualization
        without recomputing. Requires pyvista at load time. Default False.
    include_mesh : bool, optional
        Include interior mesh (nodes + connectivity) to restore interior sampling
        and mesh metrics without recomputing. Requires pyvista (and BallTree) at
        load time. Default False.
    include_patch_normals : bool, optional
        Include per‑patch normals (when available) padded to the same shape as
        PTS. This enables full Patch reconstruction on load and subsequent calls
        to Domain.build() to follow the standard workflow. Default True.

    Notes
    -----
    - The writer validates that the Domain has the fast‑eval arrays and a
      function_tree. If missing, call `domain.solve(...); domain.build(...)`
      before saving.
    - The container also stores representative patch points ("firsts") used to
      rebuild the cKDTree on load.
    """
    # Validate prerequisites
    if not hasattr(domain, "A") or not hasattr(domain, "B") or not hasattr(domain, "C") \
            or not hasattr(domain, "D") or not hasattr(domain, "PTS"):
        raise ValueError("Domain is not solved. Call domain.solve() before saving.")
    if domain.function_tree is None:
        raise ValueError("Domain.function_tree is not initialized. Call domain.solve() before saving.")

    out_path = _ensure_ext(path)

    # Core arrays for fast evaluation
    A = np.asarray(domain.A)
    B = np.asarray(domain.B)
    C = np.asarray(domain.C)
    D = np.asarray(domain.D)
    PTS = np.asarray(domain.PTS)

    # Representative points to rebuild the function k-d tree
    firsts = _compute_firsts_from_pts(PTS)

    # Base metadata
    meta = {
        "format": DMN_FORMAT,
        "dims": int(domain.points.shape[1]),
        "has_normals": bool(domain.normals is not None),
        "random_seed": None if domain.random_seed is None else int(domain.random_seed),
        "characteristic_length": None if domain.characteristic_length is None else float(domain.characteristic_length),
        "convexity": None if domain.convexity is None else float(domain.convexity),
        "has_patch_normals": False,
    }
    meta_json = json.dumps(meta)


    arrays = {
        "__meta__": np.frombuffer(meta_json.encode("utf-8"), dtype=np.uint8),
        "points": np.asarray(domain.points, dtype=np.float64),
        "normals": np.asarray(domain.normals, dtype=np.float64) if domain.normals is not None else np.empty((0, 0)),
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "PTS": PTS,
        "firsts": firsts,
    }

    # Optional: per-patch normals padded like PTS
    if include_patch_normals:
        patch_normals = []
        have_any = False
        # Prefer normals attached to built functions
        funcs = getattr(domain, 'functions', None)
        if funcs is not None and len(funcs) == A.shape[0]:
            for i, f in enumerate(funcs):
                nrm = getattr(f, 'normals', None)
                if isinstance(nrm, np.ndarray) and nrm.ndim == 2 and nrm.shape[1] == domain.points.shape[1]:
                    have_any = True
                    patch_normals.append(nrm)
                else:
                    patch_normals.append(None)
        else:
            # Fall back to patch objects if available
            patches = getattr(domain, 'patches', None)
            if patches is not None and len(patches) == A.shape[0]:
                for p in patches:
                    nrm = getattr(p, 'normals', None)
                    if isinstance(nrm, np.ndarray) and nrm.ndim == 2 and nrm.shape[1] == domain.points.shape[1]:
                        have_any = True
                        patch_normals.append(nrm)
                    else:
                        patch_normals.append(None)
        if have_any and len(patch_normals) == A.shape[0]:
            max_pts = PTS.shape[1]
            d = domain.points.shape[1]
            PNORMALS = np.full((A.shape[0], max_pts, 1, 1, 1, d), np.nan, dtype=np.float64)
            for i in range(A.shape[0]):
                if isinstance(patch_normals[i], np.ndarray):
                    n_i = min(max_pts, patch_normals[i].shape[0])
                    PNORMALS[i, :n_i, 0, 0, 0, :] = patch_normals[i][:n_i, :]
            arrays["PNORMALS"] = PNORMALS
            meta["has_patch_normals"] = True
            meta_json = json.dumps(meta)
            arrays["__meta__"] = np.frombuffer(meta_json.encode("utf-8"), dtype=np.uint8)

    # Optional boundary
    if include_boundary and getattr(domain, "boundary", None) is not None:
        if pv is None:
            raise RuntimeError("pyvista is required to save boundary data. Install pyvista or disable include_boundary.")
        arrays["boundary_points"] = np.asarray(domain.boundary.points, dtype=np.float64)
        # Save both faces and lines; one will be empty depending on dimension
        faces = getattr(domain.boundary, "faces", None)
        lines = getattr(domain.boundary, "lines", None)
        arrays["boundary_faces"] = np.asarray(faces, dtype=np.int64) if faces is not None else np.empty((0,), dtype=np.int64)
        arrays["boundary_lines"] = np.asarray(lines, dtype=np.int64) if lines is not None else np.empty((0,), dtype=np.int64)

        # Also save original_boundary if it exists (the input mesh before processing)
        orig_boundary = getattr(domain, "original_boundary", None)
        if orig_boundary is not None:
            arrays["original_boundary_points"] = np.asarray(orig_boundary.points, dtype=np.float64)
            orig_faces = getattr(orig_boundary, "faces", None)
            orig_lines = getattr(orig_boundary, "lines", None)
            arrays["original_boundary_faces"] = np.asarray(orig_faces, dtype=np.int64) if orig_faces is not None else np.empty((0,), dtype=np.int64)
            arrays["original_boundary_lines"] = np.asarray(orig_lines, dtype=np.int64) if orig_lines is not None else np.empty((0,), dtype=np.int64)

    # Optional interior mesh (nodes + connectivity) and quick stats
    if include_mesh and getattr(domain, "mesh", None) is not None:
        if pv is None:
            raise RuntimeError("pyvista is required to save mesh data. Install pyvista or disable include_mesh.")
        arrays["mesh_nodes"] = np.asarray(domain.mesh_nodes, dtype=np.float64)
        arrays["mesh_vertices"] = np.asarray(domain.mesh_vertices, dtype=np.int64)
        arrays["mesh_dim"] = np.array([domain.points.shape[1]], dtype=np.int64)

    # Use a file handle to avoid NumPy forcing a .npz extension
    with open(out_path, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    return out_path


def resolve_dmn_read_path(path: Union[str, os.PathLike]) -> str:
    """
    Resolve a path for reading a Domain from disk.

    Accepts filenames both with and without the ``.dmn`` suffix. Also attempts
    to load legacy ``.dmn.npz`` files created by passing a ``.dmn`` filename
    directly to NumPy ``savez`` routines.
    """
    path_str = os.fsdecode(os.fspath(path))
    if os.path.isfile(path_str):
        return path_str

    candidate = ensure_dmn_path(path_str)
    if os.path.isfile(candidate):
        return candidate

    legacy_candidate = candidate + ".npz"
    if os.path.isfile(legacy_candidate):
        return legacy_candidate

    return candidate


def read_dmn(path: Union[str, os.PathLike]):
    """
    Deserialize a Domain from a .dmn file.

    Returns a ready‑to‑use Domain with fast evaluation path initialized, and
    reconstructs Patch objects (including normals, when stored) so that
    Domain.build() can be invoked to derive boundary/mesh artifacts.

    Parameters
    ----------
    path : str
        The input .dmn file.

    Returns
    -------
    Domain
        A Domain instance with points, normals (if any), precomputed fast‑eval
        arrays, function_tree, and reconstructed patches. If boundary/mesh were
        saved, they are also restored.
    """
    from svv.domain.domain import Domain
    from svv.domain.patch import Patch

    file_path = resolve_dmn_read_path(path)
    data = np.load(file_path, allow_pickle=False)

    # Parse metadata
    if "__meta__" not in data:
        raise ValueError("Invalid .dmn file: missing metadata.")
    meta_json = bytes(data["__meta__"].tolist()).decode("utf-8")
    meta = json.loads(meta_json)
    if not isinstance(meta, dict) or meta.get("format") != DMN_FORMAT:
        raise ValueError("Unsupported .dmn format or version.")

    # Create bare Domain and assign core arrays
    points = np.asarray(data["points"], dtype=np.float64)
    normals = data.get("normals")
    if normals is not None and normals.size == 0:
        normals = None
    else:
        normals = np.asarray(normals, dtype=np.float64)

    dom = Domain(points, normals) if normals is not None else Domain(points)

    # Attach fast-eval arrays
    dom.A = np.asarray(data["A"])  # (n_patches, max_a, 1, 1, 1)
    dom.B = np.asarray(data["B"])  # (n_patches, max_b, 1, 1, 1, d)
    dom.C = np.asarray(data["C"])  # (n_patches, 1, 1, 1, d)
    dom.D = np.asarray(data["D"])  # (n_patches, 1, 1, 1)
    dom.PTS = np.asarray(data["PTS"])  # (n_patches, max_pts, 1, 1, 1, d)

    # Rebuild function k-d tree from representative points
    firsts = np.asarray(data["firsts"], dtype=np.float64)
    dom.function_tree = cKDTree(firsts)

    # Restore meta
    dom.random_seed = meta.get("random_seed", None)
    # Always set random_generator - it's required for tree/forest generation
    # even if no specific seed was saved
    dom.set_random_generator()
    dom.characteristic_length = meta.get("characteristic_length", None)
    dom.convexity = meta.get("convexity", None)

    # Only reconstruct patches if mesh data is NOT included in the file.
    # If mesh data is present, patches are not needed (they're only used for rebuild).
    # This significantly speeds up loading for large domains.
    has_mesh_data = "mesh_nodes" in data
    if not has_mesh_data:
        # Reconstruct patches from stored PTS (and patch normals if available) and parameter arrays so that callers
        # can invoke Domain.build() again if desired.
        try:
            n_patches = dom.PTS.shape[0]
            dom.patches = []
            d = points.shape[1]
            for i in range(n_patches):
                pts_i = dom.PTS[i, :, 0, 0, 0, :]
                valid_mask = np.any(~np.isnan(pts_i), axis=1)
                n_i = int(valid_mask.sum())
                if n_i == 0:
                    continue
                pts_trim = pts_i[valid_mask]
                # Per-patch normals if present
                norms_trim = None
                if "PNORMALS" in data:
                    pn_i = np.asarray(data["PNORMALS"][i, :, 0, 0, 0, :], dtype=np.float64)
                    valid_n = np.any(~np.isnan(pn_i), axis=1)
                    if int(valid_n.sum()) >= n_i:
                        norms_trim = pn_i[valid_n][:n_i]
                # Build constants for this patch from A/B/C/D
                a_i = dom.A[i, :n_i, 0, 0, 0]
                b_i = dom.B[i, :n_i, 0, 0, 0, :]
                c_i = dom.C[i, 0, 0, 0, :]
                d_i = float(dom.D[i, 0, 0, 0])
                constants = np.concatenate([a_i, b_i.reshape(-1), c_i.reshape(-1), np.array([d_i])])
                p = Patch()
                if norms_trim is not None:
                    p.set_data(pts_trim, norms_trim, create_kernel=False)
                else:
                    p.set_data(pts_trim, create_kernel=False)
                p.constants = constants
                dom.patches.append(p)
        except Exception:
            # If anything goes wrong, leave patches absent; evaluation still works.
            dom.patches = []
    else:
        # Skip patch reconstruction - not needed when mesh is loaded
        dom.patches = []

    # Optional: boundary
    if "boundary_points" in data and pv is not None:
        b_pts = np.asarray(data["boundary_points"], dtype=np.float64)
        b_faces = np.asarray(data.get("boundary_faces", np.empty((0,), dtype=np.int64)))
        b_lines = np.asarray(data.get("boundary_lines", np.empty((0,), dtype=np.int64)))
        if b_faces.size > 0:
            dom.boundary = pv.PolyData(b_pts, b_faces)
        elif b_lines.size > 0:
            dom.boundary = pv.PolyData(b_pts, lines=b_lines)
        else:
            dom.boundary = pv.PolyData(b_pts)
        dom.boundary = dom.boundary.compute_cell_sizes()
        if points.shape[1] == 2:
            dom.boundary_nodes = dom.boundary.points.astype(np.float64)
            dom.boundary_vertices = dom.boundary.lines.reshape(-1, 3)[:, 1:].astype(np.int64)
            dom.boundary.cell_data['Normalized_Length'] = (
                dom.boundary.cell_data['Length'] / sum(dom.boundary.cell_data['Length'])
            )
        elif points.shape[1] == 3:
            dom.boundary_nodes = dom.boundary.points.astype(np.float64)
            dom.boundary_vertices = dom.boundary.faces.reshape(-1, 4)[:, 1:].astype(np.int64)
            dom.boundary.cell_data['Normalized_Area'] = (
                dom.boundary.cell_data['Area'] / sum(dom.boundary.cell_data['Area'])
            )

    # Optional: original_boundary (the input mesh before processing)
    if "original_boundary_points" in data and pv is not None:
        ob_pts = np.asarray(data["original_boundary_points"], dtype=np.float64)
        ob_faces = np.asarray(data.get("original_boundary_faces", np.empty((0,), dtype=np.int64)))
        ob_lines = np.asarray(data.get("original_boundary_lines", np.empty((0,), dtype=np.int64)))
        if ob_faces.size > 0:
            dom.original_boundary = pv.PolyData(ob_pts, ob_faces)
        elif ob_lines.size > 0:
            dom.original_boundary = pv.PolyData(ob_pts, lines=ob_lines)
        else:
            dom.original_boundary = pv.PolyData(ob_pts)

    # Optional: interior mesh
    if "mesh_nodes" in data and pv is not None:
        m_nodes = np.asarray(data["mesh_nodes"], dtype=np.float64)
        m_verts = np.asarray(data["mesh_vertices"], dtype=np.int64)
        dom.mesh_nodes = m_nodes
        dom.mesh_vertices = m_verts
        n_cells = m_verts.shape[0]

        if points.shape[1] == 2:
            faces = np.hstack([np.full((n_cells, 1), 3, dtype=np.int64), m_verts])
            dom.mesh = pv.PolyData(m_nodes, faces)
            dom.mesh = dom.mesh.compute_cell_sizes()
            # Compute Normalized_Area if not present (compute_cell_sizes only creates 'Area')
            if 'Normalized_Area' not in dom.mesh.cell_data:
                total_area = np.sum(dom.mesh.cell_data['Area'])
                if total_area > 0:
                    dom.mesh.cell_data['Normalized_Area'] = dom.mesh.cell_data['Area'] / total_area
                else:
                    dom.mesh.cell_data['Normalized_Area'] = np.ones(n_cells) / n_cells
            # Add 'probability' alias for compatibility with older code paths
            dom.mesh.cell_data['probability'] = dom.mesh.cell_data['Normalized_Area']
            # Build spatial index for cell lookups
            cell_centers = dom.mesh.cell_centers().points[:, :points.shape[1]]
            dom.mesh_tree = cKDTree(cell_centers, leafsize=4)
            # Reuse cKDTree for radius queries instead of building a separate BallTree
            dom.mesh_tree_2 = dom.mesh_tree
            # Use numpy array instead of list for efficiency
            dom.all_mesh_cells = np.arange(n_cells, dtype=np.int64)
            dom.cumulative_probability = np.cumsum(dom.mesh.cell_data['Normalized_Area'])
            dom.characteristic_length = dom.mesh.area ** (1 / points.shape[1])
            dom.area = dom.mesh.area
            dom.volume = 0.0
        elif points.shape[1] == 3:
            # Build tetra UnstructuredGrid
            # Use numpy array for cell_types instead of list comprehension (much faster)
            cells = np.hstack([np.full((n_cells, 1), 4, dtype=np.int64), m_verts])
            cell_types = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
            dom.mesh = pv.UnstructuredGrid(cells, cell_types, m_nodes)
            dom.mesh = dom.mesh.compute_cell_sizes()
            # Compute Normalized_Volume if not present (compute_cell_sizes only creates 'Volume')
            if 'Normalized_Volume' not in dom.mesh.cell_data:
                total_volume = np.sum(dom.mesh.cell_data['Volume'])
                if total_volume > 0:
                    dom.mesh.cell_data['Normalized_Volume'] = dom.mesh.cell_data['Volume'] / total_volume
                else:
                    dom.mesh.cell_data['Normalized_Volume'] = np.ones(n_cells) / n_cells
            # Add 'probability' alias for compatibility with older code paths
            dom.mesh.cell_data['probability'] = dom.mesh.cell_data['Normalized_Volume']
            # Build spatial index for cell lookups
            cell_centers = dom.mesh.cell_centers().points[:, :points.shape[1]]
            dom.mesh_tree = cKDTree(cell_centers, leafsize=4)
            # Reuse cKDTree for radius queries instead of building a separate BallTree
            dom.mesh_tree_2 = dom.mesh_tree
            # Use numpy array instead of list for efficiency
            dom.all_mesh_cells = np.arange(n_cells, dtype=np.int64)
            dom.cumulative_probability = np.cumsum(dom.mesh.cell_data['Normalized_Volume'])
            dom.characteristic_length = dom.mesh.volume ** (1 / points.shape[1])
            dom.area = dom.mesh.area
            dom.volume = dom.mesh.volume

    return dom
