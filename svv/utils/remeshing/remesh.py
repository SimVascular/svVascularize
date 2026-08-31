# Remeshing utility based on MMG executables

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pyvista as pv
import pymeshfix
import meshio
import numpy
import numpy as np
from copy import deepcopy
from typing import Sequence, Optional

from .mmg import run_mmg


# MMG rejects an input mesh when its normalized minimum element quality is
# below MMG5_NULKAL.  Keeping the same conservative floor here prevents
# mathematically degenerate triangles from reaching the executable without
# discarding merely low-quality triangles that MMG is intended to improve.
_MMG_MIN_TRIANGLE_QUALITY = 1.0e-30


def _copy_polygon_surface(surface):
    """Return a deep, polygon-only copy while preserving associated data."""
    points = numpy.asarray(surface.points)
    faces = numpy.asarray(surface.faces)
    copied = pv.PolyData(numpy.array(points, copy=True), numpy.array(faces, copy=True))

    for name in surface.point_data.keys():
        copied.point_data[name] = numpy.array(surface.point_data[name], copy=True)

    polygon_count = copied.n_cells
    polygon_start = int(surface.GetNumberOfVerts() + surface.GetNumberOfLines())
    for name in surface.cell_data.keys():
        values = numpy.asarray(surface.cell_data[name])
        if values.shape[0] != surface.n_cells:
            raise ValueError(
                f"Cell-data array {name!r} has {values.shape[0]} values for "
                f"{surface.n_cells} cells."
            )
        copied.cell_data[name] = numpy.array(
            values[polygon_start:polygon_start + polygon_count], copy=True
        )

    for name in surface.field_data.keys():
        copied.field_data[name] = numpy.array(surface.field_data[name], copy=True)

    return copied


def _triangle_qualities(points, triangles):
    """Compute the dimensionless isotropic triangle quality used by MMGS."""
    triangle_points = points[triangles]
    ab = triangle_points[:, 1] - triangle_points[:, 0]
    ac = triangle_points[:, 2] - triangle_points[:, 0]
    bc = triangle_points[:, 2] - triangle_points[:, 1]
    with numpy.errstate(over="ignore", invalid="ignore", divide="ignore"):
        denominator = (
            numpy.einsum("ij,ij->i", ab, ab)
            + numpy.einsum("ij,ij->i", ac, ac)
            + numpy.einsum("ij,ij->i", bc, bc)
        )
        quality = numpy.linalg.norm(numpy.cross(ab, ac), axis=1) / denominator
    return quality, denominator


def _normalize_required_triangles(required_triangles, triangle_count):
    """Validate MMG's one-based required-triangle indices."""
    if required_triangles is None:
        return None

    try:
        values = list(required_triangles)
    except TypeError as exc:
        raise ValueError("required_triangles must be a one-dimensional sequence of integers") from exc

    normalized = []
    for value in values:
        if isinstance(value, (bool, numpy.bool_)) or not isinstance(value, (int, numpy.integer)):
            raise ValueError("required_triangles must contain only integer indices")
        index = int(value)
        if index < 1 or index > triangle_count:
            raise ValueError(
                f"required triangle index {index} is outside the valid one-based range "
                f"[1, {triangle_count}]"
            )
        normalized.append(index)
    return normalized


def _prepare_surface_for_mmgs(
        surface, required_triangles=None, verbosity=1,
        quality_threshold=_MMG_MIN_TRIANGLE_QUALITY):
    """Validate and remove triangles that MMGS cannot accept.

    Point ordering and point count are deliberately preserved so an existing
    per-vertex Medit solution remains aligned with the surface.  Returned
    required-triangle indices use MMG's one-based convention.
    """
    if not hasattr(surface, "points") or not hasattr(surface, "faces"):
        raise TypeError("remesh_surface requires a polygonal surface with points and faces")
    if quality_threshold < 0.0 or not numpy.isfinite(quality_threshold):
        raise ValueError("quality_threshold must be a finite, non-negative value")

    mesh = _copy_polygon_surface(surface)
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise ValueError(
            f"Cannot remesh an empty surface (points={mesh.n_points}, triangles=0)."
        )

    points = numpy.asarray(mesh.points, dtype=float)
    original_points = numpy.array(points, copy=True)
    finite_points = numpy.all(numpy.isfinite(points), axis=1)
    if not numpy.all(finite_points):
        invalid_points = (numpy.flatnonzero(~finite_points) + 1).tolist()
        preview = invalid_points[:10]
        suffix = "..." if len(invalid_points) > len(preview) else ""
        raise ValueError(
            f"Surface contains {len(invalid_points)} point(s) with non-finite coordinates; "
            f"one-based point indices: {preview}{suffix}."
        )

    if required_triangles is not None and not mesh.is_all_triangles:
        raise ValueError(
            "remesh_surface(required_triangles=...) requires the input surface to be all triangles "
            "(so indices match). Triangulate the mesh first."
        )
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate(inplace=False)

    if mesh.n_cells == 0 or not mesh.is_all_triangles:
        raise ValueError("Surface triangulation did not produce any triangular faces.")
    if (
            mesh.n_points != original_points.shape[0]
            or not numpy.array_equal(numpy.asarray(mesh.points), original_points)):
        raise ValueError(
            "Surface triangulation changed point ordering, which would invalidate per-vertex sizing data."
        )
    points = numpy.asarray(mesh.points, dtype=float)

    face_records = numpy.asarray(mesh.faces).reshape(-1, 4)
    triangles = face_records[:, 1:].astype(numpy.int64, copy=False)
    triangle_count = triangles.shape[0]
    normalized_required = _normalize_required_triangles(required_triangles, triangle_count)

    if numpy.any(triangles < 0) or numpy.any(triangles >= mesh.n_points):
        raise ValueError("Surface contains a triangle with an out-of-range point index.")

    repeated_indices = (
        (triangles[:, 0] == triangles[:, 1])
        | (triangles[:, 0] == triangles[:, 2])
        | (triangles[:, 1] == triangles[:, 2])
    )
    quality, denominator = _triangle_qualities(points, triangles)

    valid = (
        ~repeated_indices
        & numpy.isfinite(quality)
        & (denominator > 0.0)
        & (quality > quality_threshold)
    )
    invalid_zero_based = numpy.flatnonzero(~valid)
    invalid_one_based = (invalid_zero_based + 1).tolist()

    if normalized_required is not None:
        invalid_required = [index for index in normalized_required if not valid[index - 1]]
        if invalid_required:
            raise ValueError(
                "Required triangle(s) are degenerate and cannot be passed to MMGS: "
                f"{invalid_required}."
            )

    finite_quality = quality[numpy.isfinite(quality)]
    minimum_quality = float(numpy.min(finite_quality)) if finite_quality.size else None

    if invalid_zero_based.size:
        retained_triangles = triangles[valid]
        if retained_triangles.shape[0] == 0:
            raise ValueError(
                f"Surface has no MMGS-compatible triangles; rejected all {triangle_count} faces."
            )

        vtk_faces = numpy.hstack((
            numpy.full((retained_triangles.shape[0], 1), 3, dtype=numpy.int64),
            retained_triangles,
        )).ravel()
        sanitized = pv.PolyData(numpy.array(mesh.points, copy=True), vtk_faces)
        for name in mesh.point_data.keys():
            sanitized.point_data[name] = numpy.array(mesh.point_data[name], copy=True)
        for name in mesh.cell_data.keys():
            sanitized.cell_data[name] = numpy.array(mesh.cell_data[name][valid], copy=True)
        for name in mesh.field_data.keys():
            sanitized.field_data[name] = numpy.array(mesh.field_data[name], copy=True)
        mesh = sanitized

        if normalized_required is not None:
            new_indices = numpy.cumsum(valid, dtype=numpy.int64)
            normalized_required = [int(new_indices[index - 1]) for index in normalized_required]

        if verbosity is not None and verbosity > 0:
            preview = invalid_one_based[:10]
            suffix = "..." if len(invalid_one_based) > len(preview) else ""
            print(
                f"Removed {len(invalid_one_based)} MMGS-incompatible triangle(s); "
                f"one-based indices: {preview}{suffix}."
            )

    retained_face_records = numpy.asarray(mesh.faces).reshape(-1, 4)
    retained_triangles = retained_face_records[:, 1:].astype(numpy.int64, copy=False)
    retained_quality, _ = _triangle_qualities(
        numpy.asarray(mesh.points, dtype=float), retained_triangles
    )
    if (
            not numpy.all(numpy.isfinite(retained_quality))
            or numpy.any(retained_quality <= quality_threshold)):
        raise ValueError("Surface still contains an MMGS-incompatible triangle after sanitization.")

    diagnostics = {
        "original_point_count": int(points.shape[0]),
        "original_triangle_count": int(triangle_count),
        "triangle_count": int(mesh.n_cells),
        "invalid_triangle_count": int(invalid_zero_based.size),
        "invalid_triangle_indices": invalid_one_based,
        "minimum_quality": minimum_quality,
        "sanitized_minimum_quality": float(numpy.min(retained_quality)),
        "quality_threshold": float(quality_threshold),
    }
    return mesh, normalized_required, diagnostics


def _surface_topology_diagnostics(surface):
    """Return best-effort topology values for a remeshing error message."""
    try:
        manifold = bool(surface.is_manifold)
    except Exception:
        manifold = None
    try:
        open_edges = int(surface.n_open_edges)
    except Exception:
        open_edges = None
    return manifold, open_edges

def remesh_surface_2d(boundary, autofix=False, ar=None, hausd=None, hgrad=None, verbosity=1,
                   hmax=None, hmin=None, hsiz=None, noinsert=None, nomove=None, nosurf=True,
                   noswap=None, nr=None, optim=None, rn=None, nsd=None):
    """
    Remeshes a 2D surface boundary using MMG2D.

    Parameters
    ----------
    boundary : list of pyvista.PolyData or pyvista.PolyData
        The boundary geometry to be remeshed. It can be a list of pyvista.PolyData objects representing boundaries,
        or a single pyvista.PolyData object.

    autofix : bool, optional
        If True, attempts to automatically fix non-manifold issues in the remeshed surface using pymeshfix.
        Default is False.

    ar : float, optional
        Anisotropy ratio. See MMG2D documentation for details.

    hausd : float, optional
        Control on Hausdorff distance. See MMG2D documentation.

    hgrad : float, optional
        Gradation parameter. See MMG2D documentation.

    verbosity : int, optional
        Verbosity level for MMG output. Default is 1.

    hmax : float, optional
        Maximum edge size. See MMG2D documentation.

    hmin : float, optional
        Minimum edge size. See MMG2D documentation.

    hsiz : bool or float, optional
        If True, automatically determines the average edge size for remeshing.
        If a float, uses the provided value as the edge size. Default is True.

    noinsert : bool, optional
        If True, prohibits node insertion. See MMG2D documentation.

    nomove : bool, optional
        If True, prohibits node movement. See MMG2D documentation.

    nosurf : bool, optional
        If True, prohibits surface modifications. Default is True.

    noswap : bool, optional
        If True, prohibits edge swapping. See MMG2D documentation.

    nr : bool, optional
        Disables reorientation of the mesh. See MMG2D documentation.

    optim : bool or float, optional
        Optimization parameter. See MMG2D documentation.

    rn : bool, optional
        Removes nonmanifold elements. See MMG2D documentation.

    nsd : bool, optional
        Non-strict Delaunay parameter. See MMG2D documentation.

    Returns
    -------
    remeshed_surface : pyvista.PolyData
        The remeshed surface as a pyvista PolyData object.

    Raises
    ------
    NotImplementedError
        If the remeshing process does not produce triangular faces.

    Notes
    -----
    This function utilizes the MMG2D executable to perform remeshing. The MMG executables must be present in
    the appropriate directory for your operating system.

    Examples
    --------
    **Example 1: Using 2D Points**

    .. code-block:: python
        import pyvista as pv
        boundary = pv.Circle()
        remeshed = remesh_surface_2d(boundary, hmax=0.1)
    """
    #_mesh_ = pv.PolyData(pv_polydata_object.points, pv_polydata_object.faces)
    #pv.save_meshio("tmp.mesh", _mesh_)
    if isinstance(boundary, list):
        hsizes = []
        pts = []
        pts_markers = []
        lines_markers = []
        all_lines = []
        full_pts = []
        count = 0
        for i in range(len(boundary)):
            hsize = boundary[i].compute_cell_sizes(length=True).cell_data["Length"].mean()
            hsizes.append(hsize)
            if hsiz:
                hsiz = hsize
            else:
                hsiz = None
            triangulated = boundary[i].delaunay_2d()
            triangulated_quality = triangulated.compute_cell_quality().cell_data["CellQuality"]
            best = numpy.argmax(triangulated_quality)
            normals = triangulated.compute_normals(cell_normals=True, point_normals=True).cell_data["Normals"]
            normals = normals / numpy.linalg.norm(normals, axis=1).reshape(-1, 1)
            normals = normals[best]
            full_pts.append(boundary[i].points)
            z_axis = numpy.array([0, 0, 1])
            rotation_axis = numpy.cross(normals, z_axis)
            rotation_axis = rotation_axis / numpy.linalg.norm(rotation_axis).reshape(-1, 1)
            rotation_angle = numpy.degrees(numpy.arccos(numpy.dot(normals, z_axis)/(numpy.linalg.norm(normals)*numpy.linalg.norm(z_axis))))
            rotated_boundary = boundary[i].rotate_vector(rotation_axis.flatten(), rotation_angle, point=boundary[i].center, inplace=False)
            points = rotated_boundary.points
            z_values = deepcopy(points[:, 2].mean())
            points = points[:, :2]
            pts.append(points)
            pt_markers = numpy.ones(points.shape[0], dtype=int)*(i+1)
            pts_markers.append(pt_markers)
            try:
                lines = {'line': boundary[i].cells.reshape(-1, 3)[:, 1:].copy()}
            except:
                lines = {'line': boundary[i].lines.reshape(-1, 3)[:, 1:].copy()}
            lines['line'] += count
            all_lines.append(lines['line'])
            line_markers = numpy.ones(lines['line'].shape[0], dtype=int) #*(i+1)
            lines_markers.append(line_markers)
            count += points.shape[0]
        points = numpy.vstack(pts)
        full_pts = numpy.vstack(full_pts)
        points_markers = numpy.hstack(pts_markers)
        lines = {'line': numpy.vstack(all_lines)}
        lines_markers = numpy.hstack(lines_markers)
        mesh = meshio.Mesh(points, lines, point_data={"markers": points_markers}, cell_data={"a": [lines_markers]})
        meshio.write("tmp.mesh", mesh)
    elif isinstance(boundary, pv.PolyData):
        if boundary.is_all_triangles:
            triangulated_quality = boundary.compute_cell_quality().cell_data["CellQuality"]
            best = numpy.argmax(triangulated_quality)
            normals = boundary.compute_normals(cell_normals=True, point_normals=True).cell_data["Normals"]
            normals = normals / numpy.linalg.norm(normals, axis=1).reshape(-1, 1)
            normals = normals[best]
            full_pts = boundary.points
            z_axis = numpy.array([0, 0, 1])
            rotation_axis = numpy.cross(normals, z_axis)
            rotation_axis = rotation_axis / numpy.linalg.norm(rotation_axis).reshape(-1, 1)
            rotation_angle = numpy.degrees(
                numpy.arccos(numpy.dot(normals, z_axis) / (numpy.linalg.norm(normals) * numpy.linalg.norm(z_axis))))
            rotated_boundary = boundary.rotate_vector(rotation_axis.flatten(), rotation_angle,
                                                         point=boundary.center, inplace=False)
            points = rotated_boundary.points
            z_values = deepcopy(points[:, 2].mean())
            points = points[:, :2]
            faces = {'triangle': boundary.faces.reshape(-1, 4)[:, 1:].copy()}
            mesh = meshio.Mesh(points, faces)
            meshio.write("tmp.mesh", mesh)
            boundary = [boundary]
        else:
            hsizes = []
            hsize = boundary.compute_cell_sizes(length=True).cell_data["Length"].mean()
            hsizes.append(hsize)
            if hsiz:
                hsiz = hsize
            else:
                hsiz = None
            triangulated = boundary.delaunay_2d()
            triangulated_quality = triangulated.compute_cell_quality().cell_data["CellQuality"]
            best = numpy.argmax(triangulated_quality)
            normals = triangulated.compute_normals(cell_normals=True, point_normals=True).cell_data["Normals"]
            normals = normals / numpy.linalg.norm(normals, axis=1).reshape(-1, 1)
            normals = normals[best]
            full_pts = boundary.points
            z_axis = numpy.array([0, 0, 1])
            rotation_axis = numpy.cross(normals, z_axis)
            rotation_axis = rotation_axis / numpy.linalg.norm(rotation_axis).reshape(-1, 1)
            rotation_angle = numpy.degrees(
                numpy.arccos(numpy.dot(normals, z_axis) / (numpy.linalg.norm(normals) * numpy.linalg.norm(z_axis))))
            rotated_boundary = boundary.rotate_vector(rotation_axis.flatten(), rotation_angle,
                                                         point=boundary.center, inplace=False)
            points = rotated_boundary.points
            z_values = deepcopy(points[:, 2].mean())
            points = points[:, :2]
            try:
                lines = {'line': boundary.cells.reshape(-1, 3)[:, 1:].copy()}
            except:
                lines = {'line': boundary.lines.reshape(-1, 3)[:, 1:].copy()}
            mesh = meshio.Mesh(points, lines)
            meshio.write("tmp.mesh", mesh)
            boundary = [boundary]
    elif isinstance(boundary, pv.UnstructuredGrid):
        triangulated = boundary.delaunay_2d()
        triangulated_quality = triangulated.compute_cell_quality().cell_data["CellQuality"]
        best = numpy.argmax(triangulated_quality)
        normals = triangulated.compute_normals(cell_normals=True, point_normals=True).cell_data["Normals"]
        normals = normals / numpy.linalg.norm(normals, axis=1).reshape(-1, 1)
        normals = normals[best]
        full_pts = boundary.points
        z_axis = numpy.array([0, 0, 1])
        rotation_axis = numpy.cross(normals, z_axis)
        rotation_axis = rotation_axis / numpy.linalg.norm(rotation_axis).reshape(-1, 1)
        rotation_angle = numpy.degrees(
            numpy.arccos(numpy.dot(normals, z_axis) / (numpy.linalg.norm(normals) * numpy.linalg.norm(z_axis))))
        rotated_boundary = boundary.rotate_vector(rotation_axis.flatten(), rotation_angle,
                                                  point=boundary.center, inplace=False)
        points = rotated_boundary.points
        z_values = deepcopy(points[:, 2].mean())
        points = points[:, :2]
        #faces = {'triangle': boundary.faces.reshape(-1, 4)[:, 1:].copy()}
        try:
            lines = {'line': boundary.cells.reshape(-1, 3)[:, 1:].copy()}
        except:
            lines = {'line': boundary.lines.reshape(-1, 3)[:, 1:].copy()}
        mesh = meshio.Mesh(points, lines)
        meshio.write("tmp.mesh", mesh)
        boundary = [boundary]
    args = ["tmp.mesh"]
    if ar is not None:
        args.extend(["-ar", str(ar)])
    if hausd is not None:
        args.extend(["-hausd", str(hausd)])
    if hgrad is not None:
        args.extend(["-hgrad", str(hgrad)])
    if verbosity is not None:
        args.extend(["-v", str(verbosity)])
    if hmax is not None:
        args.extend(["-hmax", str(hmax)])
    if hmin is not None:
        args.extend(["-hmin", str(hmin)])
    if hsiz is not None:
        args.extend(["-hsiz", str(hsiz)])
    if noinsert is not None:
        args.extend(["-noinsert"])
    if nomove is not None:
        args.extend(["-nomove"])
    if nosurf is not None:
        args.extend(["-nosurf"])
    if noswap is not None:
        args.extend(["-noswap"])
    if nr is not None:
        args.extend(["-nr"])
    if optim is not None:
        args.extend(["-optim", str(optim)])
    if rn is not None:
        args.extend(["-rn", str(rn)])
    if nsd is not None:
        args.extend(["-nsd", str(nsd)])
    if verbosity == 0:
        run_mmg("mmg2d", args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        run_mmg("mmg2d", args)
    clean_medit("tmp.o.mesh")
    remesh_data = meshio.read("tmp.o.mesh")
    vertices = remesh_data.points
    has_triangles = False
    for cell_block in remesh_data.cells:
        if cell_block.type == "triangle":
            faces = cell_block.data
            has_triangles = True
            break
    if not has_triangles:
        raise NotImplementedError("Only triangular surfaces are supported.")
    remeshed_points = numpy.zeros((vertices.shape[0], 3))
    remeshed_points[:, :2] = vertices
    remeshed_points[:, 2] = z_values
    #remeshed_points = rotation.inv().apply(remeshed_points)
    #remeshed_points[:boundary.points.shape[0], :] = boundary.points
    faces = numpy.hstack([numpy.full((faces.shape[0], 1), 3), faces])
    remeshed_surface = pv.PolyData(remeshed_points, faces.flatten())
    remeshed_surface = remeshed_surface.rotate_vector(rotation_axis.flatten(), -rotation_angle, point=boundary[0].center, inplace=False)
    #remeshed_surface.points[:full_pts.shape[0], :] = full_pts
    if autofix:
        if not remeshed_surface.is_manifold:
            fix = pymeshfix.MeshFix(remeshed_surface)
            if verbosity == 0:
                fix.repair(verbose=False)
            fix.repair(verbose=True)
            remeshed_surface = fix.mesh
    os.remove("tmp.mesh")
    os.remove("tmp.o.sol")
    os.remove("tmp.o.mesh")
    return remeshed_surface


def remesh_surface(pv_polydata_object, autofix=True, ar=None, hausd=None, hgrad=None, verbosity=1,
                   hmax=None, hmin=None, hsiz=None, noinsert=None, nomove=None, nosurf=None,
                   noswap=None, nr=None, optim=False, rn=None, required_triangles=None):
    """
    Remeshes a 3D surface using MMGS.

    Parameters
    ----------
    pv_polydata_object : pyvista.PolyData
        The 3D surface mesh to be remeshed.

    autofix : bool, optional
        If True, attempts to fix non-manifold issues in the MMGS output using
        pymeshfix. Invalid input triangles are removed before MMGS regardless
        of this setting. Default is True.

    ar : float, optional
        Anisotropy ratio. See MMGS documentation for details.

    hausd : float, optional
        Control on Hausdorff distance. See MMGS documentation.

    hgrad : float, optional
        Gradation parameter. See MMGS documentation.

    verbosity : int, optional
        Verbosity level for MMG output. Default is 1.

    hmax : float, optional
        Maximum edge size. See MMGS documentation.

    hmin : float, optional
        Minimum edge size. See MMGS documentation.

    hsiz : float, optional
        Size parameter for remeshing. See MMGS documentation.

    noinsert : bool, optional
        If True, prohibits node insertion. See MMGS documentation.

    nomove : bool, optional
        If True, prohibits node movement. See MMGS documentation.

    nosurf : bool, optional
        If True, prohibits surface modifications. See MMGS documentation.

    noswap : bool, optional
        If True, prohibits edge swapping. See MMGS documentation.

    nr : bool, optional
        Disables reorientation of the mesh. See MMGS documentation.

    optim : bool, optional
        Optimization parameter. Default is False. See MMGS documentation.

    rn : bool, optional
        Removes nonmanifold elements. See MMGS documentation.

    required_triangles : list of int, optional
        One-based triangle indices that are required and should not be modified
        during remeshing. Indices are remapped if earlier invalid faces are
        removed. An invalid required face raises ``ValueError``.

    Returns
    -------
    remeshed_surface : pyvista.PolyData
        The remeshed surface as a pyvista PolyData object.

    Raises
    ------
    ValueError
        If the input is empty, contains non-finite points, has no
        MMGS-compatible triangles, or protects an invalid triangle.
    NotImplementedError
        If the remeshing process does not produce triangular faces.
    RuntimeError
        If MMGS rejects the sanitized surface.

    Notes
    -----
    This function removes only triangles that fail MMGS's minimum input-quality
    requirement before invoking the MMGS executable. The MMG executables must
    be present in the appropriate directory for your operating system.

    References
    ----------
    .. [1] Dapogny, C., Dobrzynski, C., & Frey, P. J. (2014). Three-dimensional adaptive domain
           remeshing, implicit domain meshing, and applications to free and moving boundary
           problems. *Journal of Computational Physics, 262*, 358-378. doi:10.1016/j.jcp.2014.01.005

    Examples
    --------
    **Example 1: Remeshing a Circular Boundary**

    .. code-block:: python

        import pyvista as pv
        boundary = pv.Circle()
        remeshed = remesh_surface_2d(boundary, hmax=0.1)

    **Example 2: Remeshing with Multiple Boundaries**

    .. code-block:: python

        import pyvista as pv
        boundary1 = pv.Circle(radius=1.0)
        boundary2 = pv.Circle(radius=0.5).translate([1, 1, 0])
        remeshed = remesh_surface_2d([boundary1, boundary2], autofix=True)

    **Example 3: Using Advanced MMG Parameters**

    .. code-block:: python

        import pyvista as pv
        boundary = pv.Circle()
        remeshed = remesh_surface_2d(boundary, hmax=0.2, hmin=0.05, hausd=0.01, verbosity=3)
    """
    _mesh_, required_triangles, input_diagnostics = _prepare_surface_for_mmgs(
        pv_polydata_object,
        required_triangles=required_triangles,
        verbosity=verbosity,
    )

    # Run MMG in an isolated temp directory to avoid permission issues and temp-file collisions.
    tmp_root = None
    if os.name == "nt":
        for env_var in ("TEMP", "TMP"):
            candidate = os.environ.get(env_var)
            if candidate and os.path.isdir(candidate):
                tmp_root = candidate
                break

    sol_src = Path("in.sol")
    sol_used = False
    mmg_succeeded = False
    with tempfile.TemporaryDirectory(prefix="svv_remesh_", dir=tmp_root) as tmpdir:
        tmpdir_path = Path(tmpdir)
        mesh_path = tmpdir_path / "tmp.mesh"
        pv.save_meshio(str(mesh_path), _mesh_, file_format="medit")
        if required_triangles is not None:
            add_required(str(mesh_path), required_triangles)

        args = ["tmp.mesh"]
        # If caller prepared a sizing function file in the working directory,
        # detect and pass it through to MMG.
        if sol_src.is_file():
            shutil.copy2(sol_src, tmpdir_path / "in.sol")
            sol_used = True
            args.extend(["-sol", "in.sol"])

        if ar is not None:
            args.extend(["-ar", str(ar)])
        if hausd is not None:
            args.extend(["-hausd", str(hausd)])
        if hgrad is not None:
            args.extend(["-hgrad", str(hgrad)])
        if verbosity is not None:
            args.extend(["-v", str(verbosity)])
        if hmax is not None:
            args.extend(["-hmax", str(hmax)])
        if hmin is not None:
            args.extend(["-hmin", str(hmin)])
        if hsiz is not None:
            args.extend(["-hsiz", str(hsiz)])
        if noinsert is not None:
            args.extend(["-noinsert"])
        if nomove is not None:
            args.extend(["-nomove"])
        if nosurf is not None:
            args.extend(["-nosurf"])
        if noswap is not None:
            args.extend(["-noswap"])
        if nr is not None:
            args.extend(["-nr"])
        if optim:
            args.extend(["-optim"])
        if rn is not None:
            args.extend(["-rn", str(rn)])

        try:
            if verbosity == 0:
                run_mmg("mmgs", args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=tmpdir)
            else:
                run_mmg("mmgs", args, cwd=tmpdir)
        except subprocess.CalledProcessError as exc:
            manifold, open_edges = _surface_topology_diagnostics(_mesh_)
            invalid_indices = input_diagnostics["invalid_triangle_indices"][:10]
            raise RuntimeError(
                "MMGS surface remeshing failed after input sanitization "
                f"(points={_mesh_.n_points}, triangles={_mesh_.n_cells}, "
                f"removed_invalid_triangles={input_diagnostics['invalid_triangle_count']}, "
                f"invalid_triangle_indices={invalid_indices}, "
                f"minimum_retained_quality={input_diagnostics['sanitized_minimum_quality']:.6e}, "
                f"manifold={manifold}, open_edges={open_edges}, args={args!r}, "
                f"returncode={exc.returncode})."
            ) from exc
        mmg_succeeded = True

        out_mesh_path = tmpdir_path / "tmp.o.mesh"
        clean_medit(str(out_mesh_path))
        remesh_data = meshio.read(str(out_mesh_path))
        vertices = remesh_data.points
        has_triangles = False
        for cell_block in remesh_data.cells:
            if cell_block.type == "triangle":
                faces = cell_block.data
                has_triangles = True
                break
        if not has_triangles:
            raise NotImplementedError("Only triangular surfaces are supported.")
        faces = numpy.hstack([numpy.full((faces.shape[0], 1), 3), faces])
        remeshed_surface = pv.PolyData(vertices, faces.flatten())
        if autofix:
            if not remeshed_surface.is_manifold:
                fix = pymeshfix.MeshFix(remeshed_surface)
                fix.repair(verbose=verbosity != 0)
                remeshed_surface = fix.mesh

    # Clean up sizing file if provided (historical behavior)
    if mmg_succeeded and sol_used:
        try:
            sol_src.unlink()
        except Exception:
            pass
    return remeshed_surface


def remesh_volume(pv_unstructured_mesh, auto=True, ar=None, hausd=None, hgrad=None, verbosity=1,
                  hmax=None, hmin=None, hsiz=None, noinsert=None, nomove=None, nosurf=None,
                  noswap=None, nr=None, optim=None, rn=None):
    """
    Remeshes a 3D volume mesh using MMG3D.

    Parameters
    ----------
    pv_unstructured_mesh : pyvista.UnstructuredGrid
        The 3D volume mesh to be remeshed.

    auto : bool, optional
        If True, attempts to automatically fix issues in the remeshed volume. Default is True.

    ar : float, optional
        Anisotropy ratio. See MMG3D documentation for details.

    hausd : float, optional
        Control on Hausdorff distance. See MMG3D documentation.

    hgrad : float, optional
        Gradation parameter. See MMG3D documentation.

    verbosity : int, optional
        Verbosity level for MMG output. Default is 1.

    hmax : float, optional
        Maximum edge size. See MMG3D documentation.

    hmin : float, optional
        Minimum edge size. See MMG3D documentation.

    hsiz : float, optional
        Size parameter for remeshing. See MMG3D documentation.

    noinsert : bool, optional
        If True, prohibits node insertion. See MMG3D documentation.

    nomove : bool, optional
        If True, prohibits node movement. See MMG3D documentation.

    nosurf : bool, optional
        If True, prohibits surface modifications. See MMG3D documentation.

    noswap : bool, optional
        If True, prohibits edge swapping. See MMG3D documentation.

    nr : bool, optional
        Disables reorientation of the mesh. See MMG3D documentation.

    optim : bool, optional
        Optimization parameter. See MMG3D documentation.

    rn : bool, optional
        Removes nonmanifold elements. See MMG3D documentation.

    Returns
    -------
    remeshed_volume : pyvista.UnstructuredGrid
        The remeshed volume as a pyvista UnstructuredGrid object.

    Raises
    ------
    NotImplementedError
        If the remeshing process does not produce tetrahedral elements.

    Notes
    -----
    This function utilizes the MMG3D executable to perform remeshing. The MMG executables must be present in
    the appropriate directory for your operating system.

    Examples
    --------
    >>> import pyvista as pv
    >>> cube = pv.Cube().triangulate().extract_cells(range(12))
    >>> volume_mesh = pv.UnstructuredGrid(cube)
    >>> remeshed_volume = remesh_volume(volume_mesh, hmax=0.1)
    """
    pv.save_meshio("tmp.mesh", pv_unstructured_mesh)
    args = ["tmp.mesh"]
    if ar is not None:
        args.extend(["-ar", str(ar)])
    if hausd is not None:
        args.extend(["-hausd", str(hausd)])
    if hgrad is not None:
        args.extend(["-hgrad", str(hgrad)])
    if verbosity is not None:
        args.extend(["-v", str(verbosity)])
    if hmax is not None:
        args.extend(["-hmax", str(hmax)])
    if hmin is not None:
        args.extend(["-hmin", str(hmin)])
    if hsiz is not None:
        args.extend(["-hsiz", str(hsiz)])
    if noinsert is not None:
        args.extend(["-noinsert"])
    if nomove is not None:
        args.extend(["-nomove"])
    if nosurf is not None:
        args.extend(["-nosurf"])
    if noswap is not None:
        args.extend(["-noswap"])
    if nr is not None:
        args.extend(["-nr"])
    if optim is not None:
        args.extend(["-optim", str(optim)])
    if rn is not None:
        args.extend(["-rn", str(rn)])
    if verbosity == 0:
        run_mmg("mmg3d", args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        run_mmg("mmg3d", args)
    clean_medit("tmp.o.mesh")
    remeshed_data = meshio.read("tmp.o.mesh")
    vertices = remeshed_data.points
    has_tetrahedra = False
    for cell_block in remeshed_data.cells:
        if cell_block.type == "tetra":
            tets = cell_block.data
            has_tetrahedra = True
            break
    if not has_tetrahedra:
        raise NotImplementedError("Only tetrahedral volume elements are supported.")
    tets = numpy.hstack([numpy.full((tets.shape[0], 1), 4), tets])
    cell_types = [pv.CellType.TETRA for i in range(tets.shape[0])]
    remeshed_volume = pv.UnstructuredGrid(tets, cell_types, vertices)
    os.remove("tmp.mesh")
    os.remove("tmp.o.sol")
    os.remove("tmp.o.mesh")
    return remeshed_volume

def add_required(file_path, triangle_indices):
    """
    Appends a 'RequiredTriangles' section with specified triangle indices to a .mesh file.

    Parameters
    ----------
    file_path : str
        Path to the .mesh file.

    triangle_indices : list of int
        List of triangle indices to mark as required.

    Notes
    -----
    This function modifies the .mesh file in-place by adding a 'RequiredTriangles' section.
    The indices should correspond to the triangle elements in the mesh file.

    Examples
    --------
    >>> triangle_indices = [1, 2, 3]
    >>> add_required('mesh.mesh', triangle_indices)
    """
    try:
        # Read the existing content from the file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Find the position of the 'End' line
        end_index = lines.index("End\n") if "End\n" in lines else len(lines)

        # Prepare the 'RequiredTriangles' section
        required_triangles_section = ["RequiredTriangles\n", f"{len(triangle_indices)}\n"] + \
                                     [f"{index}\n" for index in triangle_indices] + ["\n"]
        new_lines = lines[:end_index] + required_triangles_section + lines[end_index:]
        with open(file_path, 'w') as file:
            file.writelines(new_lines)

    except FileNotFoundError:
        print(f"Error: The file at {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")


def clean_medit(filename):
    """
    Cleans up a .mesh file by removing unsupported or unnecessary keywords.

    Parameters
    ----------
    filename : str
        The name of the .mesh file to clean.

    Notes
    -----
    This function reads the specified .mesh file and writes back a cleaned version,
    removing sections that are not supported or required by MMG.

    Examples
    --------
    >>> clean_medit('mesh.mesh')
    """
    file = open(filename)
    lines = file.readlines()
    file.close()
    keywords_index = []
    for i, s in enumerate(lines):
        if s[0].isnumeric():
            pass
        elif s[0] == '-':
            pass
        elif s[0] == '\n':
            pass
        elif s[0] == '\n':
            pass
        else:
            keywords_index.append(i)
    new_file = open(filename, 'w+')
    new_lines = []
    for i,o in enumerate(keywords_index):
        if lines[o] == 'RequiredVertices\n':
            pass
        elif lines[o] == 'Ridges\n':
            pass
        elif lines[o] == 'Tangents\n':
            pass
        elif lines[o] == 'TangentAtVertices\n':
            pass
        elif lines[o] == 'RequiredTriangles\n':
            pass
        elif lines[o] == 'RequiredEdges\n':
            pass
        else:
            if o == keywords_index[-1]:
                new_lines.append(lines[o])
            else:
                new_lines.extend(lines[o:keywords_index[i+1]])
    new_file.writelines(new_lines)
    new_file.close()

def write_medit_sol(mesh: pv.PolyData, path: str, array_name="MeshSizingFunction",
                    scale=1, default_size=None):
    npts = mesh.n_points
    vals = None
    if array_name in mesh.point_data:
        vals = np.asarray(mesh.point_data[array_name], dtype=float).reshape(-1)
        if vals.size != npts:
            raise RuntimeError(f"Array '{array_name}' length ({vals.size}) "
                               f"!= number of points ({npts})")
          # Replace non-positive entries if default provided
        if default_size is not None:
            vals = np.where(vals > 0.0, vals, float(default_size))
    else:
        if default_size is None:
            raise RuntimeError(f"Point-data array '{array_name}' not found and no default_size provided.")
        vals = np.full(npts, float(default_size), dtype=float)

    vals = scale * vals  # SV typically scales by ~0.8 before MMG

    with open(path, "w") as f:
        f.write("MeshVersionFormatted 2\n")
        f.write("Dimension 3\n\n")
        f.write("SolAtVertices\n")
        f.write(f"{npts}\n")
        f.write("1 1\n")  # one scalar per vertex
        for v in vals:
            f.write(f"{v:.15g}\n")
        f.write("\nEnd\n")

def sphere_refinement(
      mesh: pv.PolyData,
      radius: float,
      center: Sequence[float],
      local_edge_size: float,
      global_edge_size: float,
      array_name: str = "MeshSizingFunction",
      refine_id_name: Optional[str] = None,
      refine_id_value: int = 1,
      inplace: bool = True,
      ar=None,
      hausd=None,
      hgrad=None,
      verbosity=1,
      hmax=None,
      hmin=None,
      hsiz=None,
      noinsert=None,
      nomove=None,
      nosurf=None,
      noswap=None,
      nr=None,
      optim=False,
      rn=None,
      required_triangles=None
  ) -> pv.PolyData:
    """
    Set local mesh edge size for points inside a sphere.

    Args:

    mesh: pyvista.PolyData surface mesh (triangulated or not).

    radius: Sphere radius (> 0).

    center: Sphere center [cx, cy, cz].

    local_edge_size: Target edge size to assign inside the sphere (> 0).

    array_name: Point-data array name to write (default: 'MeshSizingFunction').

    global_edge_size: If provided and the array is missing, initialize all points
          to this value. If not provided and the array is missing, initialize with zeros.
          Points outside the sphere are left unchanged.

    refine_id_name: Optional point-data int array to tag refined points
          (e.g., 'RefineID'). If provided, sets tag = refine_id_value inside the sphere,
          leaves others as-is (initializes to 0 if array missing).

    refine_id_value: Tag value to set in refine_id_name for points in the sphere.

    inplace: If False, process a deep copy and return it.

    ar : float, optional
        Anisotropy ratio. See MMG3D documentation for details.

    hausd : float, optional
        Control on Hausdorff distance. See MMG3D documentation.

    hgrad : float, optional
        Gradation parameter. See MMG3D documentation.

    verbosity : int, optional
        Verbosity level for MMG output. Default is 1.

    hmax : float, optional
        Maximum edge size. See MMG3D documentation.

    hmin : float, optional
        Minimum edge size. See MMG3D documentation.

    hsiz : float, optional
        Size parameter for remeshing. See MMG3D documentation.

    noinsert : bool, optional
        If True, prohibits node insertion. See MMG3D documentation.

    nomove : bool, optional
        If True, prohibits node movement. See MMG3D documentation.

    nosurf : bool, optional
        If True, prohibits surface modifications. See MMG3D documentation.

    noswap : bool, optional
        If True, prohibits edge swapping. See MMG3D documentation.

    nr : bool, optional
        Disables reorientation of the mesh. See MMG3D documentation.

    optim : bool, optional
        Optimization parameter. See MMG3D documentation.

    rn : bool, optional
        Removes nonmanifold elements. See MMG3D documentation.
      Returns:
        pv.PolyData: The updated mesh (same object if inplace=True).
    """
    if not isinstance(mesh, pv.PolyData):
        raise TypeError("mesh must be a pyvista.PolyData")
    if radius <= 0:
        raise ValueError("radius must be > 0")
    if local_edge_size <= 0:
        raise ValueError("local_edge_size must be > 0")
    if global_edge_size <= 0:
        raise ValueError("global_edge_size must be > 0")

    out = mesh if inplace else mesh.copy(deep=True)

    pts = out.points.astype(float)
    ctr = np.asarray(center, dtype=float).reshape(3)
    if ctr.shape != (3,):
        raise ValueError("center must be a sequence of three floats")

    # Compute mask of points inside the sphere (vectorized).
    d2 = np.einsum("ij,ij->i", pts - ctr, pts - ctr)  # squared distance
    mask = d2 <= float(radius) ** 2

    # Prepare or fetch the sizing array.
    n = pts.shape[0]
    if array_name in out.point_data:
        sizes = np.asarray(out.point_data[array_name], dtype=float).copy()
        if sizes.shape[0] != n:
            raise RuntimeError(f"Existing array '{array_name}' length {sizes.shape[0]} != n_points {n}")
    else:
        if global_edge_size is None:
            sizes = np.zeros(n, dtype=float)
        else:
            sizes = np.full(n, float(global_edge_size), dtype=float)

    # Apply refinement.
    sizes[mask] = float(local_edge_size)
    out.point_data[array_name] = sizes

    # Optional: tag refined points (like SimVascular's RefineID).
    if refine_id_name:
        if refine_id_name in out.point_data:
            rid = np.asarray(out.point_data[refine_id_name], dtype=np.int32).copy()
            if rid.shape[0] != n:
                raise RuntimeError(f"Existing array '{refine_id_name}' length {rid.shape[0]} != n_points {n}")
        else:
            rid = np.zeros(n, dtype=np.int32)
        rid[mask] = int(refine_id_value)
        out.point_data[refine_id_name] = rid
    write_medit_sol(out, "in.sol", array_name = "MeshSizingFunction",scale = 1, default_size = global_edge_size)
    pv.save_meshio("tmp.mesh", out)
    if not isinstance(required_triangles, type(None)):
        add_required("tmp.mesh", required_triangles)
    args = ["tmp.mesh", "-sol", "in.sol"]
    if ar is not None:
        args.extend(["-ar", str(ar)])
    if hausd is not None:
        args.extend(["-hausd", str(hausd)])
    if hgrad is not None:
        args.extend(["-hgrad", str(hgrad)])
    if verbosity is not None:
        args.extend(["-v", str(verbosity)])
    if hmax is not None:
        args.extend(["-hmax", str(hmax)])
    if hmin is not None:
        args.extend(["-hmin", str(hmin)])
    if hsiz is not None:
        args.extend(["-hsiz", str(hsiz)])
    if noinsert is not None:
        args.extend(["-noinsert"])
    if nomove is not None:
        args.extend(["-nomove"])
    if nosurf is not None:
        args.extend(["-nosurf"])
    if noswap is not None:
        args.extend(["-noswap"])
    if nr is not None:
        args.extend(["-nr"])
    if optim:
        args.extend(["-optim"])
    if rn is not None:
        args.extend(["-rn", str(rn)])
    if verbosity == 0:
        run_mmg("mmgs", args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        run_mmg("mmgs", args)
    clean_medit("tmp.o.mesh")
    remesh_data = meshio.read("tmp.o.mesh")
    vertices = remesh_data.points
    has_triangles = False
    for cell_block in remesh_data.cells:
        if cell_block.type == "triangle":
            faces = cell_block.data
            has_triangles = True
            break
    if not has_triangles:
        raise NotImplementedError("Only triangular surfaces are supported.")
    faces = numpy.hstack([numpy.full((faces.shape[0], 1), 3), faces])
    remeshed_surface = pv.PolyData(vertices, faces.flatten())
    os.remove("tmp.mesh")
    os.remove("tmp.o.sol")
    os.remove("tmp.o.mesh")
    os.remove("in.sol")
    return remeshed_surface
