# Remeshing utility based on MMG executables

import os
import stat
import platform
import subprocess
import pyvista as pv
import pymeshfix
import meshio
import numpy
import numpy as np
from copy import deepcopy
import svv
from typing import Sequence, Optional

filepath = os.path.abspath(__file__)
dirpath = os.path.dirname(filepath)
modulepath = os.path.join(os.path.dirname(os.path.abspath(svv.__file__)), "bin")

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
    if platform.system() == 'Windows':
        if os.path.exists(modulepath + os.sep + "mmg2d_O3.exe"):
            _EXE_ = modulepath + os.sep + "mmg2d_O3.exe"
        else:
            _EXE_ = dirpath+os.sep+"Windows"+os.sep+"mmg2d_O3.exe"
    elif platform.system() == "Linux":
        if os.path.exists(modulepath + os.sep + "mmg2d_O3"):
            _EXE_ = modulepath + os.sep + "mmg2d_O3"
        else:
            _EXE_ = dirpath+os.sep+"Linux"+os.sep+"mmg2d_O3"
    elif platform.system() == "Darwin":
        if os.path.exists(modulepath + os.sep + "mmg2d_O3"):
            _EXE_ = modulepath + os.sep + "mmg2d_O3"
        else:
            _EXE_ = dirpath+os.sep+"Mac"+os.sep+"mmg2d_O3"
    else:
        raise NotImplementedError("Operating system not supported.")
    devnull = open(os.devnull, 'w')
    executable_list = [_EXE_, "tmp.mesh"]
    if ar is not None:
        executable_list.extend(["-ar", str(ar)])
    if hausd is not None:
        executable_list.extend(["-hausd", str(hausd)])
    if hgrad is not None:
        executable_list.extend(["-hgrad", str(hgrad)])
    if verbosity is not None:
        executable_list.extend(["-v", str(verbosity)])
    if hmax is not None:
        executable_list.extend(["-hmax", str(hmax)])
    if hmin is not None:
        executable_list.extend(["-hmin", str(hmin)])
    if hsiz is not None:
        executable_list.extend(["-hsiz", str(hsiz)])
    if noinsert:
        executable_list.extend(["-noinsert"])
    if nomove:
        executable_list.extend(["-nomove"])
    if nosurf:
        executable_list.extend(["-nosurf"])
    if noswap:
        executable_list.extend(["-noswap"])
    if nr:
        executable_list.extend(["-nr"])
    if optim:
        executable_list.extend(["-optim", str(optim)])
    if rn:
        executable_list.extend(["-rn"])
    if nsd:
        executable_list.extend(["-nsd"])
    if verbosity == 0:
        try:
            subprocess.check_call(executable_list, stdout=devnull, stderr=devnull)
        except:
            os.chmod(_EXE_, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            subprocess.check_call(executable_list, stdout=devnull, stderr=devnull)
    else:
        try:
            subprocess.check_call(executable_list)
        except:
            os.chmod(_EXE_, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            subprocess.check_call(executable_list)
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
        If True, attempts to automatically fix non-manifold issues in the remeshed surface using pymeshfix.
        Default is True.

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
        List of triangle indices that are required and should not be modified during remeshing.

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
    This function utilizes the MMGS executable to perform remeshing. The MMG executables must be present in
    the appropriate directory for your operating system.

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
    _mesh_ = pv.PolyData(pv_polydata_object.points, pv_polydata_object.faces)
    pv.save_meshio("tmp.mesh", _mesh_)
    if not isinstance(required_triangles, type(None)):
        add_required("tmp.mesh", required_triangles)
    if platform.system() == 'Windows':
        if os.path.exists(modulepath + os.sep + "mmgs_O3.exe"):
            _EXE_ = modulepath + os.sep + "mmgs_O3.exe"
        else:
            _EXE_ = dirpath+os.sep+"Windows"+os.sep+"mmgs_O3.exe"
    elif platform.system() == "Linux":
        if os.path.exists(modulepath + os.sep + "mmgs_O3"):
            _EXE_ = modulepath + os.sep + "mmgs_O3"
        else:
            _EXE_ = dirpath+os.sep+"Linux"+os.sep+"mmgs_O3"
    elif platform.system() == "Darwin":
        if os.path.exists(modulepath + os.sep + "mmgs_O3"):
            _EXE_ = modulepath + os.sep + "mmgs_O3"
        else:
            _EXE_ = dirpath+os.sep+"Mac"+os.sep+"mmgs_O3"
    else:
        raise NotImplementedError("Operating system not supported.")
    devnull = open(os.devnull, 'w')
    executable_list = [_EXE_, "tmp.mesh"]
    if ar is not None:
        executable_list.extend(["-ar", str(ar)])
    if hausd is not None:
        executable_list.extend(["-hausd", str(hausd)])
    if hgrad is not None:
        executable_list.extend(["-hgrad", str(hgrad)])
    if verbosity is not None:
        executable_list.extend(["-v", str(verbosity)])
    if hmax is not None:
        executable_list.extend(["-hmax", str(hmax)])
    if hmin is not None:
        executable_list.extend(["-hmin", str(hmin)])
    if hsiz is not None:
        executable_list.extend(["-hsiz", str(hsiz)])
    if noinsert:
        executable_list.extend(["-noinsert"])
    if nomove:
        executable_list.extend(["-nomove"])
    if nosurf:
        executable_list.extend(["-nosurf"])
    if noswap:
        executable_list.extend(["-noswap"])
    if nr:
        executable_list.extend(["-nr"])
    if optim:
        executable_list.extend(["-optim"])
    if rn:
        executable_list.extend(["-rn"])
    if verbosity == 0:
        try:
            subprocess.check_call(executable_list, stdout=devnull, stderr=devnull)
        except:
            os.chmod(_EXE_, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            subprocess.check_call(executable_list, stdout=devnull, stderr=devnull)
    else:
        try:
            subprocess.check_call(executable_list)
        except:
            os.chmod(_EXE_, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            subprocess.check_call(executable_list)
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
    if platform.system() == 'Windows':
        if os.path.exists(modulepath + os.sep + "mmg3d_O3.exe"):
            _EXE_ = modulepath + os.sep + "mmg3d_O3.exe"
        else:
            _EXE_ = dirpath+os.sep+"Windows"+os.sep+"mmg3d_O3.exe"
    elif platform.system() == "Linux":
        if os.path.exists(modulepath + os.sep + "mmg3d_O3"):
            _EXE_ = modulepath + os.sep + "mmg3d_O3"
        else:
            _EXE_ = dirpath+os.sep+"Linux"+os.sep+"mmg3d_O3"
    elif platform.system() == "Darwin":
        if os.path.exists(modulepath + os.sep + "mmg3d_O3"):
            _EXE_ = modulepath + os.sep + "mmg3d_O3"
        else:
            _EXE_ = dirpath+os.sep+"Mac"+os.sep+"mmg3d_O3"
    else:
        raise NotImplementedError("Operating system not supported.")
    devnull = open(os.devnull, 'w')
    executable_list = [_EXE_, "tmp.mesh"]
    if ar is not None:
        executable_list.extend(["-ar", str(ar)])
    if hausd is not None:
        executable_list.extend(["-hausd", str(hausd)])
    if hgrad is not None:
        executable_list.extend(["-hgrad", str(hgrad)])
    if verbosity is not None:
        executable_list.extend(["-v", str(verbosity)])
    if hmax is not None:
        executable_list.extend(["-hmax", str(hmax)])
    if hmin is not None:
        executable_list.extend(["-hmin", str(hmin)])
    if hsiz is not None:
        executable_list.extend(["-hsiz", str(hsiz)])
    if noinsert:
        executable_list.extend(["-noinsert"])
    if nomove:
        executable_list.extend(["-nomove"])
    if nosurf:
        executable_list.extend(["-nosurf"])
    if noswap:
        executable_list.extend(["-noswap"])
    if nr:
        executable_list.extend(["-nr"])
    if optim:
        executable_list.extend(["-optim"])
    if rn:
        executable_list.extend(["-rn"])
    if verbosity == 0:
        try:
            subprocess.check_call(executable_list, stdout=devnull, stderr=devnull)
        except:
            os.chmod(_EXE_, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            subprocess.check_call(executable_list, stdout=devnull, stderr=devnull)
    else:
        try:
            subprocess.check_call(executable_list)
        except:
            os.chmod(_EXE_, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            subprocess.check_call(executable_list)
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
    if platform.system() == 'Windows':
        if os.path.exists(modulepath + os.sep + "mmgs_O3.exe"):
            _EXE_ = modulepath + os.sep + "mmgs_O3.exe"
        else:
            _EXE_ = dirpath+os.sep+"Windows"+os.sep+"mmgs_O3.exe"
    elif platform.system() == "Linux":
        if os.path.exists(modulepath + os.sep + "mmgs_O3"):
            _EXE_ = modulepath + os.sep + "mmgs_O3"
        else:
            _EXE_ = dirpath+os.sep+"Linux"+os.sep+"mmgs_O3"
    elif platform.system() == "Darwin":
        if os.path.exists(modulepath + os.sep + "mmgs_O3"):
            _EXE_ = modulepath + os.sep + "mmgs_O3"
        else:
            _EXE_ = dirpath+os.sep+"Mac"+os.sep+"mmgs_O3"
    else:
        raise NotImplementedError("Operating system not supported.")
    devnull = open(os.devnull, 'w')
    executable_list = [_EXE_, "tmp.mesh", "-sol", "in.sol"]
    if ar is not None:
        executable_list.extend(["-ar", str(ar)])
    if hausd is not None:
        executable_list.extend(["-hausd", str(hausd)])
    if hgrad is not None:
        executable_list.extend(["-hgrad", str(hgrad)])
    if verbosity is not None:
        executable_list.extend(["-v", str(verbosity)])
    if hmax is not None:
        executable_list.extend(["-hmax", str(hmax)])
    if hmin is not None:
        executable_list.extend(["-hmin", str(hmin)])
    if hsiz is not None:
        executable_list.extend(["-hsiz", str(hsiz)])
    if noinsert is not None:
        executable_list.extend(["-noinsert"])
    if nomove is not None:
        executable_list.extend(["-nomove"])
    if nosurf is not None:
        executable_list.extend(["-nosurf"])
    if noswap is not None:
        executable_list.extend(["-noswap"])
    if nr is not None:
        executable_list.extend(["-nr"])
    if optim:
        executable_list.extend(["-optim"])
    if rn is not None:
        executable_list.extend(["-rn", str(rn)])
    if verbosity == 0:
        try:
            subprocess.check_call(executable_list, stdout=devnull, stderr=devnull)
        except:
            os.chmod(_EXE_, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            subprocess.check_call(executable_list, stdout=devnull, stderr=devnull)
    else:
        try:
            subprocess.check_call(executable_list)
        except:
            os.chmod(_EXE_, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            subprocess.check_call(executable_list)
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