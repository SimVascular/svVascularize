import numpy
import pyvista
import pymeshfix
from scipy.interpolate import splprep, splev
from svv.domain.routines.boolean import boolean
from svv.simulation.utils.extract_faces import extract_faces
from svv.utils.remeshing.remesh import remesh_surface_2d, remesh_surface


def smooth_junctions(mesh):
    """
    Smooth the junctions of a PolyData mesh using PyVista.
    Parameters:
    - mesh (pv.PolyData): The input surface mesh.
    Returns:
    - pv.PolyData: The mesh with the junctions smoothed.
    """
    # Extract the faces of the mesh
    faces, walls, caps, shared_boundaries = extract_faces(mesh, None)
    hsize = mesh.hsize
    # If there is only one wall face then proceed with smoothing
    if len(walls) == 1:
        # Smooth the mesh
        smoothed_mesh = walls[0].smooth_taubin(boundary_smoothing=True, normalize_coordinates=True)
        boundaries = smoothed_mesh.extract_feature_edges(non_manifold_edges=False, feature_edges=False,
                                                         manifold_edges=False, boundary_edges=True)
        boundaries = boundaries.split_bodies()
        caps = []
        for boundary in boundaries:
            cap = remesh_surface_2d(boundary, nosurf=True, hsiz=hsize)
            caps.append(cap)
        caps.insert(0, smoothed_mesh)
        model = pyvista.merge(caps)
        model.hsize = hsize
        return model, smoothed_mesh, caps
    else:
        return None, None, None
