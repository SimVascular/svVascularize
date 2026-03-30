import numpy
import pyvista


def read(data, **kwargs):
    """
    Read and process mesh data. Supports PyVista objects and file paths to mesh files.

    Parameters:
        data: pyvista.PolyData, pyvista.UnstructuredGrid, or str
            Input mesh data as a PyVista object or a file path to a supported mesh file.
            For UnstructuredGrid objects (e.g., from .vtu files), the surface is extracted.
        **kwargs:
            feature_angle: float
                Angle used to determine sharp edges for normal computation.


    Returns:
        points: numpy.ndarray
            Array of point coordinates from the mesh.
        normals: numpy.ndarray
            Array of computed point normals from the mesh.
        n: int
            Number of points in the mesh.
        d: int
            Dimension of the points (typically 3).
    """
    feature_angle = kwargs.get("feature_angle", 30.0)

    if isinstance(data, str):
        data = pyvista.read(data)

    # Handle UnstructuredGrid by extracting surface (e.g., from .vtu files)
    if isinstance(data, pyvista.UnstructuredGrid):
        data = data.extract_surface()

    if not isinstance(data, pyvista.PolyData):
        raise TypeError("Input data must be a PyVista PolyData, UnstructuredGrid, or a valid file path.")

    data = data.compute_normals(split_vertices=True, feature_angle=feature_angle)

    # Materialize base NumPy arrays here so downstream allocation does not pay
    # pyvista_ndarray adapter overhead on every slice/index operation.
    points = numpy.array(data.points, dtype=numpy.float64, copy=True)
    normals = numpy.array(data.point_normals, dtype=numpy.float64, copy=True)
    n = points.shape[0]
    d = points.shape[1]
    return points, normals, n, d
