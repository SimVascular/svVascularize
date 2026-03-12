import numpy as np
from .coordinate_system import angles_to_cartesian


def cost(x, h_, n, d):
    r"""
    Compute the cost function for energy minimization in implicit surface modeling.

    This function calculates the cost associated with a set of parameters `x` in the context
    of energy minimization for implicit surfaces. The parameters `x` represent the angular
    components of points on a unit hypersphere in :math:`d`-dimensional space. The radial
    component is assumed to be 1 for all points.

    The cost function is defined as:

    .. math::

        c = \mathbf{a}^\top \mathbf{H} \mathbf{a}

    where:

    - :math:`\mathbf{a}` is a column vector of size :math:`d \times n`, representing
      the Cartesian coordinates of the points.
    - :math:`\mathbf{H}` is a symmetric matrix of size (:math:`d \times n, d \times n`),
      representing the system derived from the bending energy minimization problem.

    Parameters
    ----------
    x : ndarray of shape (:math:`n \times (d - 1)`,)
        The vector of parameters to be optimized. These are the angular components of the
        spherical coordinates for each point (excluding the radial component).

    h_ : ndarray of shape (:math:`n \times d`, :math:`n \times d`)
        The :math:`\mathbf{H}` matrix representing the linear system in the energy minimization
        problem.

    n : int
        The number of points defining the :math:`(d - 1)`-dimensional manifold.

    d : int
        The dimension of the embedding space.

    Returns
    -------
    cost : float
        The scalar value of the cost function :math:`c`.

    Notes
    -----
    **Process Overview:**

    1. **Initialize Spherical Coordinates**:

       A matrix :math:`\mathbf{G}` of shape :math:`(n, d)` is initialized with ones.
       The first column corresponds to the radial component (assumed to be 1), and the
       remaining \( d - 1 \) columns are filled with the angular parameters for the
       coordinate dataset.

       .. math::

           \mathbf{G} = \begin{bmatrix}
                           1 & \phi_{1,1} & \phi_{1,2} & \dots & \phi_{1,d-1} \\
                           1 & \phi_{2,1} & \phi_{2,2} & \dots & \phi_{2,d-1} \\
                           \vdots & \vdots & \vdots & \ddots & \vdots \\
                           1 & \phi_{n,1} & \phi_{n,2} & \dots & \phi_{n,d-1}
                        \end{bmatrix}

    2. **Convert to Cartesian Coordinates**:

       The spherical coordinates :math:`\mathbf{G}` are converted to Cartesian coordinates
       using the `sph2cart` function, resulting in the matrix :math:`\mathbf{X}` of shape
       :math:`(n, d)`.

    3. **Reshape Coordinates into a Vector**:

       The Cartesian coordinates :math:`\mathbf{X}` are flattened into a column vector
       :math:`\mathbf{a}` of size :math:`d \times n`:

       .. math::

           \mathbf{a} = \text{vec}(\mathbf{X})

    4. **Compute the Cost Function**:

       The cost function is computed as the quadratic form:

       .. math::

           c = \mathbf{a}^\top \mathbf{H} \mathbf{a}

    **Understanding the Role of Each Parameter:**

    - **`x`**: Represents the angular components (:math:`\phi`) of the spherical coordinates
      for each point on the unit hypersphere. The radial component is not included in `x`
      since it is fixed at 1.

    - **`h_`**: The matrix derived from the energy minimization problem, often representing
      stiffness or bending energies in the context of surface interpolation.

    - **`n`** and **`d`**: Define the shape and dimensionality of the problem, determining
      how `x` is reshaped and how the coordinate transformations are performed.

    Examples
    --------
    **Example 1: Basic Usage**

    .. code-block:: python

        import numpy as np
        from svtoolkit.domain.kernel.cost import cost
        from svtoolkit.domain.kernel.coordinate_system import sph2cart
        from svtoolkit.domain.core.h_matrix import h_matrix

        # Number of points and dimensions
        n = 5
        d = 3

        # Generate random angular parameters
        x = np.random.rand(n * (d - 1))

        # Generate a random symmetric h_ matrix
        h_ = np.random.rand(n * d, n * d)
        h_ = (h_ + h_.T) / 2  # Ensure the matrix is symmetric

        # Compute the cost function
        c = cost(x, h_, n, d)
        print("Cost function value:", c)

    **Example 2: Using Specific Spherical Coordinates**

    .. code-block:: python

        import numpy as np
        from svtoolkit.domain.kernel.cost import cost
        from svtoolkit.domain.kernel.coordinate_system import sph2cart
        from svtoolkit.domain.core.h_matrix import h_matrix

        # Number of points and dimensions
        n = 3
        d = 2

        # Specific angular parameters (e.g., angles in radians)
        x = np.array([0.0, np.pi/4, np.pi/2])

        # Generate a dummy h_ matrix (identity for simplicity)
        h_ = np.eye(n * d)

        # Compute the cost function
        c = cost(x, h_, n, d)
        print("Cost function value:", c)

    See Also
    --------
    sph2cart : Convert spherical coordinates to Cartesian coordinates.

    """
    a = angles_to_cartesian(np.asarray(x, dtype=np.float64).reshape(n, d - 1))
    a = a.reshape(n * d)
    return float(a @ h_ @ a)
