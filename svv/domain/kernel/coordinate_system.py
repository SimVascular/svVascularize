import numpy as np


def angles_to_cartesian(angles):
    """Convert angular spherical coordinates with unit radius to Cartesian coordinates."""
    angles = np.asarray(angles, dtype=np.float64)
    n = angles.shape[0]
    angle_count = angles.shape[1]
    cart = np.empty((n, angle_count + 1), dtype=np.float64)
    if angle_count == 0:
        cart[:, 0] = 1.0
        return cart
    sin_angles = np.sin(angles)
    cos_angles = np.cos(angles)
    prefix_sines = np.ones((n, angle_count), dtype=np.float64)
    if angle_count > 1:
        prefix_sines[:, 1:] = np.cumprod(sin_angles[:, :-1], axis=1)
    cart[:, :-1] = prefix_sines * cos_angles
    cart[:, -1] = np.prod(sin_angles, axis=1)
    return cart


def cart2sph(cart):
    r"""
    Convert Cartesian coordinates to spherical coordinates.

    This function transforms a set of points in d-dimensional Cartesian coordinates to d-dimensional
    spherical coordinates. In spherical coordinates, the first element is the radial distance (norm), and
    the subsequent elements represent angular coordinates. More information on d-dimensional spherical
    coordinates can be found in [1]_.

    - **Coordinate Conversion**:


    Parameters
    ----------
    cart : ndarray of shape (N, d)
        Cartesian coordinates for N points in d-dimensional space. Each row represents the
        Cartesian coordinates of a point.

    Returns
    -------
    sph : ndarray of shape (N, d)
        Spherical coordinates for the N points. Each row represents the spherical coordinates of a
        point, where the first coordinate is the radial distance (norm), and the remaining coordinates
        are angles in radians.


    Notes
    -----
    The conversion from Cartesian to spherical coordinates in n-dimensional space is performed using
    the following formulas [2]_:

    For a point :math:`\mathbf{x} = (x_1, x_2, x_3, \dots, x_{d-1}, x_d)`:

    - **Radial distance** \( :math:`\mathbf{r}` \):

      .. math::

          r = \sqrt{\sum_{i=1}^{d} x_i^2}

    - **Angles** (:math:`\phi_1, \phi_2, \dots, \phi_{d-2}, \phi_{d-1}`):

      .. math::

          \begin{align}
            \phi_1 =& \text{atan2}(\sqrt{x_d^2 + x_{d-1}^2 + \dots + x_2^2, x_1}) \\
            \phi_2 =& \text{atan2}(\sqrt{x_d^2 + x_{d-1}^2 + \dots + x_3^2, x_2}) \\
            &\vdots \\
            \phi_{d-2} =& \text{atan2}(\sqrt{x_d^2 + x_{d-1}^2, x_{d-2}}) \\
            \phi_{d-1} =& \text{atan2}(x_d, x_{d-1})
          \end{align}

    Special handling is provided for angles to ensure they lie within the interval
    :math:`[0, 2\pi]`.

    References
    ----------

    .. [1] Wikipedia. "n-sphere" `Wiki Link <https://en.wikipedia.org/wiki/N-sphere>`_
    .. [2] Blumenson, L. E. (1960). "A Derivation of n-Dimensional Spherical Coordinates".
           The American Mathematical Monthly. 67 (1): 63–66.
           `JSTOR Link <https://doi.org/10.2307%2F2308932>`_


    Examples
    --------
    Convert a set of 3D Cartesian coordinates to spherical coordinates:

    .. code-block:: python

        import numpy as np
        from svtoolkit.domain.kernel.coordinate_system import cart2sph

        cart = np.array([[1, 1, 1], [0, 0, 1]])
        sph = cart2sph(cart)
        print(sph)
        # Output:
        # array([[1.73205081, 0.95531662, 0.78539816],
        #        [1.        , 0.        , 0.        ]])

    Convert a set of 2D Cartesian coordinates:

    .. code-block:: python

        import numpy as np
        from svtoolkit.domain.kernel.coordinate_system import cart2sph

        cart = np.array([[1, 1], [0, 1]])
        sph = cart2sph(cart)
        print(sph)
        # Output:
        # array([[1.41421356, 0.78539816],
        #        [1.        , 0.        ]])

    """
    n = cart.shape[0]
    d = cart.shape[1]
    sph = np.zeros((n, d))
    for i in range(n):
        sph[i, 0] = np.linalg.norm(cart[i, :])
        for j in range(d - 2):
            sph[i, j + 1] = np.arccos(cart[i, j] / np.linalg.norm(cart[i, j:]))
        if np.linalg.norm(cart[i, -2:]) == 0:
            sph[i, -1] = 0
        elif cart[i, -1] >= 0:
            sph[i, -1] = np.arccos(cart[i, -2] / np.linalg.norm(cart[i, -2:]))
        else:
            sph[i, -1] = 2 * np.pi - np.arccos(cart[i, -2] / np.linalg.norm(cart[i, -2:]))
    return sph


def sph2cart(sph):
    r"""
    Convert spherical coordinates to Cartesian coordinates.

    This function transforms a set of points in d-dimensional spherical coordinates to d-dimensional
    Cartesian coordinates. In Cartesian coordinates, each point is represented the d-dimensional
    signed-distance from given coordinate points to d mutually perpendicular, fixed hyperplanes [1]_.

    Parameters
    ----------
    sph : ndarray of shape (N, d)
        Spherical coordinates for N points in d-dimensional space. Each row represents the
        spherical coordinates of a point, where the first element is the radial distance, and the
        remaining elements are angles in radians.

    Returns
    -------
    cart : ndarray of shape (N, d)
        Cartesian coordinates for the N points. Each row represents the Cartesian coordinates of a
        point in d-dimensional space.


    Notes
    -----
    The conversion from spherical to Cartesian coordinates in n-dimensional space is performed using
    the following formulas [2]_:

    For a point with radial distance ( :math:`r` \ and angles ( :math:`\phi_1, \phi_2, \phi_3, \dots, \phi_{d-1}` ):

    - **Cartesian coordinates** :math:`\mathbf{x}_i \forall i = 1, \dots, d`:

      .. math::

          x_1 &= r \cos(\phi_1) \\
          x_2 &= r \sin(\phi_1) \cos(\phi_2) \\
          x_3 &= r \sin(\phi_1) \sin(\phi_2) \cos(\phi_3) \\
          &\vdots \\
          x_{d-1} &= r \left( \prod_{k=1}^{d-2} \sin(\phi_k) \right) \cos(\phi_{d-1}) \\
          x_d &= r \left( \prod_{k=1}^{d-1} \sin(\phi_k) \right)

    Special handling is provided to prevent numerical errors when angles are at critical values such as
    \( :math:`0, \pm\frac{\pi}{2}, \pi` \).

    References
    ----------

    .. [1] Wikipedia. "Hyperplane" `Wiki Link <https://en.wikipedia.org/wiki/Hyperplane>`_
    .. [2] Blumenson, L. E. (1960). "A Derivation of n-Dimensional Spherical Coordinates".
           The American Mathematical Monthly. 67 (1): 63–66.
           `JSTOR Link <https://doi.org/10.2307%2F2308932>`_

    Examples
    --------

    Convert a set of 2D spherical coordinates:

    .. code-block:: python

        import numpy as np
        from svtoolkit.domain.kernel.coordinate_system import sph2cart

        sph = np.array([[1.41421356, 0.78539816], [1.0, 0.0]])
        cart = sph2cart(sph)
        print(cart)
        # Output:
        # array([[1., 1.],
        #        [1., 0.]])

    Convert a set of 3D spherical coordinates to Cartesian coordinates:

    .. code-block:: python

        import numpy as np
        from svtoolkit.domain.kernel.coordinate_system import sph2cart

        sph = np.array([[1.73205081, 0.95531662, 0.78539816], [1.0, 0.0, 0.0]])
        cart = sph2cart(sph)
        print(cart)
        # Output:
        # array([[1., 1., 1.],
        #        [1., 0., 0.]])
    """
    sph = np.asarray(sph, dtype=np.float64)
    if sph.shape[1] == 1:
        return sph.copy()
    cart = angles_to_cartesian(sph[:, 1:])
    cart *= sph[:, :1]
    return cart
