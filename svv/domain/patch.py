import numpy as np
from .kernel.kernel import Kernel
from .solver.solver import Solver, constants_from_cartesian


class Patch:
    """
    Patch class decomposes a domain into a set of interpolation sub-problems which
    can be solved independently and then blended together to form the final
    domain interpolation function.
    """
    def __init__(self, lam=0):
        self.points = None
        self.normals = None
        self.kernel = None
        self.solver = None
        self.constants = None
        self.rbf_degree = 3
        self.lam = lam

    def set_data(self, *args, **kwargs):
        """
        Set the data for the domain.
        """
        create_kernel = kwargs.get('create_kernel', True)
        if len(args) > 1:
            self.points = args[0]
            self.normals = args[1]
            if create_kernel:
                self.kernel = Kernel(self.points, lam=self.lam)
                self.kernel.set_initial_values(self.normals)
            else:
                self.kernel = None
        elif len(args) > 0:
            self.points = args[0]
            if create_kernel:
                self.kernel = Kernel(self.points, lam=self.lam)
                self.kernel.set_initial_values()
            else:
                self.kernel = None
        else:
            print("Error: No data provided.")
        return None

    def solve(self, method="L-BFGS-B", precision=9):
        """
        Solve the interpolation problem for the patch object

        Parameters:
        -----------
        method : str
            The method to use for the solver. Default is L-BFGS-B.
            (Limited-memory Broyden-Fletcher-Goldfarb-Shanno Bounded)

        precision : int
            The number of decimal places to round the constants to.
            Default is 9.

        Returns:
        --------
        None
        """
        if self.normals is not None:
            normals = np.asarray(self.normals, dtype=np.float64)
            normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
            self.solver = None
            self.constants = np.round(constants_from_cartesian(self.kernel, normals), decimals=precision)
            return None
        if self.solver is None or self.solver.kernel is not self.kernel:
            self.solver = Solver(self.kernel)
        self.solver.set_solver(method=method)
        self.solver.solve()
        self.constants = np.round(self.solver.get_constants(), decimals=precision)
        return None

    def build(self):
        """
        Build the interpolation function for the patch object
        :return:
        """
        # Avoid heavy kernel builds if constants are already present by
        # deriving sizes from points directly.
        n = self.points.shape[0]
        d_dim = self.points.shape[1]
        a = np.array(self.constants[:n])
        b = np.array(self.constants[n:n * (d_dim + 1)].reshape(n, d_dim))
        c = np.array(self.constants[n * (d_dim + 1):n * (d_dim + 1) + d_dim])
        d = np.array(self.constants[-1])

        # If kernel doesn't exist (e.g., loaded from .dmn), create a minimal one for dimension info
        if self.kernel is None:
            if self.normals is not None:
                self.kernel = Kernel(self.points, lam=self.lam)
            else:
                self.kernel = Kernel(self.points, lam=self.lam)

        a = a.reshape(a.shape + (1,) * self.kernel.d)
        b = b.reshape(tuple([b.shape[0]]) + (1,) * self.kernel.d + tuple([b.shape[1]]))
        c = c.reshape((1,) * self.kernel.d + c.shape)
        pts = np.array(self.points.reshape((n,) + (1,) * d_dim + (d_dim,)))
        def f(x, a_=a, b_=b, c_=c, d_=d, points=pts, show=False):
            """
            Interpolation function for a patch of the domain.

            Parameters:
            -----------
            x : array_like
                The point(s) at which to evaluate the interpolation function. Given as an
                ndarray of shape (..., d), where d is the dimension of the domain.

            a_ : array_like
                The coefficients for the Duchon interpolation function.

            b_ : array_like
                The coefficients for the gradient of the Duchon interpolation function.

            c_ : array_like
                The coefficients for the Hessian of the Duchon interpolation function.

            d_ : float
                The constant term for the Duchon interpolation function.

            Returns:
            --------
            value : array_like
                The value of the interpolation function at the given point(s). Given as an
                ndarray of shape (..., 1).
            """
            diff = x - points
            diff_2 = np.sum(np.square(diff), axis=-1)
            if show:
                print("Diff: ", diff)
                print("Diff^2: ", diff_2)
            #a_value = np.sum(a_ * np.sum(np.square(diff), axis=-1) ** (self.rbf_degree / 2), axis=0)
            # A Value
            value = np.sum(a_ * np.power(diff_2, (self.rbf_degree / 2)), axis=0)
            if show:
                print("A Value: ", value)
            # B Value
            value += np.sum(self.rbf_degree * np.sum((-b_) * diff, axis=-1) * np.power(diff_2, (
                        self.rbf_degree / 2 - 1)), axis=0)
            if show:
                print("A + B Value: ", value)
            # C Value
            value += np.sum(c_ * x, axis=-1)
            if show:
                print("A + B + C Value: ", value)
            # D Value
            value += d_
            if show:
                print("A + B + C + D Value: ", value)
            return value
        f.points = self.points
        f.pts = pts
        if self.normals is not None:
            f.normals = self.normals
        else:
            f.normals = self.solver.get_normals()
        f.dimensions = d_dim
        f.min = np.min(self.points, axis=0)
        f.max = np.max(self.points, axis=0)
        f.centroid = np.mean(self.points, axis=0)
        f.first = self.points[0, :]
        f.a = a
        f.b = b
        f.c = c
        f.d = d
        # attach normals when available to support serialization
        if getattr(self, 'normals', None) is not None:
            f.normals = self.normals
        return f
