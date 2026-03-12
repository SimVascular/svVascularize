import numpy as np
from scipy import optimize
from functools import partial
from ..kernel.coordinate_system import angles_to_cartesian


class DomainSolution:
    def __init__(self):
        self.x = None
        self.fun = None


def constants_from_cartesian(kernel, cartesian_solution):
    g = np.asarray(cartesian_solution, dtype=np.float64).reshape(-1)
    if kernel.lam == 0:
        s = np.zeros(kernel.n, dtype=np.float64)
    else:
        rhs = kernel.j01_ @ g
        system = np.eye(kernel.j00_.shape[0], dtype=np.float64) + kernel.lam * kernel.j00_
        s = -kernel.lam * np.linalg.solve(system, rhs)
    left_side = np.zeros(kernel.a_inv.shape[0], dtype=np.float64)
    left_side[:kernel.n] = s
    left_side[kernel.n:kernel.n + g.shape[0]] = g
    return kernel.a_inv @ left_side


class Solver:
    """
    This class defines the solver for the implicit domain object.
    """

    def __init__(self, kernel):
        self.kernel = kernel
        self.algorithm = None
        self.solution = None
        self.solutions = None

    def set_solver(self, method='L-BFGS-B', verbose=False):
        """
        This function sets the solver method to be used for the optimization problem.
        """
        scipy_solvers = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG',
                         'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr',
                         'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
        if method in scipy_solvers:
            if verbose:
                if method == 'trust-constr':
                    options = {'disp': True}
                elif method == 'L-BFGS-B':
                    options = {'disp': 99, 'maxls': 40, 'ftol': 1e-9}
                elif method in ['Newton-CG', 'CG', 'BFGS', 'Nelder-Mead',
                                'Powell', 'TNC', 'COBYLA', 'SLSQP']:
                    options = {'disp': True}
                else:
                    print('No verbosity allowed for this method.')
                    options = {}
            else:
                options = {}
            algorithm = partial(optimize.minimize, method=method, tol=1e-09, options=options)
        else:
            print('Not an available solver method.')
            print('See scipy methods: {}'.format(scipy_solvers))
            algorithm = None
        self.algorithm = algorithm

    def solve(self, skip=False):
        if self.algorithm is None:
            print('Solver not set.')
            return None
        if skip:
            solution = DomainSolution()
            solution.fun = self.kernel.__cost__(self.kernel.x0[0])
            solution.x = self.kernel.x0[0]
            self.solution = solution
            self.solutions = [solution]
            return solution
        kernel_bounds = self.kernel.get_bounds()
        bounds = []
        for i in range(len(kernel_bounds[0])):
            bounds.append(tuple([kernel_bounds[0][i], kernel_bounds[1][i]]))
        solutions = []
        solution_values = []
        for i in range(len(self.kernel.x0)):
            solution = self.algorithm(self.kernel.__costs__[i], self.kernel.x0[i], bounds=bounds)
            solution_values.append(solution.fun)
            solutions.append(solution)
        best_solution = solutions[np.argmin(solution_values)]
        x0 = best_solution.x
        solution = self.algorithm(self.kernel.__cost__, x0, bounds=bounds)
        self.solution = solution
        self.solutions = solutions
        return solution

    def get_constants(self):
        if self.solution is None:
            print('Solver not run.')
            return None
        cartesian_solution = angles_to_cartesian(self.solution.x.reshape(self.kernel.n, self.kernel.d - 1))
        return constants_from_cartesian(self.kernel, cartesian_solution)

    def get_normals(self):
        if self.solution is None:
            print('Solver not run.')
            return None
        return angles_to_cartesian(self.solution.x.reshape(self.kernel.n, self.kernel.d - 1))
