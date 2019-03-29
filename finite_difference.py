import numpy as np

class FiniteDifference():
    
    def __init__(
        self, 
        number_of_points=4, 
        left_boundary_condition=('Dirichlet', 100.),
        right_boundary_condition=('Dirichlet', 20.),
        source_term_function=lambda x : 0. * x
    ):
        self._length = 10. # m
        self._number_of_points = number_of_points
        self._delta_x = self._length / self._number_of_points
        self._x = np.linspace(0, self._number_of_points, self._number_of_points) * self._delta_x
        self._left_temperature = left_boundary_condition[1]
        self._right_temperature = right_boundary_condition[1]
        self._b = np.zeros(self._number_of_points)
        self._A = np.zeros((self._number_of_points, self._number_of_points))
        self._source_term_function = source_term_function

    def _SourceTerm(self, x):
        source_term = self._source_term_function(x)
        return -source_term * self._delta_x * self._delta_x
        
    def AssemblyLinearSystem(self):
        n = self._number_of_points

        # Central points
        self._A.flat[0:n*n:n+1] = -2.
        self._A.flat[n+2:n*n-n:n+1] = 1.
        self._A.flat[n+0:n*n-n:n+1] = 1.

        # Boundary Condition
        self._A[0, 0] = 1
        self._A[-1, -1] = 1
        
        # Independent Term - Boundary Condition
        self._b[0] = self._left_temperature
        self._b[-1] = self._right_temperature
        
        # Independent Term - Central points - Source
        self._b[1:-1] = self._SourceTerm(self._x[1:-1])

    def Solve(self):
        x = np.linalg.solve(self._A, self._b)
        return x

    def __str__(self):
        string = 'Number of points = ' + str(self._number_of_points) + '\n'
        string += 'dx = ' + str(self._delta_x) + '\n'
        string += 'Linear system:\n'
        string += 'A =\n' + str(self._A) + '\n'
        string += 'b =\n' + str(self._b) + '\n'
        return string
    