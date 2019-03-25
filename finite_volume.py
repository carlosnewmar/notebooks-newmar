import numpy as np

class FiniteVolume():
    
    def __init__(
        self,
        number_of_volumes=4,
        left_boundary_condition=('Dirichlet', 100.),
        right_boundary_condition=('Dirichlet', 20.),
        source_term_function=lambda x : 0. * x
    ):
        self._length = 10.
        self._number_of_volumes = number_of_volumes
        self._delta_x = self._length / self._number_of_volumes
        self._x = np.arange(0.5 * self._delta_x, self._length, self._delta_x)
        self._k = 1.
        
        assert type(left_boundary_condition) == tuple
        assert type(left_boundary_condition[0]) == str
        assert type(left_boundary_condition[1]) == float
        assert left_boundary_condition[0] in ['Dirichlet', 'Neumann', 'Robin']
        if left_boundary_condition[0] == 'Robin':
            assert type(left_boundary_condition[2]) == float
        self._left_boundary_condition = left_boundary_condition
        
        assert type(right_boundary_condition) == tuple
        assert type(right_boundary_condition[0]) == str
        assert type(right_boundary_condition[1]) == float
        assert right_boundary_condition[0] in ['Dirichlet', 'Neumann', 'Robin']
        if right_boundary_condition[0] == 'Robin':
            assert type(right_boundary_condition[2]) == float
        self._right_boundary_condition = right_boundary_condition
        
        self._b = np.zeros(self._number_of_volumes)
        self._A = np.zeros((self._number_of_volumes, self._number_of_volumes))
        self._source_term_function = source_term_function

    def _SourceTerm(self, x):
        source_term = self._source_term_function(x)
        return source_term * self._delta_x
    
    def _ApplyLeftBoundaryCondition(self):
        if self._left_boundary_condition[0] == 'Dirichlet':
            a_w = 0.
            a_p = 3. * self._k / self._delta_x
            a_e = self._k / self._delta_x
            b = 2. * self._k / self._delta_x * self._left_boundary_condition[1]
        elif self._left_boundary_condition[0] == 'Neumann':
            a_w = 0.
            a_p = self._k / self._delta_x
            a_e = self._k / self._delta_x
            b = self._left_boundary_condition[1]
        elif self._left_boundary_condition[0] == 'Robin':
            a_w = 0.
            a_p = self._k / self._delta_x - self._left_boundary_condition[2]
            a_e = self._k / self._delta_x
            b = - self._left_boundary_condition[1] * self._left_boundary_condition[2]            
        else:
             print('Invalid boundary condition')
        
        return a_w, a_p, a_e, b
    
    def _ApplyRightBoundaryCondition(self):
        if self._right_boundary_condition[0] == 'Dirichlet':
            a_w = self._k / self._delta_x
            a_p = 3. * self._k / self._delta_x
            a_e = 0.
            b = 2. * self._k / self._delta_x * self._right_boundary_condition[1]
        elif self._right_boundary_condition[0] == 'Neumann':
            a_w = self._k / self._delta_x
            a_p = self._k / self._delta_x
            a_e = 0.
            b = self._right_boundary_condition[1]
        elif self._right_boundary_condition[0] == 'Robin':
            a_w = self._k / self._delta_x
            a_p = self._k / self._delta_x + self._right_boundary_condition[2]
            a_e = 0.
            b = self._right_boundary_condition[1] * self._right_boundary_condition[2]
        else:
             print('Invalid boundary condition')
        
        return a_w, a_p, a_e, b
    
    def AssemblyLinearSystem(self):
        n = self._number_of_volumes
        
        a_p = np.ones(n)
        a_w = -np.ones(n)
        a_e = -np.ones(n)
        
        a_p[1:-1] = 2. * self._k / self._delta_x
        a_e[1:-1] = self._k / self._delta_x
        a_w[1:-1] = self._k / self._delta_x
        
        self._b = self._SourceTerm(self._x)

        a_w[0], a_p[0], a_e[0], b = self._ApplyLeftBoundaryCondition()
        self._b[0] += b
        a_w[-1], a_p[-1], a_e[-1], b = self._ApplyRightBoundaryCondition()
        self._b[-1] += b
        
        self._A.flat[0:n*n:n+1] = a_p
        self._A.flat[1:n*n:n+1] = -a_e[0:n]
        self._A.flat[n:n*n:n+1] = -a_w[1:n-1]

    def Solve(self):
        x = np.linalg.solve(self._A, self._b)
        return x

    def __str__(self):
        string = 'Number of volumes = ' + str(self._number_of_volumes) + '\n'
        string += 'dx = ' + str(self._delta_x) + '\n'
        string += 'Linear system:\n'
        string += 'A =\n' + str(self._A) + '\n'
        string += 'b =\n' + str(self._b) + '\n'
        return string
    