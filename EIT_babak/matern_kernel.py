import numpy as np
import scipy.linalg as linalg

from dolfin import *
from mshr import *

from time import time

set_log_level(50)

class source(UserExpression):
    def eval(self,values,x):
        values[0] = 10*np.exp(-(np.power(x[0]-0.5, 2) + np.power(x[1], 2)) )

def boundary(x, on_boundary):
    return on_boundary

class matern_cov():
    def __init__(self):
        #domain = Circle(Point(0,0),1)
        #self.mesh = generate_mesh(domain, 20)
        #mesh_file = File("mesh.xml")
        #mesh_file << self.mesh
        self.mesh = Mesh('mesh.xml')
        self.V = FunctionSpace(self.mesh, "CG", 1)
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)

        u0 = Constant('0.0')
        self.bc = DirichletBC(self.V, u0, boundary)

        FEM_el = self.V.ufl_element()
        self.source = source(element=FEM_el)

    def compute_eigen_decomp(self, l=0.1, N_eig=32):
        tau2 = 1/l/l
        a = tau2*self.u*self.v*dx + inner( grad(self.u) , grad(self.v) )*dx

        u = Function(self.V)
        L = u*self.v*dx
        K = PETScMatrix()
        assemble_system(a, L,self.bc, A_tensor=K)
        eigen_solver = SLEPcEigenSolver(K)
        eigen_solver.parameters['spectrum'] = 'smallest magnitude'

        eigen_solver.solve(N_eig)
        eigvals = np.zeros(N_eig)
        eigvecs = np.zeros( [ u.vector().get_local().shape[0], N_eig ] )

        for i in range( N_eig ):
            val, c, vec, cx = eigen_solver.get_eigenpair(i)
            eigvals[i] = val
            eigvecs[:,i] = vec.get_local()

        eigvals = np.reciprocal( eigvals )
        eigvals /= np.linalg.norm( eigvals )

        path = 'matern_basis.npz'
        np.savez(path,tau2=tau2,l=eigvals,e=eigvecs)

    def sample(self):
        for i in range(1):
            u = np.random.standard_normal( self.l.shape[0] )
            vec = self.e@(self.l*u) 

            func = Function(self.V)
            func.vector().set_local( vec )
            file = File("sample{}.pvd".format(i))
            file << func

    def save_basis(self, path='./matern_basis.npz'):
        np.savez(path,tau2=self.tau2,nu=self.nu,l=self.l,e=self.e)

    def load_basis(self, path):
        basis_data = np.load(path)
        self.l = basis_data['l']
        self.e = basis_data['e']

if __name__ == '__main__':
    problem = matern_cov()
    problem.compute_eigen_decomp(l=0.2,N_eig=64)
    #problem.save_basis('basis.npz')
    problem.load_basis('matern_basis.npz')
    problem.sample()