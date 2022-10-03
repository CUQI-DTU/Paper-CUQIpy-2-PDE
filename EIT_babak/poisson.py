import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
#from matern import matern

from progressbar import progressbar

from dolfin import *
import mshr

from matern import matern

from time import time

set_log_level(50)

def boundary_diriichlet(x, on_boundary):
    return on_boundary

# Two circular inclusions
class custom_field(UserExpression):
    def set_params(self,cx=np.array([0.5,-0.5]),cy=np.array([0.5,0.6]), r = np.array([0.2,0.1]) ):
        self.cx = cx
        self.cy = cy
        self.r2 = r**2

    def eval(self,values,x):
        if( (x[0]-self.cx[0])**2 + (x[1]-self.cy[0])**2 < self.r2[0] ):
            values[0] = 10.
        elif( (x[0]-self.cx[1])**2 + (x[1]-self.cy[1])**2 < self.r2[1] ):
            values[0] = 10.
        else:
            values[0] = 1.

# Diriclet boundary conditions as input current
class boundary_input(UserExpression):
    def set_freq(self, freq=1.):
        self.freq = freq
    def eval(self, values, x, tag='sin'):
        theta = np.arctan2(x[1], x[0])
        values[0] = np.sin(self.freq*theta)

# This class solves the Laplace equation on the 
# unit circle:
# 
# div( kappa(x,y) grad u(x,y) ) = 0, in the domain
# u(x,y) = sin(k arctan(y/x) ), on the boundary
#
# to solve this we introduce a new residual function R
# which satisfies the boundary condition and is zero 
# inside the boundary. We then solve
# 
# div( kappa(x,y) grad( u0(x,y) + R(x,y)) ) = 0, in the domain
# u0 = 0, on the boundary
#
# once we find u0, we can then find u = u0 + R
# 
# This class also considers the neuman condition of u
# as an observation function, i.e.,
#
# y_obs = kappa(x,y) grad u(x,y), on the boundary

class poisson():
    def __init__(self):
        # defining the domain and mesh
        #domain = mshr.Circle(Point(0,0),1)
        #self.mesh = mshr.generate_mesh(domain, 20)
        self.mesh = Mesh('mesh.xml')
        self.V = FunctionSpace(self.mesh,'CG', 1)

        # defining test and trial spaces
        u0 = Constant('0.0')
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)

        # ufl
        FEM_el = self.V.ufl_element()

        # diffusivity field
        #self.kappa = custom_field(element=FEM_el)
        #self.kappa.set_params()

        self.kappa = Function(self.V)
        self.matern_field = matern('matern_basis.npz')
        self.matern_field.set_levels(c_minus=1.,c_plus=10.)
        #self.kappa.vector().set_local( self.matern_field.assemble(p) )
        #self.kappa.vector().set_local( np.ones(833) )


        #file = File('rand_field.pvd')
        #file << self.kappa
        #exit()

        # defining the bilinear form
        self.a = self.kappa*self.u.dx(0)*self.v.dx(0)*dx + self.kappa*self.u.dx(1)*self.v.dx(1)*dx

        # defining the boundary condition with particular frequency
        self.bc_func = boundary_input(element=FEM_el)
        self.bc_func.set_freq(freq=10.)

        # defining the physical boundary condition and the zero boundary condition
        self.bc = DirichletBC(self.V, self.bc_func, boundary_diriichlet)
        self.zero_bc = DirichletBC(self.V, u0, boundary_diriichlet)

        # defining the residual function
        self.R = Function(self.V)
        self.bc.apply(self.R.vector())
        self.L = self.R.dx(0)*self.v.dx(0)*dx + self.R.dx(1)*self.v.dx(1)*dx

        # extracting the indecies of the boundary elements
        f = Function(self.V)
        ones_vec = np.ones_like( f.vector().get_local() )
        f.vector().set_local( ones_vec )
        self.zero_bc.apply(f.vector())
        self.bnd_idx = np.argwhere( f.vector().get_local() == 0 ).flatten()

        p = np.random.standard_normal(self.matern_field.dim)
        vec = self.matern_field.assemble(p)
        vec[self.bnd_idx] = np.ones(94)
        self.kappa.vector().set_local( vec )
        file = File('rand_field.pvd')
        file << self.kappa

        # defining the normal vector to the boundary
        self.n = FacetNormal(self.mesh)

    # precomputing the right hand side and the 
    # residual function for all frequencies
    def precomute_rhs(self):
        self.rhs = []
        self.residuals = []
        for i in progressbar( range(5) ):
            self.bc_func.set_freq( freq=float(i) )
            self.bc.apply(self.R.vector())
            self.residuals.append( self.R.vector().get_local() )
            b = assemble(self.L)
            self.zero_bc.apply(b)

            self.rhs.append( b )

    # Solving for multiple frequencies in the 
    # dirichlet boundary
    def custom_solve(self):
        # computing the mass matrix
        A = assemble(self.a)
        # applying the zero boundary condition
        self.zero_bc.apply(A)

        # computing the LU decomposition of the mass matrix
        #t1 = time()
        solver = LUSolver(A)
        #print(time() - t1)

        # defining the functions u0 and u
        u0 = Function(self.V)
        vec = u0.vector()
        u = Function(self.V)

        path = 'sol{}.pvd'

        # defining the observation form 
        #obs_form = self.kappa*u.dx(0)*self.n[0]*self.v*ds + self.kappa*u.dx(1)*self.n[1]*self.v*ds
        obs_form = inner( grad(u),self.n )*self.v*ds

        obs_vec = []
        for i in range(1,5):
            # solving the system for u0
            #t1 = time()
            solver.solve(vec, self.rhs[i])
            #print(time() - t1)
            u.vector().set_local( u0.vector().get_local() - self.residuals[i] )


            # computing the observation function
            obs = assemble( obs_form )
            #file = File('bnd{}.pvd'.format(i))
            #temp = Function(self.V)
            #temp.vector().set_local( obs.get_local() )
            #file << temp
            #print( obs.get_local()[self.bnd_idx] )
            obs_vec.append( obs.get_local()[self.bnd_idx] )
            file = File(path.format(i))
            file << u
        return np.array(obs_vec)

    def forward(self,p):
        vec = self.matern_field.assemble(p)
        vec[self.bnd_idx] = np.ones(94)
        self.kappa.vector().set_local( vec )
        return self.custom_solve()

    def save_obs(self):
        FEM_el = self.V.ufl_element()
        kappa = custom_field(element=FEM_el)
        kappa.set_params()

        a = kappa*self.u.dx(0)*self.v.dx(0)*dx + kappa*self.u.dx(1)*self.v.dx(1)*dx
        A = assemble(a)
        self.zero_bc.apply(A)
        solver = LUSolver(A)

        u0 = Function(self.V)
        vec = u0.vector()
        u = Function(self.V)

        obs_form = inner( grad(u),self.n )*self.v*ds
        obs_vec = []

        for i in range(1,5):
            solver.solve(vec, self.rhs[i])
            u.vector().set_local( u0.vector().get_local() - self.residuals[i] )
            obs = assemble( obs_form )
            obs_vec.append( obs.get_local()[self.bnd_idx] )

        obs_vec = np.array(obs_vec)

        noise_vec = np.random.standard_normal( obs_vec.shape )
        noise_vec = noise_vec/np.linalg.norm(noise_vec)

        np.savez('./obs/obs1.npz',obs_vec=obs_vec,noise_vec=noise_vec)

    def post_process(self):
        stat_data = np.load('./stat/stat1.npz')
        samples = stat_data['samples']

        samples_thin = samples[60000:,:]
        mean_params = np.mean(samples_thin,axis=0)

        FEM_el = self.V.ufl_element()
        true_field = custom_field(element=FEM_el)
        true_field.set_params()
        file = File('true_field.pvd')
        tf = Function(self.V)
        tf = interpolate(true_field, self.V)
        file << tf

        
        vec = self.matern_field.assemble(mean_params)
        vec[self.bnd_idx] = np.ones(94)
        self.kappa.vector().set_local( vec )
        file = File('posterior_mean.pvd')
        file << self.kappa

        fields = []
        for i, s in progressbar( enumerate( samples_thin )):
            vec = self.matern_field.assemble(s)
            vec[self.bnd_idx] = np.ones(94)

            fields.append(vec)

        fields = np.array(fields)
        var = np.var( fields, axis = 0 )

        var_func = Function(self.V)
        var_func.vector().set_local( var )

        file = File('variace.pvd')
        file << var_func    


        #self.custom_solve()


if __name__ == '__main__':
    problem = poisson()
    problem.precomute_rhs()

    #problem.custom_solve()

    #problem.save_obs()
    #problem.post_process()

    p = np.random.standard_normal(64)
    res = problem.forward(p)


