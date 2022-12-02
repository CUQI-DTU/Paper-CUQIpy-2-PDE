
import dolfin as dl
import numpy as np
import cuqipy_fenics
import cuqi
import mshr
import matplotlib.pyplot as plt

exit()

#%% 1.1 Define domain and mesh
domain = mshr.Circle(dl.Point(0,0),1)
mesh = mshr.generate_mesh(domain, 20)

#%% 1.2 Define function spaces 
parameter_space = dl.FunctionSpace(mesh, "CG", 1)
solution_space = dl.FunctionSpace(mesh, "CG", 1)

#%% 1.3 Define boundary input as source term
class boundary_input(dl.UserExpression):
    def set_freq(self, freq=1.):
        self.freq = freq
    def eval(self, values, x, tag='sin'):
        theta = np.arctan2(x[1], x[0])
        values[0] = np.sin(self.freq*theta)

boundary = lambda x, on_boundary: on_boundary

FEM_el = solution_space.ufl_element()

bc_func = boundary_input(element=FEM_el)
bc_func.set_freq(freq=2.)

bc = dl.DirichletBC(solution_space, bc_func, boundary)

w = dl.Function(solution_space)
bc.apply(w.vector())

#%% 1.3.0 Defining custom conductivity with circular inclusions
class custom_field(dl.UserExpression):
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

FEM_el = parameter_space.ufl_element()
kappa_custom = custom_field(element=FEM_el)
kappa_custom.set_params()

#%% 1.3.1 Defining zero boundary for the extended problem
u0 = dl.Constant('0.0')
zero_bc = dl.DirichletBC(solution_space, u0, boundary)

#%% 1.4 Define Poisson problem form
def form(vec,u,v):
    kappa = dl.Function(parameter_space)
    kappa.vector().set_local( vec )
    return dl.inner( kappa*dl.grad(u), dl.grad(v) )*dl.dx + dl.inner( dl.grad(w), dl.grad(v) )*dl.dx

#%% 1.6 Define observation map (applied to the solution to generate the 
# observables)

#%% 1.6.1 extracting the index of boundary elements
dummy = dl.Function(solution_space)
dummy.vector().set_local( np.ones_like( dummy.vector().get_local() ) )
zero_bc.apply( dummy.vector() )
bnd_idx = np.argwhere( dummy.vector().get_local() == 0 ).flatten()

normal_vec = dl.FacetNormal( mesh )
tests = dl.TestFunction( solution_space )

#u = dl.TrialFunction(solution_space)
#v = dl.TestFunction(solution_space)

#A = dl.lhs(form(1,u,v))
#b = dl.rhs(form(1,u,v))

#solution = dl.Function(solution_space)
#dl.solve(A==b,solution,zero_bc)
#dl.plot(solution+ w)

def obs_func(kappa, u):
  obs_form = dl.inner( dl.grad(u + w), normal_vec )*tests*dl.ds

  obs = dl.assemble( obs_form )
  return obs.get_local()[bnd_idx]

#%% 2.1 Create the domain geometry
# 2.1.1 The space on which the Bayesian parameters are defined
domain_geometry = cuqipy_fenics.geometry.FEniCSContinuous(parameter_space)

#%% 2.2 Create the range geomtry 
range_geometry = cuqi.geometry.Continuous1D(94)

#%% 2.3 Create CUQI PDE (which encapsulates the FEniCS formulation
# of the PDE)
PDE = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE( form, mesh, solution_space, parameter_space,zero_bc, observation_operator=obs_func)

#%% 2.4 Create CUQI model
model = cuqi.model.PDEModel(PDE,range_geometry,domain_geometry)

#%% 2.6 Define the exact solution
func = dl.interpolate( kappa_custom, parameter_space )
exactSolution = cuqi.samples.CUQIarray(func.vector().get_local(), is_par=False, geometry= domain_geometry)

#%% 2.7 Generate exact data 
b_exact = model(exactSolution)

