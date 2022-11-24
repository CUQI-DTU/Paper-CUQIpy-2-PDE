import numpy as np
import dolfin as dl
import cuqipy_fenics
import cuqi
import matplotlib.pyplot as plt
import pickle

#%% 1.1 loading mesh
mesh = dl.Mesh("mesh.xml")

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
class form():
    def set_w(self, w):
        self.w = w
    def lhs(self, kappa, u, v):
        return dl.inner( kappa*dl.grad(u), dl.grad(v) )*dl.dx

    def rhs(self, kappa, v):
        return - dl.inner( dl.grad(self.w), dl.grad(v) )*dl.dx

forms = []
residuals = []
for i in range(1,5):
    bc_func = boundary_input(element=FEM_el)
    bc_func.set_freq(freq=i)

    bc = dl.DirichletBC(solution_space, bc_func, boundary)

    w = dl.Function(solution_space)
    bc.apply(w.vector())

    temp_form = form()
    temp_form.set_w( w )
    forms.append( temp_form )


#%% 1.6 Define observation map (applied to the solution to generate the 
# observables)

#%% 1.6.1 extracting the index of boundary elements


class observation():
  def __init__(self):
    dummy = dl.Function(solution_space)
    dummy.vector().set_local( np.ones_like( dummy.vector().get_local() ) )
    zero_bc.apply( dummy.vector() )
    self.bnd_idx = np.argwhere( dummy.vector().get_local() == 0 ).flatten()

    self.normal_vec = dl.FacetNormal( mesh )
    self.tests = dl.TestFunction( solution_space )

  def set_w(self, w):
    self.w = w

  def obs_func(self, kappa, u):
    obs_form = dl.inner( dl.grad(u + self.w), self.normal_vec )*self.tests*dl.ds
    obs = dl.assemble( obs_form )
    return obs.get_local()[self.bnd_idx]

obs_funcs = []
for i in range(4):
  temp_obs_func = observation()
  temp_obs_func.set_w( forms[i].w )
  obs_funcs.append( temp_obs_func )

#u = dl.TrialFunction(solution_space)
#v = dl.TestFunction(solution_space)

#bc_func = boundary_input(element=FEM_el)
#bc_func.set_freq(freq=2.)
#bc = dl.DirichletBC(solution_space, bc_func, boundary)
#w = dl.Function(solution_space)

#bc.apply(w.vector())

#file = dl.File('input.pvd')
#file << w

#temp = form()
#temp.set_w( w )

#A = temp.lhs(kappa_custom ,u,v)
#b = temp.rhs(kappa_custom ,v)

#solution = dl.Function(solution_space)
#dl.solve(A==b,solution,zero_bc)
#solution.vector().set_local( solution.vector().get_local() + w.vector().get_local() )
#file = dl.File('sol.pvd')
#file << solution

#%% 2.1 Create the domain geometry
# 2.1.1 The space on which the Bayesian parameters are defined
domain_geometry = cuqipy_fenics.geometry.FEniCSContinuous(parameter_space)

#%% 2.2 Create the range geomtry 
range_geometry = cuqi.geometry.Continuous1D(94)

PDE_models = []
models = []
b_exact = []
sigma2 = []
data_dists = []
data = []

#print(cuqipy_fenics.__version__)
#exit()

#%% 2.6 Define the exact solution
func = dl.interpolate( kappa_custom, parameter_space )
exactSolution = cuqi.samples.CUQIarray(func, is_par=False, geometry= domain_geometry)

for i in range(4):
    if (i  == 0):
        PDE_model = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE( (forms[i].lhs, forms[i].rhs), mesh, solution_space, parameter_space,zero_bc, observation_operator=obs_funcs[i].obs_func, reuse_assembled=True)
    else:
        PDE_model = PDE_models[0].with_updated_rhs(forms[i].rhs)
        PDE_model.observation_operator = PDE_model._create_observation_operator( obs_funcs[i].obs_func )
    PDE_models.append( PDE_model )
    models.append( cuqi.model.PDEModel( PDE_model,range_geometry,domain_geometry)  )

    #%% 2.7 Generate exact data 
    b_exact.append( models[i](exactSolution) )

    #%% 2.8 Create the data distribution
    SNR = 100
    sigma = np.linalg.norm(b_exact[i])/SNR/np.sqrt(94)
    sigma2.append( sigma*sigma ) # variance of the observation Gaussian noise
    data_dists.append( cuqi.distribution.Gaussian(mean=models[i], cov=sigma2[i]*np.ones(range_geometry.par_dim), geometry=range_geometry, name='y{}'.format(i+1)) )

    #%% 2.9 Generate noisy data
    data.append( data_dists[i](x=exactSolution).sample() )

data = np.array(data)
print(data.shape)
b_exact = np.array(b_exact)
noise_vec = data - b_exact
np.savez( './obs/obs_circular_inclusion', data=data, b_exact=b_exact, noise_vec=noise_vec )

