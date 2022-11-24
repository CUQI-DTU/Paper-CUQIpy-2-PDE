import numpy as np
import dolfin as dl
import cuqipy_fenics
import cuqi
import matplotlib.pyplot as plt

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

#%% 2.1 Create the domain geometry
# 2.1.1 The space on which the Bayesian parameters are defined
fenics_continuous_geo = cuqipy_fenics.geometry.FEniCSContinuous(parameter_space)

# 2.1.2 The Matern fieled (maps i.i.d normal random vector of dimension `num_terms`
# to Matern field realization on `fenics_continuous_geo`)
matern_geo = cuqipy_fenics.geometry.MaternExpansion(fenics_continuous_geo, length_scale = .2, num_terms=64)

# 2.1.3 We create a map `heavy_map` to map the Matern field realization to two levels
# c_minus and c_plus 
c_minus = 1
c_plus = 10

ones_vec = np.ones(94)
bnd_idx = obs_funcs[0].bnd_idx
def heavy_map(func):
    dofs = func.vector().get_local()
    updated_dofs = c_minus*0.5*(1 + np.sign(dofs)) + c_plus*0.5*(1 - np.sign(dofs))

    updated_dofs[bnd_idx] = np.ones(94)
    func.vector().set_local(updated_dofs)
    return func

# 2.1.4 Finally, we create the domain geometry which applies the
# map `heavy_map` on Matern realizations.
domain_geometry = cuqipy_fenics.geometry.FEniCSMappedGeometry(matern_geo, map = heavy_map)

#%% 2.2 Create the range geomtry 
range_geometry = cuqi.geometry.Continuous1D(94) 


#%% 2.5 Create a prior
pr_mean = np.zeros(domain_geometry.par_dim)
prior = cuqi.distribution.Gaussian(pr_mean, cov=np.eye(domain_geometry.par_dim), geometry= domain_geometry, name='x')

#%% 2.6 Define the exact solution
exactSolution = prior.sample()

#%% 2.3 Create CUQI PDE (which encapsulates the FEniCS formulation
# of the PDE) and Create CUQI model

PDE_models = []
models = []
b_exact = []
sigma2 = []
data_dists = []
datas = []

#print(cuqipy_fenics.__version__)
#exit()

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
    datas.append( data_dists[i](x=exactSolution).sample() )

#%% 2.10 Create the joint data distribution and the joint likelihood
posterior = cuqi.distribution.JointDistribution(prior,data_dists[0],data_dists[1],data_dists[2],data_dists[3])(y1=datas[0], y2=datas[1], y3=datas[2], y4=datas[3])._as_stacked()


#%% 3 Third, we define a pCN sampler, sample, and inspect the prior and the posterior samples. 

#%% 3.1 Plot the exact solution
exactSolution.plot()

#%% 3.2 Plot prior samples
prior_samples = prior.sample(5)
ims = prior_samples.plot(title="prior")
plt.colorbar(ims[-1])


#%% 3.3 Create pCN Sampler 
Sampler = cuqi.sampler.MetropolisHastings(
    posterior,
    scale=None,
    x0=None,
)

#%% 3.4 Sample using the pCN sampler
samples = Sampler.sample_adapt(1000)

#%% 3.5 Plot posterior pCN samples 
#ims = samples.plot([0, 100, 300, 600, 800, 900],title="posterior")
#plt.colorbar(ims[-1])

# %% 3.6 Plot trace and autocorrelation (pCN)
#samples.plot_trace()
#samples.plot_autocorrelation(max_lag=300)

#%% 3.7 Plot credible interval (pCN)
#plt.figure()
#samples.plot_ci(plot_par = True, exact=exactSolution)
#plt.xticks(range(128)[::20], range(128)[::20])
#plt.title("Credible interval")
# %%