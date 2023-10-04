import numpy as np
import dolfin as dl
import cuqipy_fenics
import cuqi
import matplotlib.pyplot as plt

#%% 1 setting up FEniCS
# loading computational mesh
mesh = dl.Mesh("mesh.xml")

#%% 1.1 Define function spaces 
parameter_space = dl.FunctionSpace(mesh, "CG", 1)
solution_space = dl.FunctionSpace(mesh, "CG", 1)

#%% 1.2 defining the lifted weak formulation
# defining the ufl_element which is required 
# to construct FEniCS user defined functions
FEM_el = solution_space.ufl_element()

# unction that marks the boundary of the computational mesh
boundary = lambda x, on_boundary: on_boundary

# Creating lhs form
form_lhs = lambda sigma, v, t: dl.inner(sigma*dl.grad(v), dl.grad(t))*dl.dx

# Creating rhs for frequencies k=1,2,3,4
#bc_func = boundary_input(element=FEM_el) # creating a function for the boundary condition. This will create u_lift_1, u_lift_2, u_lift_3, u_lift_4
boundary_expression = dl.Expression("sin(k*atan2(x[1], x[0]))", k=1, degree=1)
bc = dl.DirichletBC(solution_space, boundary_expression, boundary) # applying this boundary condition on a function will create u_lift for the frequency in bc_func

# creating u_lift_1, u_lift_2, u_lift_3, u_lift_4
boundary_expression.k = 1 # setting the frequency
u_lift_1 = dl.Function(solution_space)
bc.apply(u_lift_1.vector())

boundary_expression.k = 2 # setting the frequency
u_lift_2 = dl.Function(solution_space)
bc.apply(u_lift_2.vector())

boundary_expression.k = 3 # setting the frequency
u_lift_3 = dl.Function(solution_space)
bc.apply(u_lift_3.vector())

boundary_expression.k = 4 # setting the frequency
u_lift_4 = dl.Function(solution_space)
bc.apply(u_lift_4.vector())

# creating rhs forms
form_rhs1 = lambda sigma, t: -dl.inner(dl.grad(u_lift_1), dl.grad(t))*dl.dx
form_rhs2 = lambda sigma, t: -dl.inner(dl.grad(u_lift_2), dl.grad(t))*dl.dx
form_rhs3 = lambda sigma, t: -dl.inner(dl.grad(u_lift_3), dl.grad(t))*dl.dx
form_rhs4 = lambda sigma, t: -dl.inner(dl.grad(u_lift_4), dl.grad(t))*dl.dx

#%% 1.3 defining the observation function
# extracting indecies for elements at the boundary of the computational mesh
#  Defining zero boundary for the lifted problem
v0 = dl.Constant('0.0')
zero_bc = dl.DirichletBC(solution_space, v0, boundary)
dummy = dl.Function(solution_space)
dummy.vector().set_local( np.ones_like( dummy.vector().get_local() ) )
zero_bc.apply( dummy.vector() )
bnd_idx = np.argwhere( dummy.vector().get_local() == 0 ).flatten() # this holds the indecies of the boundary elements

# defining normal vectors to the cell boundaries
n = dl.FacetNormal( mesh )
# defining FEniCS test functions
w = dl.TestFunction( solution_space )

# defining a function that returns the values at the boundaries
def give_bnd_vals(obs_form):
    return obs_form.get_local()[bnd_idx]

# defining the observation functions
def observation1(sigma, u):
    obs_form = dl.inner(dl.grad(u + u_lift_1), n)*w*dl.ds
    assembled_form = dl.assemble(obs_form)
    boundary_values = give_bnd_vals(assembled_form)
    return boundary_values

def observation2(sigma, u):
    obs_form = dl.inner(dl.grad(u + u_lift_2), n)*w*dl.ds
    assembled_form = dl.assemble(obs_form)
    boundary_values = give_bnd_vals(assembled_form)
    return boundary_values

def observation3(sigma, u):
    obs_form = dl.inner(dl.grad(u + u_lift_3), n)*w*dl.ds
    assembled_form = dl.assemble(obs_form)
    boundary_values = give_bnd_vals(assembled_form)
    return boundary_values

def observation4(sigma, u):
    obs_form = dl.inner(dl.grad(u + u_lift_4), n)*w*dl.ds
    assembled_form = dl.assemble(obs_form)
    boundary_values = give_bnd_vals(assembled_form)
    return boundary_values

#%% 2 paramterization and geometries
# The geometry on which the Bayesian parameters are defined correspods to the FEM paramterization
G_FEM = cuqipy_fenics.geometry.FEniCSContinuous(parameter_space)

# The KL para
G_KL = cuqipy_fenics.geometry.MaternExpansion(G_FEM, length_scale = .2, num_terms=64)

# Defining the Heaviside map
c_minus = 1
c_plus = 10
def Heaviside(func):
    dofs = func.vector().get_local() # extracting the function values at FEM nodes (this only works for linear element)
    updated_dofs = c_minus*0.5*(1 + np.sign(dofs)) + c_plus*0.5*(1 - np.sign(dofs))

    # Here we insure that the boundary values of the conductivity is always one
    updated_dofs[bnd_idx] = np.ones_like(bnd_idx)
    func.vector().set_local(updated_dofs)
    return func

# creating the domain geometry which applies the map Heaviside mapt to G_KL realizations.
G_Heavi = cuqipy_fenics.geometry.FEniCSMappedGeometry(G_KL, map=Heaviside)

#%% 3 Creating the prior distribution
# Create the range geomtry 
G_cont = cuqi.geometry.Continuous1D( len(bnd_idx) )

# Create a prior
x = cuqi.distribution.Gaussian(np.zeros(G_Heavi.par_dim), cov=np.eye(G_Heavi.par_dim), geometry=G_Heavi, name='x')

#%% 4 Creating the posterior distribution
# loading signal from file
obs_data = np.load('./obs/obs_circular_inclusion_2_10per_noise.npz')
b_exact = obs_data['b_exact']
s_noise2 = obs_data['sigma2']
data = obs_data['data']
y1_obs = data[0]
y2_obs = data[1]
y3_obs = data[2]
y4_obs = data[3]

# creating PDE forms
PDE_form1 = (form_lhs, form_rhs1)
PDE_form2 = (form_lhs, form_rhs2)
PDE_form3 = (form_lhs, form_rhs3)
PDE_form4 = (form_lhs, form_rhs4)


# creating PDE models
# for the first PDE problems we specify to reuse the factorization of the lhs for the rest of the PDE modesl
PDE1 = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE(PDE_form1, mesh, solution_space, parameter_space, zero_bc, observation_operator=observation1, reuse_assembled=True)
# We copy the PDE1 for the rest of the PDE problems 
PDE2 = PDE1.with_updated_rhs(form_rhs2)
PDE2.observation_operator = PDE2._create_observation_operator( observation2 )
PDE3 = PDE1.with_updated_rhs(form_rhs3)
PDE3.observation_operator = PDE3._create_observation_operator( observation3 )
PDE4 = PDE1.with_updated_rhs(form_rhs4)
PDE4.observation_operator = PDE4._create_observation_operator( observation4 )

# Creating the forward operators
A1 = cuqi.model.PDEModel(PDE1, range_geometry=G_cont, domain_geometry=G_Heavi)
A2 = cuqi.model.PDEModel(PDE2, range_geometry=G_cont, domain_geometry=G_Heavi)
A3 = cuqi.model.PDEModel(PDE3, range_geometry=G_cont, domain_geometry=G_Heavi)
A4 = cuqi.model.PDEModel(PDE4, range_geometry=G_cont, domain_geometry=G_Heavi)

# creating data distributions
y1 = cuqi.distribution.Gaussian(A1, s_noise2[0], geometry=G_cont)
y2 = cuqi.distribution.Gaussian(A2, s_noise2[1], geometry=G_cont)
y3 = cuqi.distribution.Gaussian(A3, s_noise2[2], geometry=G_cont)
y4 = cuqi.distribution.Gaussian(A4, s_noise2[3], geometry=G_cont)

# Creating the joint data distribution and the joint likelihood
joint = cuqi.distribution.JointDistribution(x,y1,y2,y3,y4)
posterior = joint(y1=y1_obs, y2=y2_obs, y3=y3_obs, y4=y4_obs)._as_stacked()

#%% 5 sampling
# Create Metropolis-Hastings Sampler 
Sampler = cuqi.sampler.MH(posterior)

# Sampling using the Metropolis-Hastings sampler
num_samples = 10000
posterior_samples = Sampler.sample_adapt(num_samples)

#%% 6 visualization
# plotting prior samples
f, axes = plt.subplots(1,3)
plt.sca(axes[0])
sample = x.sample()
sample.plot(subplots=False)
plt.sca(axes[1])
sample = x.sample()
sample.plot(subplots=False)
plt.sca(axes[2])
sample = x.sample()
sample.plot(subplots=False)
axes[1].set_title('prior samples')


# plotting posterior samples
posterior_samples.geometry = G_Heavi # setting the geometry to domain geometry.
idx = np.random.permutation(num_samples) # randomizing the posterior samples
f, axes = plt.subplots(1,3)
plt.sca(axes[0])
posterior_samples.plot(idx[0],subplots=False)
plt.sca(axes[1])
posterior_samples.plot(idx[1],subplots=False)
plt.sca(axes[2])
posterior_samples.plot(idx[2],subplots=False)
axes[1].set_title('posterior samples')

# plotting the mean
f, axes = plt.subplots(1,2)
plt.sca(axes[0])
im = posterior_samples.plot_mean(subplots=False)
axes[0].set_title('sample mean')

#plotting the variance
# Note: the following code for plotting the variance 
# will be replaced with one line of code 
# posterior_samples.funvals.plot_variance()
# in future release. 
plt.sca(axes[1])
func_vals_np = [] # a numpy list for holding fuction values
for s in posterior_samples:
    func_vals_np.append( G_Heavi.par2fun(s).vector().get_local() )
func_vals_np = np.array(func_vals_np)
var_np = np.var(func_vals_np,axis=0)
var_func = dl.Function(parameter_space)
var_func.vector().set_local( var_np )
dl.plot(var_func)
axes[1].set_title('variance')




