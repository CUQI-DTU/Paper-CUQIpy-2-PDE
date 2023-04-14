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

#%% 1.2 Define boundary condition or source term in the lifted formulation
class boundary_input(dl.UserExpression):
    def set_freq(self, freq=1.):
        self.freq = freq
    def eval(self, values, x, tag='sin'):
        theta = np.arctan2(x[1], x[0])
        values[0] = np.sin(self.freq*theta)

#%% 1.2 defining the lifted weak formulation
# defining the ufl_element which is required 
# to construct FEniCS user defined functions
FEM_el = solution_space.ufl_element()

# unction that marks the boundary of the computational mesh
boundary = lambda x, on_boundary: on_boundary

#  Defining zero boundary for the lifted problem
v0 = dl.Constant('0.0')
zero_bc = dl.DirichletBC(solution_space, v0, boundary)

# the class that creates the lhs and rhs of the lifted problem
class form():
    def set_u_lift(self, u_lift):
        self.u_lift = u_lift
    def lhs(self, kappa, u, v):
        return dl.inner( kappa*dl.grad(u), dl.grad(v) )*dl.dx

    def rhs(self, kappa, v):
        return - dl.inner( dl.grad(self.u_lift), dl.grad(v) )*dl.dx

# Creating a list of lhs and rhs for frequencies k=1,2,3,4
forms = [] # this list holds all lhs and rhs forms
for i in range(1,5):
    bc_func = boundary_input(element=FEM_el)
    bc_func.set_freq(freq=i)

    # the boundary condition which applies the boundary_input to a lifting function
    bc = dl.DirichletBC(solution_space, bc_func, boundary)

    u_lift = dl.Function(solution_space)
    bc.apply(u_lift.vector())

    temp_form = form()
    temp_form.set_u_lift( u_lift ) # setting u_lift for the i-th frequency
    forms.append( temp_form )

#%% 1.3 defining the observation operator
# extracting the index of boundary elements
class observation():
    def __init__(self):
        # extracting indecies for elements at the boundary of the computational mesh
        dummy = dl.Function(solution_space)
        dummy.vector().set_local( np.ones_like( dummy.vector().get_local() ) )
        zero_bc.apply( dummy.vector() )
        self.bnd_idx = np.argwhere( dummy.vector().get_local() == 0 ).flatten()

        # defining the normal vector to cell boundaries
        self.normal_vec = dl.FacetNormal( mesh )
        self.tests = dl.TestFunction( solution_space )

    def set_u_lift(self, u_lift):
        self.u_lift = u_lift

    def obs_func(self, kappa, u):
        obs_form = dl.inner( dl.grad(u + self.u_lift), self.normal_vec )*self.tests*dl.ds
        obs = dl.assemble( obs_form ) # assembling the form on the entire boundary
        return obs.get_local()[self.bnd_idx] # returning only the boundary part of obs

# Creating a list of observation operators for frequencies k=1,2,3,4
obs_funcs = []
for i in range(4):
    temp_obs_func = observation()
    temp_obs_func.set_u_lift( forms[i].u_lift )
    obs_funcs.append( temp_obs_func )

#%% 2 paramterization and geometries
# The geometry on which the Bayesian parameters are defined correspods to the FEM paramterization
G_FEM = cuqipy_fenics.geometry.FEniCSContinuous(parameter_space)

# The KL para
G_KL = cuqipy_fenics.geometry.MaternExpansion(G_FEM, length_scale = .2, num_terms=64)

# Defining the Heaviside map
c_minus = 1
c_plus = 10
ones_vec = np.ones(94)
bnd_idx = obs_funcs[0].bnd_idx
def Heaviside(func):
    dofs = func.vector().get_local()
    updated_dofs = c_minus*0.5*(1 + np.sign(dofs)) + c_plus*0.5*(1 - np.sign(dofs))

    # Here we insure that the boundary values of the conductivity is always one
    updated_dofs[bnd_idx] = np.ones(94)
    func.vector().set_local(updated_dofs)
    return func

# creating the domain geometry which applies the map Heaviside mapt to G_KL realizations.
G_Heavi = cuqipy_fenics.geometry.FEniCSMappedGeometry(G_KL, map=Heaviside)

#%% 3 Creating the prior distribution
# Create the range geomtry 
G_cont = cuqi.geometry.Continuous1D(94)

# Create a prior
x = cuqi.distribution.Gaussian(np.zeros(G_Heavi.par_dim), cov=np.eye(G_Heavi.par_dim), geometry=G_Heavi, name='x')

#%% 4 Creating the posterior distribution
# loading signal from file
obs_data = np.load('./obs/obs_circular_inclusion_2_10per_noise.npz')
b_exact = obs_data['b_exact']
sigma2 = obs_data['sigma2']
data = obs_data['data']

# defining PDE models and data distributions for frequencies k=1,2,3,4
PDEs = [] # list of all the PDEs
As = [] # list of all forward operators
ys = [] # list of all data distributions

for i in range(4):
    if (i  == 0):
        PDE = cuqipy_fenics.pde.SteadyStateLinearFEniCSPDE( (forms[i].lhs, forms[i].rhs), mesh, solution_space, parameter_space, zero_bc, observation_operator=obs_funcs[i].obs_func, reuse_assembled=True)
    else:
        PDE = PDEs[0].with_updated_rhs(forms[i].rhs)
        PDE.observation_operator = PDE._create_observation_operator( obs_funcs[i].obs_func )

    PDEs.append( PDE )
    As.append( cuqi.model.PDEModel( PDE,range_geometry=G_cont,domain_geometry=G_Heavi)  )

    #%% 2.8 Create the data distribution using data
    ys.append( cuqi.distribution.Gaussian(mean=As[i], cov=sigma2[i]*np.ones(G_cont.par_dim), geometry=G_cont, name='y{}'.format(i+1)) )

# Creating the joint data distribution and the joint likelihood
joint = cuqi.distribution.JointDistribution(x,ys[0],ys[1],ys[2],ys[3])
posterior = joint(y1=data[0], y2=data[1], y3=data[2], y4=data[3])._as_stacked()


#%% 5 sampling
# Create Metropolis-Hastings Sampler 
Sampler = cuqi.sampler.MetropolisHastings(posterior)

# Sampling using the Metropolis-Hastings sampler
samples = Sampler.sample_adapt(100)

#%% 6 visualization

#defining labels for x axis
labels = list(range(0,36,7))

# plotting the paramters
plt.figure()
f,ax = plt.subplots(1)
samples.plot_ci(95, plot_par=True, marker='.')
ax.legend([r'Mean',r'95% CT'], loc=4)
ax.set_xticks(labels)
ax.set_xticklabels(labels)
ax.set_xlim([-1,36])
ax.set_ylim([-5,3])
ax.grid()
ax.set_xlabel(r'$i$')
ax.set_ylabel(r'$x_i$')
ax.yaxis.labelpad = -3
ax.xaxis.labelpad = 0
ax.set_title(r'(a) 5% noise')

plt.savefig('dummy.pdf',format='pdf')




