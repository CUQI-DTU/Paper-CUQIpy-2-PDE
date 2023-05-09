import numpy as np
from dolfin import *
import dolfin as dl
import cuqi
import matplotlib.pyplot as plt
from wave import wave

#import sys
#sys.path.append('./CUQIpy-FEniCS')
import cuqipy_fenics

# This script solve the Photo-Acoustic with full bounary data

#%% 1 setting up FEniCS function spaces
mesh = UnitIntervalMesh(120) # defining the mesh
parameter_space = FunctionSpace(mesh,'CG', 1) # defining the function space

#%% 2 setting domain and rage geometry and mappings
# The geometry on which the Bayesian parameters are defined correspods to the FEM paramterization
G_FEM = cuqipy_fenics.geometry.FEniCSContinuous(parameter_space)
# The KL parameterization
G_KL = cuqipy_fenics.geometry.MaternKLExpansion(G_FEM, length_scale=0.1, nu=0.75, num_terms=100)

# This map scales the output of G_KL with a constant factor
def prior_map(func):
    dofs = func.vector().get_local()
    updated_dofs = 15*dofs
    func.vector().set_local(updated_dofs)
    return func

# Defining the domain geometry
G = cuqipy_fenics.geometry.FEniCSMappedGeometry(G_KL, map = prior_map)

#%% 3 Defining the photo-acoustic forward operator
# loading the blackbox forward operator
problem = wave()
# The function that maps the initial pressure to the boundary observations
PAT = problem.forward_full # uncomment for full boundary data
#PAT = problem.forward_half # uncomment for half boundary data

# loading signal 
obs_data = np.load( './obs/full_boundary_5per.npz' ) # uncomment for full boundary data
#obs_data = np.load( './obs/half_boundary_5per.npz' ) # uncomment for half boundary data
data = obs_data['data']
sigma2 = obs_data['sigma2']
b_exact = obs_data['b_exact'].reshape(251,-1)
# Defining the range geometry
m = data.shape[0] # dimension of the observation
G_cont = cuqi.geometry.Continuous1D( m )

# defining the CUQIpy forward operator
A = cuqi.model.Model(PAT,domain_geometry=G, range_geometry=G_cont)

#%% 4 creating prior distribution
x = cuqi.distribution.Gaussian(0, cov=np.eye(G.par_dim), geometry=G) # 

#%% 5 creating data distribution
# defining data distribution
y = cuqi.distribution.Gaussian(A(x), sigma2*np.ones(data.shape[0]), geometry=G_cont)

# defining the joint and the posterior distributions
joint = cuqi.distribution.JointDistribution(x, y)
posterior = joint(y=data)

#%% 5 sampling the posterior
# defining the pCN sampler and sampling
num_samples=200000
sampler = cuqi.sampler.pCN(posterior)
samples = sampler.sample_adapt(num_samples)

#%% 6 visualization and plotting
# generating a numpy array of the function values of the samples
continuous_samples = G.par2fun(samples.samples)
fun_vals = []
for i in range(num_samples):
    fun_vals.append( continuous_samples[i].vector().get_local()[::-1] )
fun_vals = np.array( fun_vals )

# computing a visualization grid
grid = np.linspace(0,1,121)
G_vis = cuqi.geometry.Continuous1D(grid)


cuqi_continuous_samples = cuqi.samples.Samples(fun_vals.T, geometry=G_vis)


# loading the true initial pressure profile
init_pressure_data = np.load( './obs/init_pressure.npz' )
exact_solution = init_pressure_data['init_pressure']

# plotting the data
t = np.linspace(0,1,251)
labels = np.linspace(0,1,5)
if( b_exact.shape[1] == 2 ):
    f, ax = plt.subplots(1,2)
    data = data.reshape(251,-1)
    ax[0].plot(t,data[:,0])
    ax[0].plot(t,b_exact[:,0])
    ax[0].set_xticks(labels)
    ax[0].set_xticklabels(labels)
    ax[0].set_xlim([-.05,1.05])
    ax[0].set_ylim([-0.05,.55])
    ax[0].set_xlabel(r'$\tau$')
    ax[0].set_ylabel(r'$u(\xi_L)$')
    ax[0].set_title(r'pressure, left boudnary')
    ax[0].grid()
    ax[1].plot(t,data[:,1])
    ax[1].plot(t,b_exact[:,1])
    ax[1].set_xticks(labels)
    ax[1].set_xticklabels(labels)
    ax[1].set_xlim([-.05,1.05])
    ax[1].set_ylim([-0.05,.55])
    ax[1].set_xlabel(r'$\tau$')
    ax[1].set_ylabel(r'$u(\xi_L)$')
    ax[1].set_title(r'pressure, right boudnary')
    ax[1].grid()
    ax[1].legend([r'noisy data',r'exact data'], loc=1)
else:
    f, ax = plt.subplots(1)
    ax.plot(t,data)
    ax.plot(t,b_exact[:,0])
    ax.set_xticks(labels)
    ax.set_xticklabels(labels)
    ax.set_xlim([-.05,1.05])
    ax.set_ylim([-0.05,.55])
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$u(\xi_L)$')
    ax.set_title(r'pressure, left boudnary')
    ax.grid()
    ax.legend([r'noisy data',r'exact data'], loc=1)

# plotting the the posterior mean and the uncertainty on the G_vis geometry
f, ax = plt.subplots(1)
cuqi_continuous_samples.plot_ci(95, exact=exact_solution)
ax.legend([r'95% CI',r'Mean',r'Exact'], loc=1)
ax.set_xlim([-.05,1.05])
ax.set_ylim([-0.3,0.7])
ax.set_xlabel(r'$\xi$')
ax.set_ylabel(r'$g$')
ax.grid()
ax.set_title(r'estimated initial pressure')

# plotting the the posterior mean and the uncertainty for the Bayesian paramters
f, ax = plt.subplots(1)
samples.plot_ci(95, plot_par=True, marker='.')
ax.legend([r'Mean',r'95% CT'], loc=4)
ax.set_xlim([-1,25])
ax.set_ylim([-2.5,2.5])
ax.grid()
ax.set_xlabel(r'$i$')
ax.set_ylabel(r'$x_i$')
ax.set_title(r'estimated Bayesian paramters')


