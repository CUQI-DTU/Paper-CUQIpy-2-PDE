#%% Importing the required libraries
import numpy as np
from dolfin import *
import matplotlib.pyplot as plt
from wave import wave
from cuqi.geometry import Continuous2D, Continuous1D
from cuqi.distribution import Gaussian, JointDistribution
from cuqi.sampler import pCN
from cuqi.array import CUQIarray
from cuqi.model import Model
from cuqipy_fenics.geometry import FEniCSContinuous, MaternKLExpansion,\
FEniCSMappedGeometry
import os

def create_domain_geometry(parameter_space):
    """
    This function creates the domain geometry for the PAT problem.
    """
    # The geometry on which the Bayesian parameters are defined corresponds to
    # the FEM parameterization
    G_FEM = FEniCSContinuous(parameter_space)
    # The KL parameterization
    G_KL = MaternKLExpansion(G_FEM, length_scale=0.1, nu=0.75, num_terms=100)
    
    # This map scales the output of G_KL with a constant factor
    def prior_map(func):
        dofs = func.vector().get_local()
        updated_dofs = 15*dofs
        func.vector().set_local(updated_dofs)
        return func
    
    # Defining the domain geometry
    G = FEniCSMappedGeometry(G_KL, map=prior_map)
    return G

if __name__ == "__main__":
    # Fix the random seed for reproducibility
    np.random.seed(0)
    
    # This script solve the Photo-Acoustic with full or partial boundary data
    full_data = True # if false, data are obtained only at the left boundary
                      # otherwise, data are obtained on both boundaries. 
    
    #%% 1 Setting up FEniCS function spaces
    mesh = UnitIntervalMesh(120) # defining the mesh
    parameter_space = FunctionSpace(mesh,'CG', 1) # defining the function space
    
    #%% 2 Setting domain geometry
    G = create_domain_geometry(parameter_space)
    
    #%% 3 Defining the photo-acoustic forward operator
    # Loading the blackbox forward operator
    problem = wave()
    # The function `PAT` that maps the initial pressure to the boundary
    # observations
    if full_data:
        PAT = problem.forward_full
        r = 2 # number of sensors
        label = 'full'
    else:
        PAT = problem.forward_half
        r = 1 # number of sensors
        label = 'half'
    
    # Loading the data signal
    obs_data = np.load( './obs/'+label+'_boundary_5per.npz' )
    
    y_obs = obs_data['data']
    s_noise = np.sqrt(obs_data['sigma2'].reshape(-1)[0])
    b_exact = obs_data['b_exact'].reshape(251,-1)
    
    #%% 4 Defining the range geometry
    m = 251 # dimension of the observation
    obs_times = np.linspace(0,1,m)
    
    if full_data:
        obs_locations = np.array([0.001, 0.999])
    else:
        obs_locations = np.array([0.001])
    
    G_cont = Continuous2D((obs_times, obs_locations))
    
    #%% 5 Defining the CUQIpy forward operator
    A = Model(PAT, domain_geometry=G, range_geometry=G_cont)
    
    #%% 6 Creating prior distribution
    x = Gaussian(0, 1, geometry=G)
    
    #%% 7 Creating data distribution
    # Defining data distribution
    y = Gaussian(A(x), s_noise**2, geometry=G_cont)
    
    #%% 8 Defining the joint and the posterior distributions
    joint = JointDistribution(x, y)
    posterior = joint(y=y_obs)
    
    #%% 9 Sampling the posterior
    # Defining the pCN sampler and sampling
    sampler = pCN(posterior)
    samples = sampler.sample_adapt(50000)
    
    # Thin the samples
    samples = samples.burnthin(0, 100)
    
    #%% 10 Visualization and plotting
    
    # Loading the true initial pressure profile
    init_pressure_data = np.load('./obs/init_pressure.npz')
    g_true = init_pressure_data['init_pressure']
    
    # Create FEniCS function for g_true wrapped in CUQIarray
    g_true_function = Function(parameter_space)
    g_true_function.vector().set_local(g_true[::-1])
    g_true = CUQIarray(
        g_true_function, is_par=False, geometry=G)
    # Plotting the data
    # create `plots` directory if it does not exist
    if not os.path.exists('plots'):
        os.makedirs('plots') 
    t = np.linspace(0,1,251)
    labels = np.linspace(0,1,5)
    if full_data:
        f, ax = plt.subplots(1,2)
        y_obs = y_obs.reshape(251,-1)
        ax[0].plot(t,y_obs[:,0])
        ax[0].plot(t,b_exact[:,0])
        ax[0].set_xticks(labels)
        ax[0].set_xticklabels(labels)
        ax[0].set_xlim([-.05,1.05])
        ax[0].set_ylim([-0.05,.55])
        ax[0].set_xlabel(r'$\tau$')
        ax[0].set_ylabel(r'$u(\xi_L)$')
        ax[0].set_title(r'pressure, left boundary')
        ax[0].grid()
        ax[1].plot(t,y_obs[:,1])
        ax[1].plot(t,b_exact[:,1])
        ax[1].set_xticks(labels)
        ax[1].set_xticklabels(labels)
        ax[1].set_xlim([-.05,1.05])
        ax[1].set_ylim([-0.05,.55])
        ax[1].set_xlabel(r'$\tau$')
        ax[1].set_ylabel(r'$u(\xi_L)$')
        ax[1].set_title(r'pressure, right boundary')
        ax[1].grid()
        ax[1].legend([r'noisy data',r'exact data'], loc=1)
    else:
        f, ax = plt.subplots(1)
        ax.plot(t,y_obs)
        ax.plot(t,b_exact[:,0])
        ax.set_xticks(labels)
        ax.set_xticklabels(labels)
        ax.set_xlim([-.05,1.05])
        ax.set_ylim([-0.05,.55])
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel(r'$u(\xi_L)$')
        ax.set_title(r'pressure, left boundary')
        ax.grid()
        ax.legend([r'noisy data',r'exact data'], loc=1)
    plt.savefig("./plots/data_"+label+".png")
    
    # Plotting the the posterior mean and the uncertainty on the continuous
    # domain
    f, ax = plt.subplots(1)
    samples.funvals.vector.plot_ci(95, exact=g_true)
    ax.legend([r'95% CI',r'Mean',r'Exact'], loc=1)
    ax.set_xlim([-.05,1.05])
    ax.set_ylim([-0.3,0.7])
    ax.set_xlabel(r'$\xi$')
    ax.set_ylabel(r'$g$')
    ax.grid()
    ax.set_title(r'estimated initial pressure')
    plt.savefig("./plots/posterior_cont_"+label+".png")
    
    # Plotting the the posterior mean and the uncertainty for the KL parameters
    f, ax = plt.subplots(1)
    samples.plot_ci(95, plot_par=True, marker='.')
    ax.legend([r'Mean',r'95% CT'], loc=4)
    ax.set_xlim([-1,25])
    ax.set_ylim([-2.5,2.5])
    ax.grid()
    ax.set_xlabel(r'$i$')
    ax.set_ylabel(r'$x_i$')
    ax.set_title(r'estimated Bayesian parameters')
    plt.savefig("./plots/posterior_par_"+label+".png")
    
    # %%
    # Saving the samples
    # create `stat` directory if it does not exist
    if not os.path.exists('stat'):
        os.makedirs('stat')
    # save the samples
    np.savez('stat/samples_thinned_'+label+'.npz', samples=samples.samples)
    