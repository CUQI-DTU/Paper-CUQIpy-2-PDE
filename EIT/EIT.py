import numpy as np
import dolfin as dl
import matplotlib.pyplot as plt
import os
import sys
from cuqi.geometry import Continuous1D
from cuqi.model import PDEModel
from cuqi.distribution import Gaussian, JointDistribution
from cuqi.sampler import MH
from cuqipy_fenics.geometry import FEniCSContinuous, MaternKLExpansion,\
                                   FEniCSMappedGeometry 
from cuqipy_fenics.pde import SteadyStateLinearFEniCSPDE
import cuqi
import cuqipy_fenics

# Function for extracting the indices of the boundary nodes
def extract_boundary_dofs_indices(solution_space):

    v0 = dl.Constant('0.0')
    boundary = lambda x, on_boundary: on_boundary
    zero_bc = dl.DirichletBC(solution_space, v0, boundary)
    dummy = dl.Function(solution_space)
    dummy.vector().set_local(np.ones_like(dummy.vector().get_local()))
    zero_bc.apply(dummy.vector())
    bnd_idx = np.argwhere(
        dummy.vector().get_local()==0).flatten() # this holds the indecies of
                                                 # the boundary elements
    return bnd_idx

def create_domain_geometry(parameter_space, bnd_idx):
    # The geometry on which the Bayesian parameters are defined corresponds to
    # the FEM parameterization
    G_FEM = FEniCSContinuous(parameter_space)
    
    # The KL parameterization
    G_KL = MaternKLExpansion(G_FEM, length_scale=0.2, num_terms=64)
    
    # Defining the Heaviside map
    c_minus = 1
    c_plus = 10
    def Heaviside(func):
        dofs = \
            func.vector().get_local() # extracting the function values at FEM 
                                      # nodes (this only works for linear 
                                      # element)
        updated_dofs = c_minus*0.5*(1 + np.sign(dofs))+\
                       c_plus*0.5*(1 - np.sign(dofs))
    
        # Here we insure that the boundary values of the conductivity is 
        # always one
        updated_dofs[bnd_idx] = np.ones_like(bnd_idx)
        func.vector().set_local(updated_dofs)
        return func
    
    # creating the domain geometry which applies the map Heaviside map to G_KL 
    # realizations.
    G_Heavi = FEniCSMappedGeometry(G_KL, map=Heaviside)
    return G_Heavi

if __name__ == "__main__":
    # Parse command line arguments: noise level, number of samples,
    # number of burn-in samples, thinning factor, and random seed.
    # Parse noise level which is passed as a command line argument. Only
    # 5, 10, and 20 percent noise levels are supported.

    if len(sys.argv) != 6:
        print("Usage: python EIT.py <noise_percent> <num_samples> <num_burnin> <thinning_factor> <random_seed>")
        sys.exit(1)

    noise_percent = int(sys.argv[1])
    Ns = int(sys.argv[2])
    Nb = int(sys.argv[3])
    Nt = int(sys.argv[4])
    seed = int(sys.argv[5])

    # Check if noise level is supported
    if noise_percent not in [5, 10, 20]:
        print("Only 5, 10, and 20 percent noise levels are supported")
        sys.exit(1)
    print("Running EIT with noise level: ", noise_percent, "%")

    # Print cuqi and cuqipy_fenics version
    print("cuqi version: ", cuqi.__version__)
    print("cuqipy_fenics version: ", cuqipy_fenics.__version__)

    # Fix the random seed for reproducibility 
    np.random.seed(seed)
    
    #%% 1 setting up FEniCS
    # loading computational mesh
    mesh = dl.Mesh("mesh.xml")
    
    #%% 1.1 Define function spaces 
    parameter_space = dl.FunctionSpace(mesh, "CG", 1)
    solution_space = dl.FunctionSpace(mesh, "CG", 1)
    
    #%% 1.2 defining the lifted weak formulation
    
    # Function that marks the boundary of the computational mesh
    boundary = lambda x, on_boundary: on_boundary
    
    # Creating lhs form
    form_lhs = lambda sigma, v, t: dl.inner(sigma*dl.grad(v), dl.grad(t))*dl.dx
    
    # Creating rhs for frequencies k=1,2,3,4
    # Creating the boundary condition expression and DirichletBC to be used
    # in building the rhs (using lifting).
    boundary_expression = dl.Expression(
        "sin(k*atan2(x[1], x[0]))", k=1, degree=1)
    bc = dl.DirichletBC(solution_space,
                        boundary_expression,
                        boundary) # applying this boundary condition on a
                                  # function will create u_lift for the 
                                  # frequency in bc_func
    
    # creating functions for the boundary condition u_lift_1, u_lift_2,
    # u_lift_3, u_lift_4
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
    # Defining zero boundary for the lifted problem
    v0 = dl.Constant('0.0')
    zero_bc = dl.DirichletBC(solution_space, v0, boundary)
    # extracting indices for elements at the boundary of the computational mesh
    bnd_idx = extract_boundary_dofs_indices(solution_space)
    
    # defining normal vectors to the cell boundaries
    n = dl.FacetNormal( mesh )
    # defining FEniCS test functions
    w = dl.TestFunction( solution_space )
    
    # defining a function that returns the values at the boundaries
    def give_bnd_vals(obs_form):
        return obs_form.get_local()[bnd_idx]
    
    # defining the observation functions
    def observation1(sigma, v1):
        obs_form = dl.inner(dl.grad(v1 + u_lift_1), n)*w*dl.ds
        assembled_form = dl.assemble(obs_form)
        boundary_values = give_bnd_vals(assembled_form)
        return boundary_values
    
    def observation2(sigma, v2):
        obs_form = dl.inner(dl.grad(v2 + u_lift_2), n)*w*dl.ds
        assembled_form = dl.assemble(obs_form)
        boundary_values = give_bnd_vals(assembled_form)
        return boundary_values
    
    def observation3(sigma, v3):
        obs_form = dl.inner(dl.grad(v3 + u_lift_3), n)*w*dl.ds
        assembled_form = dl.assemble(obs_form)
        boundary_values = give_bnd_vals(assembled_form)
        return boundary_values
    
    def observation4(sigma, v4):
        obs_form = dl.inner(dl.grad(v4 + u_lift_4), n)*w*dl.ds
        assembled_form = dl.assemble(obs_form)
        boundary_values = give_bnd_vals(assembled_form)
        return boundary_values
    
    #%% 2 parameterization and geometries
    # Create the domain geometry
    G_Heavi = create_domain_geometry(parameter_space, bnd_idx)

    #%% 3 Creating the prior distribution
    # Create the range geometry 
    m = len(bnd_idx)
    G_cont = Continuous1D(m)
    
    # Create a prior
    n_KL = G_Heavi.par_dim
    x = Gaussian(np.zeros(n_KL), 1, geometry=G_Heavi)
    
    #%% 4 Creating the posterior distribution
    # loading signal from file
    obs_data = np.load('./data/obs_circular_inclusion_2_'+str(noise_percent)+'per_noise.npz')
    b_exact = obs_data['b_exact']
    s_noise_list = np.sqrt(obs_data['sigma2']) # read the noise variance and
                                               # convert to std
    
    data = obs_data['data']
    y1_obs = data[0]
    y2_obs = data[1]
    y3_obs = data[2]
    y4_obs = data[3]
    
    # creating PDE forms
    PDE_form1 = (form_lhs, form_rhs1)
    
    # creating PDE models
    # for the first PDE problems we specify to reuse the factorization of the 
    # lhs for the rest of the PDE models
    PDE1 = SteadyStateLinearFEniCSPDE(
        PDE_form1,
        mesh,
        solution_space,
        parameter_space,
        zero_bc,
        observation_operator=observation1,
        reuse_assembled=True)
    
    # We copy the PDE1 for the rest of the PDE problems with updated rhs
    PDE2 = PDE1.with_updated_rhs(form_rhs2)
    PDE2.observation_operator = observation2
    PDE3 = PDE1.with_updated_rhs(form_rhs3)
    PDE3.observation_operator = observation3
    PDE4 = PDE1.with_updated_rhs(form_rhs4)
    PDE4.observation_operator = observation4
    
    # Creating the forward operators
    A1 = PDEModel(PDE1, range_geometry=G_cont, domain_geometry=G_Heavi)
    A2 = PDEModel(PDE2, range_geometry=G_cont, domain_geometry=G_Heavi)
    A3 = PDEModel(PDE3, range_geometry=G_cont, domain_geometry=G_Heavi)
    A4 = PDEModel(PDE4, range_geometry=G_cont, domain_geometry=G_Heavi)
    
    # creating data distributions
    s_noise = s_noise_list[0]
    y1 = Gaussian(A1(x), s_noise**2, geometry=G_cont)
    
    s_noise = s_noise_list[1]
    y2 = Gaussian(A2(x), s_noise**2, geometry=G_cont)
    
    s_noise = s_noise_list[2]
    y3 = Gaussian(A3(x), s_noise**2, geometry=G_cont)
    
    s_noise = s_noise_list[3]
    y4 = Gaussian(A4(x), s_noise**2, geometry=G_cont)
    
    # Creating the joint data distribution and the joint likelihood
    joint = JointDistribution(x, y1, y2, y3, y4)
    posterior = joint(y1=y1_obs, y2=y2_obs, y3=y3_obs, y4=y4_obs)
    
    #%% 5 sampling
    # Create Metropolis-Hastings Sampler 
    sampler = MH(posterior)
    
    # Sampling using the Metropolis-Hastings sampler
    posterior_samples = sampler.sample_adapt(Ns)
    
    #%% 6 visualization
    # plotting prior samples
    f, axes = plt.subplots(1,3)
    plt.sca(axes[0])
    prior_sample = x.sample()
    prior_sample.plot(subplots=False)
    plt.sca(axes[1])
    prior_sample = x.sample()
    prior_sample.plot(subplots=False)
    plt.sca(axes[2])
    prior_sample = x.sample()
    prior_sample.plot(subplots=False)
    axes[1].set_title('prior samples')
    plt.savefig("plot_prior_samples"+str(noise_percent)+".png")
    
    # plotting posterior samples
    idx = np.random.permutation(Ns) # create randomized index
    f, axes = plt.subplots(1,3)
    plt.sca(axes[0])
    posterior_samples.plot(idx[0],subplots=False)
    plt.sca(axes[1])
    posterior_samples.plot(idx[1],subplots=False)
    plt.sca(axes[2])
    posterior_samples.plot(idx[2],subplots=False)
    axes[1].set_title('posterior samples')
    plt.savefig("plot_posterior_samples"+str(noise_percent)+".png")
    
    # burn-thin the samples
    posterior_samples = posterior_samples.burnthin(Nb, Nt)
    
    # plotting the mean
    f, axes = plt.subplots(1,2)
    plt.sca(axes[0])
    posterior_samples.plot_mean(subplots=False)
    axes[0].set_title('sample mean')
    
    # plotting the variance
    plt.sca(axes[1])
    posterior_samples.funvals.vector.plot_variance(subplots=False)
    axes[1].set_title('variance')
    plt.savefig("plot_mean_variance"+str(noise_percent)+".png")
    
    # plotting the credible intervals
    plt.figure()
    posterior_samples.plot_ci(95, plot_par=True)
    plt.savefig("plot_ci"+str(noise_percent)+".png")
    
    # Create directory for saving the results if it does not exist
    if not os.path.exists('./stat'):
        os.makedirs('./stat')

    # Save the posterior samples
    np.savez("./stat/stat_circular_inclusion_2_"
             +str(noise_percent)+"per_noise_thinned.npz",
             samples=posterior_samples.samples)
    