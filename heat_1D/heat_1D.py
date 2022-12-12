#!/usr/bin/env python

#%%
### Not to be included in the paper ###
import numpy as np
import matplotlib.pyplot as plt
from math import floor
import sys
import cuqi
cuqi.__version__
import os
import time


global_Ns = 1000000
use_global_Ns = True
data_folder = './data2/'

cases = []

case_name = 'paper_case1'
# We increased the number of samples to 50000
# 
# Prepare PDE form
N = 100   # Number of solution nodes
L = 1.0  # Length of the domain
T = 0.01 # Final time
dx = L/(N+1)   # Space step size
cfl = 5/11 # The cfl condition to have a stable solution
dt_approx = cfl*dx**2 # Defining approximate time step size
num_time_steps = int(T/dt_approx)+1 # Number of time steps
sampler_choice = 'MetropolisHastings'
domain_dim = N
Ns_factor = domain_dim if sampler_choice == 'CWMH' else 1
Ns =  int(global_Ns/Ns_factor) if use_global_Ns  else 50000 # Number of samples
scale = 0.001
dg = 'Continuous1D'
cov = 0.05**2
x0=np.zeros(domain_dim)*0.1
cases.append({'case_name':case_name, 'N':N, 'L':L, 'T':T, 'dx':dx, 'cfl':cfl, 'dt_approx':dt_approx, 'num_time_steps':num_time_steps, 'Ns':Ns, 'domain_dim':domain_dim, 'scale':scale, 'dg':dg, 'cov':cov, 'x0':x0, 'sampler_choice':sampler_choice})



case_name = 'paper_case2'
# Very Good results, min ESS ~60, I will use this case for the paper
# unless we decide to increase the final time. Can display 50% CI
# Prepare PDE form
N = 100   # Number of solution nodes
L = 1.0  # Length of the domain
T = 0.01 # Final time
dx = L/(N+1)   # Space step size
cfl = 5/11 # The cfl condition to have a stable solution
dt_approx = cfl*dx**2 # Defining approximate time step size
num_time_steps = int(T/dt_approx)+1 # Number of time steps
domain_dim = 20
sampler_choice = 'CWMH'
Ns_factor = domain_dim if sampler_choice == 'CWMH' else 1
Ns =  int(global_Ns/Ns_factor) if use_global_Ns  else 1000 # Number of samples
scale = np.ones(domain_dim)
scale[0] = 0.05
scale[1] = 0.1
scale[2] = 0.2
# Added later
dg = 'KL'
cov = 1
x0=None

cases.append({'case_name':case_name, 'N':N, 'L':L, 'T':T, 'dx':dx, 'cfl':cfl, 'dt_approx':dt_approx, 'num_time_steps':num_time_steps, 'Ns':Ns, 'domain_dim':domain_dim, 'scale':scale, 'dg':dg, 'cov':cov, 'x0':x0, 'sampler_choice':sampler_choice})




for case in cases:
    print("Sampling for case", case['case_name'])
    print(case)
    case_name = case['case_name']
    N = case['N']
    L = case['L']
    T = case['T']
    dx = case['dx']
    cfl = case['cfl']
    dt_approx = case['dt_approx']
    num_time_steps = case['num_time_steps']
    Ns = case['Ns']
    domain_dim = case['domain_dim']
    scale = case['scale']
    dg = case['dg']
    cov = case['cov']
    x0 = case['x0']
    sampler_choice = case['sampler_choice']
    

    # Grid for the heat model
    grid = np.linspace(dx, L, N, endpoint=False)
    
    # Time steps
    time_steps = np.linspace(0, T, num_time_steps, endpoint=True)
    
    # FD diffusion operator
    Dxx = (np.diag(-2*np.ones(N)) + np.diag(np.ones(N-1), -1)
           + np.diag(np.ones(N-1), 1))/dx**2  
    
    
    # PDE form (diff_op, IC, time_steps)
    def PDE_form(initial_condition, t): return (Dxx, np.zeros(N),
                                                initial_condition)
    
    PDE = cuqi.pde.TimeDependentLinearPDE(
        PDE_form, time_steps, grid_sol=grid)
    

    
    # Set up geometries for model
    if (dg == 'KL'): # (paper) remove the options
        domain_geometry = cuqi.geometry.KLExpansion(grid, decay_rate=1.7,
                                                normalizer=12, 
                                                num_modes=domain_dim)
    elif (dg == 'Continuous1D'):
        domain_geometry = cuqi.geometry.Continuous1D(grid)
    
    range_geometry = cuqi.geometry.Continuous1D(grid)
    
    # Prepare model
    model = cuqi.model.PDEModel(PDE, range_geometry, domain_geometry)
    
    # Create the prior distribution
    x = cuqi.distribution.Gaussian(np.zeros(model.domain_dim), cov=cov,
                                   geometry=domain_geometry)
    
    
    # Now samples from the prior will look like:
    
    ### Not to be included in the paper ###

    prior_samples = x.sample(5) 
    plt.figure()
    for s in prior_samples:
        domain_geometry.plot(s, is_par=True)
    
    # True parameters that we want to infer
    x_exact_raw = grid*np.exp(-2*grid)*np.sin(L-grid)
    x_exact = cuqi.samples.CUQIarray(x_exact_raw, is_par=False,
                                     geometry=domain_geometry)
    

    # Generate the exact data
    y_exact = model.forward(x_exact)
    

    
    ### Not to be included in the paper ###
    sigma =1.0/np.sqrt(N)* 0.01*np.linalg.norm(y_exact)
    

    
    # Create the data distribution
    y = cuqi.distribution.Gaussian(model(x),
                                   sigma**2*np.eye(model.range_dim),
                                   geometry=range_geometry)

    # Generate noisy data
    data = y(x = x_exact).sample()
    plt.figure()
    x_exact.plot()
    y_exact.plot()
    data.plot()
    plt.legend(['exact solution', 'exact data', 'noisy data']);
    
    
    #plt.plot(data-y_exact)
    print(np.linalg.norm(data-y_exact))
    print(np.linalg.norm(y_exact))
    print(np.linalg.norm(data-y_exact)/np.linalg.norm(y_exact))
    
    
    
    # Bayesian model
    joint_distribution = cuqi.distribution.JointDistribution(x, y)
    posterior = joint_distribution(y = data) 
    

    if (sampler_choice == 'MetropolisHastings'):
        MySampler = cuqi.sampler.MetropolisHastings(posterior, scale = scale, x0=x0)
    elif (sampler_choice == 'CWMH'):
        MySampler = cuqi.sampler.CWMH(posterior, scale = scale, x0=x0)
    
    t1 = time.time()    
    posterior_samples = MySampler.sample_adapt(Ns)
    t2 = time.time()
    case["sampling_time"] = t2-t1
    
    ### Not to be included in the paper ###
    case["updated_scale"] = MySampler.scale
    case["ESS"] = posterior_samples.compute_ess()
    
    
    
    # Create the reustls folder if it does not exist
    if not os.path.exists(data_folder+case_name):
        os.makedirs(data_folder+case_name)
    
    # Save samples:
    import pickle
    pickle.dump(posterior_samples, open(data_folder+case_name + '/samples.pkl', 'wb'))

    # Save the case parameters
    pickle.dump(case, open(data_folder+case_name + '/parameters.pkl', 'wb'))
    
    # Save prior plot
    plt.figure()
    for s in prior_samples:
        domain_geometry.plot(s, is_par=True)
    plt.savefig(data_folder+case_name + '/prior.png')

    # Plot true solution, exact data and noisy data
    plt.figure()
    x_exact.plot()
    y_exact.plot()
    data.plot()
    plt.legend(['exact solution', 'exact data', 'noisy data']);
    plt.savefig(data_folder+case_name + '/sol_data.png')

    # Plot ESS
    plt.figure()
    plt.plot(case["ESS"], 'o-')
    plt.savefig(data_folder+case_name + '/ESS.png')

    # Plot trace
    plt.figure()
    posterior_samples.plot_trace([0,1,2,3,4,5,6,7,8,9,10]);
    plt.savefig(data_folder+case_name + '/trace.png')

    # Plot pair
    plt.figure()
    posterior_samples.plot_pair([0,1,5,10]);
    plt.savefig(data_folder+case_name + '/pair_plot.png')

    # Plot the ci as par and save the plot
    plt.figure()
    posterior_samples.plot_ci(95, plot_par=True)
    plt.savefig(data_folder+case_name + '/plot_ci_par.png')

    # Plot ci as funvals and save the plot
    plt.figure()
    posterior_samples.plot_ci(95, exact=x_exact)
    plt.savefig(data_folder+case_name + '/plot_ci.png')
    
    # Plot the ci after funvals conversion
    plt.figure()
    posterior_samples.funvals.plot_ci(95, exact=x_exact)
    plt.savefig(data_folder+case_name + '/plot_ci_funvals.png')
