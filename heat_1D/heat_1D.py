#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Not to be included in the paper ###
import numpy as np
import matplotlib.pyplot as plt
from math import floor
import sys
import cuqi
cuqi.__version__


# 

# # CUQIpy example: 1D Heat problem
# 
# Here we go through the steps of creating a time dependant PDE-based Bayesian problem within the CUQIpy library. The problem we consider is a one dimensional (1D) initial-boundary value heat equation with zero boundary conditions.
# 
# \begin{align}
# \frac{\partial u(x,t)}{\partial t} - c^2 \frac{\partial^2 u(x,t)}{\partial x^2}   & = f(x,t), \;x\in[0,L],\; 0\le t \le T\\
# u(0,t)= u(L,t)&= 0\\
# u(x,0)&= g(x) 
# \end{align}
# 
# 
# where $u(x,t)$ is the temperature and $c^2$ is the thermal diffusivity (assumed to be 1 here). We assume the source term $f$ is zero. The unknown Bayesian parameters (random variable) for this test problem is the initial heat profile $g(x)$.
# 
# The data we obtain for this problem is the measurement of the temperature profile in the domain at time $T$. We assume that the measurement error $\eta$ follows a Gaussian distribution. In a Bayesian setting, we can represent the data as a random variable $y$:
# 
# \begin{align}
# y = \mathcal{G}(\theta) + \eta, \;\;\; \eta\sim\mathcal{N}(0,\sigma_\text{noise}^2\mathbf{I}),
# \end{align}
# 
# where $\mathcal{G}(\theta)$ is the forward model that maps the initial condition $\theta$ to the final time solution $u(x,T)$ via solving the 1D time-dependent heat problem. For this test case, $T=0.01$, $L=1$, relative noise level is $1\%$, and the number of grid nodes for the finite difference discretization is $100$.
# 
# In the remaining of this section, we create and solve the inverse heat equation in a Bayesian setting, i.e. constructing the posterior distribution for $\theta(x)$ given some observed data $y_\text{obs}$.

# ### 1. Create the PDE problem

# In[ ]:


### Not to be included in the paper ###

# Prepare PDE form
N = 80   # Number of solution nodes
L = 1.0  # Length of the domain
T = 0.04 # Final time
dx = L/(N+1)   # Space step size
cfl = 5/11 # The cfl condition to have a stable solution
dt_approx = cfl*dx**2 # Defining approximate time step size
num_time_steps = int(T/dt_approx)+1 # Number of time steps


# Here we set the physical and numerical parameters and create the components that we need to define the 1D time-dependant heat problem. We discretize the heat problem on a grid of  `N=80` nodes. We choose the domain length `L=1`, and the time `T=0.04`. We discretize the time interval $[0,T]$ into `num_time_steps=434` time steps. In the following, we define three python objects: a `numpy.ndarray` representing the spatial grid `grid`, a `numpy.ndarray` array representing the time steps `time_steps` and a `numpy.ndarray` representing the discretized diffusion differential operator using centered difference. 

# In[ ]:


# Grid for the heat model
grid = np.linspace(dx, L, N, endpoint=False)

# Time steps
time_steps = np.linspace(0, T, num_time_steps, endpoint=True)

# PDE form (diff_op, IC, time_steps)
Dxx = (np.diag( -2*np.ones(N) ) + np.diag(np.ones(N-1),-1) + np.diag(np.ones(N-1),1))/dx**2 # FD diffusion operator


# ### 2. Create `cuqi.pde.TimeDependentLinearPDE` object

# The details of the PDE problem is encapsulated in a $\code{cuqi.pde.PDE}$ object. For 
# this time-dependent problem we create `cuqi.pde.TimeDependentLinearPDE` object.
#  This object needs information about the grid `grid`, the time steps 
# `time_steps`, and a representation of the PDE `PDE_form` at a given time `t`.
# As an optional argument, one can also specify whether the time discretization
#  scheme is implicit or explicit Euler. the user can also provide advanced 
#  arguments such as the linear solver to be used.
# 
# This representation `PDE_form` is a python function that accepts as the first
# argument an instance of the inverse problem parameter, the `initial_condition`
# here, and the time `t` as a second argument. It returns a tuple of the 
# differential operator at time `t`, the right hand side at time `t` and the
# initial condition. 

# 

# In[ ]:


def PDE_form(initial_condition, t): return (Dxx, np.zeros(N), initial_condition)

PDE = cuqi.pde.TimeDependentLinearPDE(
    PDE_form, time_steps, grid_sol=grid)


# ### 3. Create the forward model

# Next we create a `cuqipy` forward model. We create a `cuqi.model.PDEModel` which is a subclass of `cuqi.model.Model`. To initialize an object of this class, we pass a `cuqi.pde.PDE` along with two `cuqi.geometry.Geometry` objects to represent the domain and the range of the forward PDE problem. `cuqi.model.PDEModel` is agnostic to the underlying details of the PDE. It interact with the `PDE` object through the functions `assemble`, `solve`, and `observe`. 
# 
# The domain geometry represents the domain of the forward problem. For the heat equation the domain geometry represents the function space of the discretized $g(x)$. To impose some regularity on the initial condition $g(x)$, we parametrize it using Karhunen–Loève (KL) expansion  
# 
# $$g(x_j) = u(x_j,0) = \sum_{i=0}^{N-2} \left(\frac{1}{(i+1)^\gamma\tau}\right)  \theta_i \, \text{sin}\left(\frac{\pi}{N}(i+1)(j+\frac{1}{2})\right) + \frac{(-1)^j}{2}\left(\frac{1}{N^\gamma\tau}\right)  \theta_{N-1},$$
# 
#     
# where $x_j$ is the $j^\text{th}$ grid point (in a regular grid), $j=0, 1, 2, 3, ..., N-1$, $\gamma$ is the decay rate, $\tau$ is a normalization constant, and $\theta_i$ are the expansion coefficients. We note that using the KL-expansion parameterization, the Bayesian parameters becomes the coefficients of expansion $\theta_i$. We set up the domain geometry as a `cuqi.geometry.KLExpansion` object and pass the arguments `decay_rate=1.7` and `normalizer=12` for the decay rate and the normalization constants, respectively.
# 
# The range geometry represents the function space of the observed data, $u(0,T)$ in this case, which can be represented by a `cuqi.geometry.Continuous1D` object.

# In[ ]:


# Set up geometries for model
domain_geometry = cuqi.geometry.KLExpansion(grid, decay_rate=1.7, normalizer=12)
range_geometry = cuqi.geometry.Continuous1D(grid)

# Prepare model
model = cuqi.model.PDEModel(PDE,range_geometry,domain_geometry)


# ### 4. Create the Bayesian model

# After constructing the forward model, we want to set up the Bayesian model, i.e. the posterior distribution. In `CUQIpy` we achieve this by creating a joint distribution on the Bayesian parameters $\theta$ and the data $y$ using the `JointDistribution` class, and then, condition it on a synthesized data that we create for this test case. The joint distribution is given by:
# 
# \begin{align}
# p(\theta,y) = p(y|\theta)p(\theta)
# \end{align}
# where $p(x)$ is the prior probability density function (PDF) and $p(y|\theta)$ is the data distribution PDF.
# 
# We start by defining the prior distribution $p(\theta)$ as a standard multivariate Gaussian distribution $\mathcal{N}(\mathbf{0}, \mathbf{I})$ using `cuqi.distribution.GaussianCov` class. We pass the keyword argument `geometry=domain_geometry` when initializing this distribution so that the distribution encapsulates the knowledge that the multivariate random variable $\theta$ represents the expansion coefficient in the KL expansion (<equation_number>).

# In[ ]:


# Create the prior distribution 
x = cuqi.distribution.GaussianCov(np.zeros(model.domain_dim), 1, geometry=domain_geometry)


# Now samples from the prior will look like:

# In[ ]:


### Not to be included in the paper ###
for i in range(5):
    x.sample().plot()


# For this test case, we assume that the true initial heat profile is given by the expression
# 
# \begin{align}
# g_\text{exact}(x) = e^{-2x} \sin(L-x)
# \end{align}
# 
# We create `cuqi.samples.CUQIarray` object representing this signal and we call it `theta_true`. We apply the forward model on `theta_exact` to obtain what we call exact data `y_exact`, which is the outcome of the forward model that is not corrupted by measurement noise. 

# In[ ]:


### Not to be included in the paper ###

# True parameters that we want to infer
x_exact_raw = grid*np.exp(-2*grid)*np.sin(L-grid)
x_exact = cuqi.samples.CUQIarray(x_exact_raw, is_par=False, geometry=domain_geometry)


# In[ ]:


# Generate the exact data
y_exact = model.forward(x_exact)


# 
# We then create the data distribution `y` as a `cuqi.distribution.GaussianCov` object with the mean being the forward model applied to the Bayesian parameter $\theta$ `model(x)` and covariance matrix given by $\sigma^2\mathbf{I}_M$ where $\sigma= \frac{0.01}{\sqrt{N}} ||\mathcal{G}(g_\text{exact}(x))||$ and $\mathbf{I}_M$ is the $M$ dimensional identity matrix. We also equip the data distribution with the range geometry `range_geometry`.

# In[ ]:


### Not to be included in the paper ###
sigma =1.0/np.sqrt(N)* 0.01*np.linalg.norm(y_exact)


# In[ ]:


# Create the data distribution 
y = cuqi.distribution.GaussianCov(model(x), sigma**2*np.eye(model.range_dim), geometry=range_geometry)


# An instance of a noisy data (`data`) can then be simply generated as a sample of the distribution `y` conditioned on `x=x_exact`. Figure <fig_num> shows the exact solution $g(x)$, and the exact and the noisy data.

# In[ ]:


# Generate noisy data
data = y(x = x_exact).sample()


# In[ ]:


### Not to be included in the paper ###

x_exact.plot()
y_exact.plot()
data.plot()
plt.legend(['exact solution', 'exact data', 'noisy data']);


# In[ ]:


### Not to be included in the paper ###

#plt.plot(data-y_exact)
print(np.linalg.norm(data-y_exact))
print(np.linalg.norm(y_exact))
print(np.linalg.norm(data-y_exact)/np.linalg.norm(y_exact))


# Now that we have the distributions `x` and `y`, we can create the joint distribution $p(x,y)$. Conditioning the joint distribution on `y=data` gives the posterior distribution. 

# In[ ]:


# Bayesian model
joint_distribution = cuqi.distribution.JointDistribution(x, y)
posterior = joint_distribution(y = data) 


# ### 5. Estimate the Bayesian solution

# In `CUQIpy` we use MCMC sampling methods to approximate the posterior and compute its moments. In this test case, we use a component wise Metropolis Hastings (CWMH) algorithm which is implemented in the class `cuqi.sampler.CWMH`. We create a sampler, which takes the `posterior` as an argument in the initialization, and then generates 4000 samples. The `CWMH` method `sample_adapt` adjusts the step size of the algorithm (the step size) to achieve a target acceptance rate of about $0.23$. 

# In[ ]:


MySampler = cuqi.sampler.CWMH(posterior)
posterior_samples = MySampler.sample_adapt(4000)


# This is a test

# CUQIpy provides postprocessing methods and visualization methods that we can use to study the posterior samples. In figure <ci_fig1> we show a credible interval computed on the coefficient space then transformed to the function space. Furthermore, in Figure <ci_fig2> we transform the samples to the function space first, then compute (test) the credible interval. The later case can be achieved by applying `plot_ci` on the `Samples` property `funvals`. This property returns a `Samples` object which contains the function values of the samples. 

# In[ ]:


posterior_samples.plot_ci(95, exact=x_exact)


# In[ ]:


posterior_samples.funvals.plot_ci(95, exact=x_exact)


# In[ ]:


posterior_samples.plot_ci(95, plot_par=True)


# In[ ]:


plt.plot(posterior_samples.compute_ess(), 'o-')


# In[ ]:


posterior_samples.plot_trace([0,1,2,3,4,5,6,7,8,9,10]);


# |
# 
# |
# 
# |
# 
# |
# 
# |
# 
# |
# 
# |
# 
# |
# 
# |
# 
# |
# 

# Notes:
# * Partial observation 
# * Change end point to L
# * Enforce positivity 
# * Change x to theta
# * Fix the noise level
# * Try the heavy-side after fixing the noise.
# * Combine pCN with CWMH (each good at different modes)
# * Reduce the number of nodes
# * MAP point 
# * hierarchical model for estimating regularity of the heat problem

# In[ ]:


# Sample using pCN sampler
pcn_sampler = cuqi.sampler.pCN(posterior)
pcn_samples = pcn_sampler.sample_adapt(40000)


# In[ ]:


# Visualize/Analyze pCN samples
pcn_samples.funvals.plot_ci(100, exact=x_exact)
plt.figure()
pcn_samples.plot_trace([0,1,2,3,4,5,6,7,8,9,10])
plt.figure()
pcn_samples.plot_ci(plot_par=True)
pcn_samples.compute_ess()


# In[ ]:


# Create samples object that combines 2 samplers results
combined_samples = cuqi.samples.Samples(np.hstack([posterior_samples.samples,pcn_samples.samples]), geometry=domain_geometry)


# In[ ]:


# Visualize/Analyze the combined samples
combined_samples.funvals.plot_ci(95, exact=x_exact)
plt.figure()
combined_samples.plot_ci(plot_par=True)


# In[ ]:


# Visualize/Analyze the combined samples, cont.
posterior_samples.plot_pair([0,1,2,3,4])
pcn_samples.plot_pair([0,1,2,3,4])
combined_samples.plot_pair([0,1,2,3,4])


# In[ ]:


# Compute the MAP point
maximizer = cuqi.solver.maximize(posterior.logpdf, np.zeros(N))
sol = maximizer.solve()
plt.plot(grid, domain_geometry.par2fun(sol[0]))
x_exact.plot()


# In[ ]:


# Plot the posterior ci and MAP point
posterior_samples.plot_ci(95, exact=x_exact)
plt.plot(grid, domain_geometry.par2fun(sol[0]))


# In[ ]:


# Test that the factor 1/sqrt(N) is required to normalize the error vector.
norms = []
for i in range(100):
    norms.append(np.linalg.norm(np.random.randn(i))/np.sqrt(i))
norms

