# Run this script to generate the figures 10, 11, and 12 of the paper
# CUQIpy – Part II: computational uncertainty quantification for PDE-based
# inverse problems in Python.

# Before running this script, please run the PAT.py script to generate the
# sample files (for cases full_data = True and full_data = False).

#%% Import libraries
import numpy as np
from dolfin import *
from cuqi.samples import Samples
from cuqi.array import CUQIarray
import matplotlib.pyplot as plt
from figures_util import plot_figure10, plot_figure11, plot_figure12
import os
from PAT import create_domain_geometry

# Set up matplotlib parameters
SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 9
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Load observation data
obs_data_full = np.load( './obs/full_boundary_5per.npz' )
y_obs_full = obs_data_full['data'].reshape(251,2)
b_exact = obs_data_full['b_exact'].reshape(251,2)
obs_data_half = np.load( './obs/half_boundary_5per.npz' )
y_obs_half = obs_data_half['data'].reshape(251,1)

# Load true parameter
init_pressure_data = np.load( './obs/init_pressure.npz' )
g_true = init_pressure_data['init_pressure']

# Load samples
print('loading ...')
print('data 1 ...')
data_full = np.load('./stat/samples_thinned_full.npz')
samples_full = data_full['samples']
print('data 2 ...')
data_half = np.load('./stat/samples_thinned_half.npz')
samples_half = data_half['samples']

# Create domain geometry
mesh = UnitIntervalMesh(120)
parameter_space = FunctionSpace(mesh,'CG', 1)
G = create_domain_geometry(parameter_space)

# Create CUQIpy samples
cuqi_samples_full = Samples(samples_full, geometry=G)
cuqi_samples_half = Samples(samples_half, geometry=G)

# Create true FEniCS function wrapped in CUQIarray
g_true_function = Function(parameter_space)
g_true_function.vector().set_local(g_true[::-1])
g_true_function = CUQIarray(g_true_function, is_par=False, geometry=G)

#%% Create plot directory if it does not exists
if not os.path.exists('./plots'):
    os.makedirs('./plots')

#%% Plot figure 10
plt.figure()
plot_figure10(g_true, b_exact, y_obs_full)
plt.savefig('./plots/data.pdf',format='pdf')

#%% Plot figure 11
plt.figure()
plot_figure11(g_true_function, cuqi_samples_full, cuqi_samples_half)
plt.savefig('./plots/uq.pdf',format='pdf')

#%% Plot figure 12
plot_figure12(cuqi_samples_full, cuqi_samples_half)
plt.savefig('./plots/params.pdf',format='pdf')