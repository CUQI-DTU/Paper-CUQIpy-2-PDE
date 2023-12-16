# Run this script to generate the figures 6, 7, 8, and 9 of the paper
# CUQIpy – Part II: computational uncertainty quantification for PDE-based
# inverse problems in Python.

# Before running this script, please run the EIT.py script to generate the
# sample files (for noise percent cases 5, 10, 20).

#%% Import libraries
import numpy as np
import dolfin as dl
from cuqi.samples import Samples
import matplotlib.pyplot as plt
from EIT import extract_boundary_dofs_indices, create_domain_geometry
from figures_util import plot_figure6, plot_figure7, plot_figure8, plot_figure9
import os

#%% Set up matplotlib parameters
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

#%% Load the sample files
print('loading ...')
print('data 1 ...')
data1 = np.load('./stat/stat_circular_inclusion_2_5per_noise_thinned.npz')
samples1 = data1['samples']
print('data 2 ...')
data2 = np.load('./stat/stat_circular_inclusion_2_10per_noise_thinned.npz')
samples2 = data2['samples']
print('data 3 ...')
data3 = np.load('./stat/stat_circular_inclusion_2_20per_noise_thinned.npz')
samples3 = data3['samples']

#%% Load the exact conductivity and the data
obs_data3 = np.load('./obs/obs_circular_inclusion_2_20per_noise.npz')
data = obs_data3['data']
exact_data = obs_data3['b_exact']

#%% Create the domain geometry domain_geometry
# Load mesh
mesh = dl.Mesh("mesh.xml")

# Define function spaces 
parameter_space = dl.FunctionSpace(mesh, "CG", 1)
solution_space = dl.FunctionSpace(mesh, "CG", 1)

# extracting indices for elements at the boundary of the computational mesh
bnd_idx = extract_boundary_dofs_indices(solution_space)

# Create the domain geometry
G_Heavi = create_domain_geometry(parameter_space, bnd_idx)

# Create Samples objects
cuqi_samples1 = Samples(samples1, geometry=G_Heavi)
cuqi_samples2 = Samples(samples2, geometry=G_Heavi)
cuqi_samples3 = Samples(samples3, geometry=G_Heavi)

#%% Create plot directory if it does not exists
if not os.path.exists('./plots'):
    os.makedirs('./plots')

#%% Plot figure 6
#plt.figure()
#plot_figure6(parameter_space, exact_data, data)
#plt.savefig('./plots/data.pdf',format='pdf')

#%% Plot figure 7
plt.figure()
plot_figure7(cuqi_samples1, cuqi_samples1, cuqi_samples2, cuqi_samples3)
plt.savefig('./plots/samples.pdf',format='pdf')

#%% Plot figure 8
plot_figure8(cuqi_samples1, cuqi_samples2, cuqi_samples3)
plt.savefig('./plots/uq.pdf',format='pdf')

#%% Plot figure 9
plot_figure9(cuqi_samples1, cuqi_samples2, cuqi_samples3)
plt.savefig('./plots/params.pdf',format='pdf')