import numpy as np
from dolfin import *
import sys
sys.path.append('./CUQIpy-FEniCS') 
import cuqipy_fenics
import cuqi
import matplotlib.pyplot as plt

import scipy.sparse as sparse

from wave import wave

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

data_full = np.load('./stat/full_boundary_5per2_pcn.npz')
data_half = np.load('./stat/half_boundary_5per2_pcn.npz')
obs_data_full = np.load( './obs/full_boundary_5per.npz' )
obs_data_full = obs_data_full['data']
obs_data_half = np.load( './obs/half_boundary_5per.npz' )
obs_data_half = obs_data_half['data']

print('loading ...')
print('data 1 ...')
samples_full = data_full['samples']
samples_full = samples_full
print('data 2 ...')
samples_half = data_half['samples']
samples_half = samples_half

mesh = UnitIntervalMesh(120)
V = FunctionSpace(mesh,'CG', 1)

problem = wave()
forward_full = problem.forward_full
forward_half = problem.forward_half

fenics_continuous_geo = cuqipy_fenics.geometry.FEniCSContinuous(V)

def prior_map(func):
    dofs = func.vector().get_local()
    updated_dofs = 15*dofs
    func.vector().set_local(updated_dofs)
    return func


matern_geometry = cuqipy_fenics.geometry.MaternExpansion(fenics_continuous_geo, length_scale = .1, nu=.75, num_terms=100)
domain_geometry = cuqipy_fenics.geometry.FEniCSMappedGeometry(matern_geometry, map = prior_map)

range_geometry_full = cuqi.geometry.Discrete( obs_data_full.shape[0] )
model_full = cuqi.model.Model(forward_full,range_geometry_full, domain_geometry)
range_geometry_half = cuqi.geometry.Discrete( obs_data_half.shape[0] )
model_half = cuqi.model.Model(forward_half,range_geometry_half, domain_geometry)

p0 = cuqi.distribution.Gaussian(0, cov=np.eye(domain_geometry.par_dim), geometry= domain_geometry)

cuqi_samples_full = cuqi.samples.Samples(samples_full, geometry=domain_geometry)
cuqi_samples_half = cuqi.samples.Samples(samples_half, geometry=domain_geometry)

cm_to_in = 1/2.54
#fig = plt.figure( figsize=(17.8*cm_to_in, 5*cm_to_in),layout='constrained')
#subfigs = fig.subfigures(1)
f, axes = plt.subplots(1,2, figsize=(17.8*cm_to_in, 5*cm_to_in), sharey=True)

labels = list(range(0,36,7))

plt.sca(axes[0])
cuqi_samples_full.plot_ci(95, plot_par=True, marker='.')
axes[0].legend([r'Mean',r'95% CT'], loc=4)
axes[0].set_xticks(labels)
axes[0].set_xticklabels(labels)
axes[0].set_xlim([-1,25])
axes[0].set_ylim([-2.5,2.5])
axes[0].grid()
axes[0].set_xlabel(r'$i$')
axes[0].set_ylabel(r'$x_i$')
axes[0].yaxis.labelpad = -3
axes[0].xaxis.labelpad = 0
axes[0].set_title(r'(a) data from both boundaries')
plt.sca(axes[1])
cuqi_samples_half.plot_ci(95, plot_par=True, marker='.')
axes[1].legend([r'Mean',r'95% CT'], loc=4)
axes[1].set_xticks(labels)
axes[1].set_xticklabels(labels)
axes[1].set_xlim([-1,25])
axes[1].set_ylim([-2.5,2.5])
axes[1].grid()
axes[1].set_xlabel(r'$i$')
axes[1].yaxis.labelpad = -3
axes[1].xaxis.labelpad = 0
axes[1].set_title(r'(b) data from one boundary')

plt.tight_layout()

plt.savefig('./plots/params.pdf',format='pdf')

