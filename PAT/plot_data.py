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

obs_data_full = np.load( './obs/full_boundary_5per.npz' )
data = obs_data_full['data'].reshape(251,2)
b_exact = obs_data_full['b_exact'].reshape(251,2)
init_pressure_data = np.load( './obs/init_pressure.npz' )
exact_solution = init_pressure_data['init_pressure']

cm_to_in = 1/2.54
#fig = plt.figure( figsize=(17.8*cm_to_in, 5*cm_to_in),layout='constrained')
#subfigs = fig.subfigures(1)
f, axes = plt.subplots(1,3, figsize=(17.8*cm_to_in, 6*cm_to_in), sharey=True)

t = np.linspace(0,1,251)
labels = np.linspace(0,1,5)

plt.sca(axes[0])
plt.plot(t,data[:,0])
plt.plot(t,b_exact[:,0])
axes[0].legend([r'noisy data',r'exact data'], loc=1)
axes[0].set_xticks(labels)
axes[0].set_xticklabels(labels)
axes[0].set_xlim([-.05,1.05])
axes[0].set_ylim([-0.05,.55])
axes[0].set_xlabel(r'$\tau$')
axes[0].set_ylabel(r'$u(\xi_L)$')
axes[0].yaxis.labelpad = -3
axes[0].xaxis.labelpad = 0
axes[0].set_title(r'(a) data from left boundaries')
axes[0].grid()

plt.sca(axes[2])
plt.plot(t,data[:,1])
plt.plot(t,b_exact[:,1])
axes[2].legend([r'noisy data',r'exact data'], loc=1)
axes[2].set_xticks(labels)
axes[2].set_xticklabels(labels)
axes[2].set_xlim([-.05,1.05])
axes[2].set_ylim([-0.05,.55])
axes[2].set_xlabel(r'$\tau$')
axes[2].set_ylabel(r'$u(\xi_R)$')
axes[2].yaxis.labelpad = -3
axes[2].xaxis.labelpad = 0
axes[2].set_title(r'(c) data from right boundaries')
axes[2].grid()

t = np.linspace(0,1,121)
labels = np.linspace(0,1,5)

plt.sca(axes[1])
plt.plot(t,exact_solution)
axes[1].set_xticks(labels)
axes[1].set_xticklabels(labels)
axes[1].set_xlim([-.05,1.05])
axes[1].set_ylim([-0.05,.55])
axes[1].set_xlabel(r'$\xi$')
axes[1].set_ylabel(r'$u(\xi)$')
axes[1].yaxis.labelpad = -3
axes[1].xaxis.labelpad = 0
axes[1].set_title(r'(b) initial pressure profile')
axes[1].grid()

plt.tight_layout()

plt.savefig('./plots/data.pdf',format='pdf')