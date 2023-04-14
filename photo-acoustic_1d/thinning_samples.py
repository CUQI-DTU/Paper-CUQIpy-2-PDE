import numpy as np
from dolfin import *
import sys
sys.path.append('./CUQIpy-FEniCS') 
import cuqipy_fenics
import cuqi
import matplotlib.pyplot as plt

import scipy.sparse as sparse

from wave import wave
import arviz

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
samples = data_full['samples']

samples = samples[:,50000::100]
np.savez( './stat/full_boundary_5per2_pcn_thinned.npz',samples=samples )

data_half = np.load('./stat/half_boundary_5per2_pcn.npz')
samples = data_half['samples']

samples = samples[:,50000::100]
np.savez( './stat/half_boundary_5per2_pcn_thinned.npz',samples=samples )

