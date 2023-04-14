import numpy as np
from dolfin import *
import sys
sys.path.append('./CUQIpy-FEniCS') 
import cuqipy_fenics
import cuqi
import matplotlib.pyplot as plt

import scipy.sparse as sparse

from wave import wave

data = np.load('./stat/full_boundary_5per2.npz')
samples = data['samples']

problem = wave()
forward = problem.forward_full

obs_data = np.load( './obs/full_boundary_5per.npz' )
data = obs_data['data']
sigma2 = obs_data['sigma2']

mesh = UnitIntervalMesh(120)
V = FunctionSpace(mesh,'CG', 1)

fenics_continuous_geo = cuqipy_fenics.geometry.FEniCSContinuous(V)

def prior_map(func):
    dofs = func.vector().get_local()
    updated_dofs = 15*dofs
    func.vector().set_local(updated_dofs)
    return func


matern_geometry = cuqipy_fenics.geometry.MaternExpansion(fenics_continuous_geo, length_scale = .1, nu=.75, num_terms=100)
domain_geometry = cuqipy_fenics.geometry.FEniCSMappedGeometry(matern_geometry, map = prior_map)

range_geometry = cuqi.geometry.Discrete( data.shape[0] )
model = cuqi.model.Model(forward,range_geometry, domain_geometry)

p0 = cuqi.distribution.Gaussian(0, cov=np.eye(domain_geometry.par_dim), geometry= domain_geometry)

cuqi_samples = cuqi.samples.Samples(samples, geometry=domain_geometry)
new_samples = cuqi_samples.burnthin(1000)

plt.figure()
new_samples.plot_ci(95, plot_par=True, marker='.') 
plt.savefig('./plots/coefficients.pdf',format='pdf')

plt.figure()
im = new_samples.plot_mean()
#plt.colorbar(im[0])
plt.savefig('./plots/mean_kl_space.pdf',format='pdf')

