import numpy as np
from dolfin import *
import sys
sys.path.append('./CUQIpy-FEniCS') 
import cuqipy_fenics
import cuqi
import matplotlib.pyplot as plt

import scipy.sparse as sparse

from wave import wave

problem = wave()
forward = problem.forward_half

obs_data = np.load( './obs/half_boundary_5per.npz' )
data = obs_data['data']
sigma2 = obs_data['sigma2']

mesh = UnitIntervalMesh(120)
V = FunctionSpace(mesh,'CG', 1)

fenics_continuous_geo = cuqipy_fenics.geometry.FEniCSContinuous(V)
matern_geometry = cuqipy_fenics.geometry.MaternExpansion(fenics_continuous_geo, length_scale = .1, nu=0.75, num_terms=100)

def prior_map(func):
    dofs = func.vector().get_local()
    updated_dofs = 15*dofs
    func.vector().set_local(updated_dofs)
    return func

domain_geometry = cuqipy_fenics.geometry.FEniCSMappedGeometry(matern_geometry, map = prior_map)

range_geometry = cuqi.geometry.Discrete( data.shape[0] )
model = cuqi.model.Model(forward,range_geometry, domain_geometry)

p0 = cuqi.distribution.Gaussian(0, cov=np.eye(domain_geometry.par_dim), geometry= domain_geometry)
#samps = p0.sample(10)
#samps.plot()
#plt.savefig('samples.pdf',format='pdf',dpi=300)
#exit()

#prior.sample().plot()
#plt.savefig('prior.pdf',format='pdf')

#x0 = p0.sample()
#out = model(x0)
#print(out.shape)
#print(range_geometry.par_dim)

eye_sparse = sparse.diags(np.ones(251))

y = cuqi.distribution.Gaussian(model(p0), cov=sigma2*np.ones(data.shape[0]), geometry=range_geometry)

y = y(y = data)
posterior = cuqi.distribution.Posterior(y, p0)

#sampler = cuqi.sampler.MetropolisHastings(posterior)
#samples = sampler.sample_adapt(200000)
sampler = cuqi.sampler.pCN(posterior)
samples = sampler.sample_adapt(200000)


new_samples = samples.burnthin(500)
new_samples.geometry = domain_geometry

plt.figure()
new_samples.plot_ci(95, plot_par=True, marker='.') 
plt.savefig('./plots/coefficients.pdf',format='pdf')
#

new_samples.geometry = domain_geometry


plt.figure()
im = new_samples.plot_mean()
#plt.colorbar(im[0])
plt.savefig('./plots/mean_kl_space.pdf',format='pdf')

np.savez( './stat/half_boundary_5per2_pcn.npz', samples=samples.samples)