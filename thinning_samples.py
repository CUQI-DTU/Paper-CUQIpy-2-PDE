import numpy as np
import dolfin as dl
import cuqipy_fenics
import cuqi
import matplotlib.pyplot as plt
from progressbar import progressbar
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

data1 = np.load('./stat/stat_circular_inclusion_2_20per_noise.npz')

print('loading ...')
print('data 1 ...')
samples1 = data1['samples']

mean1 = np.mean( samples1, axis=0 )

#%% 1.1 loading mesh
mesh = dl.Mesh("mesh.xml")

#%% 1.2 Define function spaces 
parameter_space = dl.FunctionSpace(mesh, "CG", 1)

u0 = dl.Constant('0.0')
boundary = lambda x, on_boundary: on_boundary
zero_bc = dl.DirichletBC(parameter_space, u0, boundary)

dummy = dl.Function(parameter_space)
dummy.vector().set_local( np.ones_like( dummy.vector().get_local() ) )
zero_bc.apply( dummy.vector() )
bnd_idx = np.argwhere( dummy.vector().get_local() == 0 ).flatten()

#%% defining geometry for the paramters
fenics_continuous_geo = cuqipy_fenics.geometry.FEniCSContinuous(parameter_space)

#%% defining the matern geometry
matern_geo = cuqipy_fenics.geometry.MaternExpansion(fenics_continuous_geo, length_scale = .2, num_terms=64)

#%% defining the nonlinear map to piece-wise constant field
c_minus = 1
c_plus = 10
ones_vec = np.ones(94)
def heavy_map(func):
    dofs = func.vector().get_local()
    updated_dofs = c_minus*0.5*(1 + np.sign(dofs)) + c_plus*0.5*(1 - np.sign(dofs))

    updated_dofs[bnd_idx] = np.ones(94)
    func.vector().set_local(updated_dofs)
    return func

# map `heavy_map` on Matern realizations.
domain_geometry = cuqipy_fenics.geometry.FEniCSMappedGeometry(matern_geo, map = heavy_map)

samples1 = samples1[:,200000::4000]

np.savez('./stat/stat_circular_inclusion_2_20per_noise_thinned.npz',samples=samples1)
