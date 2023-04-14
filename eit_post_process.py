import numpy as np
import dolfin as dl
import cuqipy_fenics
import cuqi
import matplotlib.pyplot as plt
from progressbar import progressbar

data = np.load('./stat/stat_circular_inclusion_2_1per_noise.npz')

samples = data['samples']
#samples = samples[-50000:,:]

mean = np.mean( samples, axis=0 )

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

cuqi_samples = cuqi.samples.Samples(samples, geometry=matern_geo)
new_samples = cuqi_samples.burnthin(10000)

plt.figure()
new_samples.plot_ci(95, plot_par=True, marker='.') 
plt.savefig('./plots/coefficients.pdf',format='pdf')


plt.figure()
new_samples.plot_mean()
plt.savefig('./plots/mean_kl_space.pdf',format='pdf')

plt.figure()
func_vals_np = []
for s in progressbar(new_samples):
    func_vals_np.append( matern_geo.par2fun(s).vector().get_local() )
func_vals_np = np.array(func_vals_np)
var_np = np.var(func_vals_np,axis=0)

var_func = dl.Function(parameter_space)
var_func.vector().set_local( var_np )

plt.figure()
dl.plot(var_func)
plt.savefig('./plots/variance_kl_space.pdf',format='pdf')

plt.figure()
new_samples.plot([0, 500000, 750000, -1])
plt.savefig('./plots/samples_kl_space.pdf',format='pdf')

cuqi_samples = cuqi.samples.Samples(samples, geometry=domain_geometry)
new_samples = cuqi_samples.burnthin(10000)

plt.figure()
new_samples.plot_mean()
plt.savefig('./plots/mean_mapped.pdf',format='pdf')

plt.figure()
new_samples.plot([0, 500000, 750000, -1])
plt.savefig('./plots/samples_mapped.pdf',format='pdf')

func_vals_np = []
for s in progressbar(new_samples):
    func_vals_np.append( domain_geometry.par2fun(s).vector().get_local() )
func_vals_np = np.array(func_vals_np)
var_np = np.var(func_vals_np,axis=0)

var_func = dl.Function(parameter_space)
var_func.vector().set_local( var_np )

plt.figure()
dl.plot(var_func)
plt.savefig('./plots/variance_mapped.pdf',format='pdf')