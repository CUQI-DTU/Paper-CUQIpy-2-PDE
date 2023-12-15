import numpy as np
import dolfin as dl
import cuqipy_fenics
import cuqi
import matplotlib.pyplot as plt
from progressbar import progressbar

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

data1 = np.load('./stat/stat_circular_inclusion_2_5per_noise_thinned.npz')
data2 = np.load('./stat/stat_circular_inclusion_2_10per_noise_thinned.npz')
data3 = np.load('./stat/stat_circular_inclusion_2_20per_noise_thinned.npz')

print('loading ...')
print('data 1 ...')
samples1 = data1['samples']
samples1 = samples1
print('data 2 ...')
samples2 = data2['samples']
samples2 = samples2
print('data 3 ...')
samples3 = data3['samples']
samples3 = samples3
#samples = samples[-50000:,:]

mean1 = np.mean( samples1, axis=0 )
mean2 = np.mean( samples2, axis=0 )
mean3 = np.mean( samples3, axis=0 )

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

cuqi_samples1 = cuqi.samples.Samples(samples1, geometry=matern_geo)
cuqi_samples2 = cuqi.samples.Samples(samples2, geometry=matern_geo)
cuqi_samples3 = cuqi.samples.Samples(samples3, geometry=matern_geo)

#print('thinning samples ...')
#samples_thin1 = cuqi_samples1.burnthin(990000)
#samples_thin2 = cuqi_samples2.burnthin(990000)
#samples_thin3 = cuqi_samples3.burnthin(990000)

cm_to_in = 1/2.54
#fig = plt.figure( figsize=(17.8*cm_to_in, 5*cm_to_in),layout='constrained')
#subfigs = fig.subfigures(1)
f, axes = plt.subplots(1,3, figsize=(17.8*cm_to_in, 5*cm_to_in), sharey=True)

#axes = subfigs.subplots(1,3,sharey=True)

labels = list(range(0,36,7))

plt.sca(axes[0])
cuqi_samples1.plot_ci(95, plot_par=True, marker='.')
axes[0].legend([r'Mean',r'95% CT'], loc=4)
axes[0].set_xticks(labels)
axes[0].set_xticklabels(labels)
axes[0].set_xlim([-1,36])
axes[0].set_ylim([-5,3])
axes[0].grid()
axes[0].set_xlabel(r'$i$')
axes[0].set_ylabel(r'$x_i$')
axes[0].yaxis.labelpad = -3
axes[0].xaxis.labelpad = 0
axes[0].set_title(r'(a) 5% noise')
plt.sca(axes[1])
cuqi_samples2.plot_ci(95, plot_par=True, marker='.')
axes[1].legend([r'Mean',r'95% CT'], loc=4)
axes[1].set_xticks(labels)
axes[1].set_xticklabels(labels)
axes[1].set_xlim([-1,36])
axes[1].set_ylim([-5,3])
axes[1].grid()
axes[1].set_xlabel(r'$i$')
axes[1].yaxis.labelpad = -3
axes[1].xaxis.labelpad = 0
axes[1].set_title(r'(b) 10% noise')
plt.sca(axes[2])
cuqi_samples3.plot_ci(95, plot_par=True, marker='.')
axes[2].legend([r'Mean',r'95% CT'], loc=4)
axes[2].set_xticks(labels)
axes[2].set_xticklabels(labels)
axes[2].set_xlim([-1,36])
axes[2].set_ylim([-5,3])
axes[2].grid()
axes[2].set_xlabel(r'$i$')
axes[2].yaxis.labelpad = -3
axes[2].xaxis.labelpad = 0
axes[2].set_title(r'(c) 20% noise')

plt.tight_layout()

plt.savefig('./plots/params.pdf',format='pdf')