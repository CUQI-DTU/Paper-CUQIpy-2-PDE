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
print('data 2 ...')
samples2 = data2['samples']
print('data 3 ...')
samples3 = data3['samples']
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

#%% 2.5 Create a prior
pr_mean = np.zeros(domain_geometry.par_dim)
prior = cuqi.distribution.Gaussian(pr_mean, cov=np.eye(domain_geometry.par_dim), geometry= domain_geometry, name='x')

prior_samples = prior.sample(5)

cuqi_samples1 = cuqi.samples.Samples(samples1, geometry=domain_geometry)
cuqi_samples2 = cuqi.samples.Samples(samples2, geometry=domain_geometry)
cuqi_samples3 = cuqi.samples.Samples(samples3, geometry=domain_geometry)

#print('thinning samples ...')
#samples_thin1 = cuqi_samples1.burnthin(900000)
#samples_thin2 = cuqi_samples2.burnthin(900000)
#samples_thin3 = cuqi_samples3.burnthin(900000)

cm_to_in = 1/2.54
fig = plt.figure( figsize=(17.8*cm_to_in, 20*cm_to_in),layout='constrained')
subfigs = fig.subfigures(4, 1)

axes = subfigs[0].subplots(1,5,sharey=True)
plt.sca(axes[0])
prior_samples.plot([ 0 ], subplots=False)
axes[0].set_xlim([-1.1,1.1])
axes[0].set_ylim([-1.1,1.1])
axes[0].set_xticks([-1,0,1])
axes[0].set_yticks([-1,0,1])
axes[0].yaxis.labelpad = -3
axes[0].xaxis.labelpad = 0
axes[0].set_xlabel(r'$\xi_1$')
axes[0].set_ylabel(r'$\xi_2$')
plt.sca(axes[1])
prior_samples.plot([ 1 ], subplots=False)
#axes[0,1].set_yticks([])
axes[1].set_ylabel('')
axes[1].set_xlim([-1.1,1.1])
axes[1].set_ylim([-1.1,1.1])
axes[1].set_xticks([-1,0,1])
axes[1].xaxis.labelpad = 0
axes[1].set_xlabel(r'$\xi_1$')
plt.sca(axes[2])
prior_samples.plot([ 2 ], subplots=False)
#axes[0,2].set_yticks([])
axes[2].set_ylabel('')
axes[2].set_xlim([-1.1,1.1])
axes[2].set_ylim([-1.1,1.1])
axes[2].set_xticks([-1,0,1])
axes[2].xaxis.labelpad = 0
axes[2].set_xlabel(r'$\xi_1$')
plt.sca(axes[3])
prior_samples.plot([ 3 ], subplots=False)
#axes[0,3].set_yticks([])
axes[3].set_ylabel('')
axes[3].set_xlim([-1.1,1.1])
axes[3].set_ylim([-1.1,1.1])
axes[3].set_xticks([-1,0,1])
axes[3].xaxis.labelpad = 0
axes[3].set_xlabel(r'$\xi_1$')
plt.sca(axes[4])
prior_samples.plot([ 4 ], subplots=False)
#axes[0,4].set_yticks([])
axes[4].set_ylabel('')
axes[4].set_xlim([-1.1,1.1])
axes[4].set_ylim([-1.1,1.1])
axes[4].set_xticks([-1,0,1])
axes[4].xaxis.labelpad = 0
axes[4].set_xlabel(r'$\xi_1$')

subfigs[0].suptitle('(a) prior samples', fontsize=12)

axes = subfigs[1].subplots(1,5,sharey=True)
idx = [0,50, 100, 150, -2]
plt.sca(axes[0])
cuqi_samples1.plot([ idx[0] ], subplots=False)
axes[0].set_xlim([-1.1,1.1])
axes[0].set_ylim([-1.1,1.1])
axes[0].set_xticks([-1,0,1])
axes[0].set_yticks([-1,0,1])
axes[0].yaxis.labelpad = -3
axes[0].xaxis.labelpad = 0
axes[0].set_xlabel(r'$\xi_1$')
axes[0].set_ylabel(r'$\xi_2$')
plt.sca(axes[1])
cuqi_samples1.plot([ idx[1] ], subplots=False)
#axes[1,1].set_yticks([])
axes[1].set_ylabel('')
axes[1].set_xlim([-1.1,1.1])
axes[1].set_ylim([-1.1,1.1])
axes[1].set_xticks([-1,0,1])
axes[1].xaxis.labelpad = 0
axes[1].set_xlabel(r'$\xi_1$')
plt.sca(axes[2])
cuqi_samples1.plot([ idx[2] ], subplots=False)
#axes[1,2].set_yticks([])
axes[2].set_ylabel('')
axes[2].set_xlim([-1.1,1.1])
axes[2].set_ylim([-1.1,1.1])
axes[2].set_xticks([-1,0,1])
axes[2].xaxis.labelpad = 0
axes[2].set_xlabel(r'$\xi_1$')
plt.sca(axes[3])
cuqi_samples1.plot([ idx[3] ], subplots=False)
#axes[1,3].set_yticks([])
axes[3].set_ylabel('')
axes[3].set_xlim([-1.1,1.1])
axes[3].set_ylim([-1.1,1.1])
axes[3].set_xticks([-1,0,1])
axes[3].xaxis.labelpad = 0
axes[3].set_xlabel(r'$\xi_1$')
plt.sca(axes[4])
cuqi_samples1.plot([ idx[4] ], subplots=False)
#axes[1,4].set_yticks([])
axes[4].set_ylabel('')
axes[4].set_xlim([-1.1,1.1])
axes[4].set_ylim([-1.1,1.1])
axes[4].set_xticks([-1,0,1])
axes[4].xaxis.labelpad = 0
axes[4].set_xlabel(r'$\xi_1$')

subfigs[1].suptitle('(b) posterior samples with 5% noise', fontsize=12)

axes = subfigs[2].subplots(1,5,sharey=True)
plt.sca(axes[0])
cuqi_samples2.plot([ idx[0] ], subplots=False)
axes[0].set_xlim([-1.1,1.1])
axes[0].set_ylim([-1.1,1.1])
axes[0].set_xticks([-1,0,1])
axes[0].set_yticks([-1,0,1])
axes[0].yaxis.labelpad = -3
axes[0].xaxis.labelpad = 0
axes[0].set_xlabel(r'$\xi_1$')
axes[0].set_ylabel(r'$\xi_2$')
plt.sca(axes[1])
cuqi_samples2.plot([ idx[1] ], subplots=False)
#axes[2,1].set_yticks([])
axes[1].set_ylabel('')
axes[1].set_xlim([-1.1,1.1])
axes[1].set_ylim([-1.1,1.1])
axes[1].set_xticks([-1,0,1])
axes[1].xaxis.labelpad = 0
axes[1].set_xlabel(r'$\xi_1$')
plt.sca(axes[2])
cuqi_samples2.plot([ idx[2] ], subplots=False)
#axes[2,2].set_yticks([])
axes[2].set_ylabel('')
axes[2].set_xlim([-1.1,1.1])
axes[2].set_ylim([-1.1,1.1])
axes[2].set_xticks([-1,0,1])
axes[2].xaxis.labelpad = 0
axes[2].set_xlabel(r'$\xi_1$')
plt.sca(axes[3])
cuqi_samples2.plot([ idx[3] ], subplots=False)
#axes[2,3].set_yticks([])
axes[3].set_ylabel('')
axes[3].set_xlim([-1.1,1.1])
axes[3].set_ylim([-1.1,1.1])
axes[3].set_xticks([-1,0,1])
axes[3].xaxis.labelpad = 0
axes[3].set_xlabel(r'$\xi_1$')
plt.sca(axes[4])
cuqi_samples2.plot([ idx[4] ], subplots=False)
#axes[2,4].set_yticks([])
axes[4].set_ylabel('')
axes[4].set_xlim([-1.1,1.1])
axes[4].set_ylim([-1.1,1.1])
axes[4].set_xticks([-1,0,1])
axes[4].xaxis.labelpad = 0
axes[4].set_xlabel(r'$\xi_1$')

subfigs[2].suptitle('(c) posterior samples with 10% noise', fontsize=12)

axes = subfigs[3].subplots(1,5,sharey=True)
plt.sca(axes[0])
cuqi_samples3.plot([ idx[0] ], subplots=False)
axes[0].set_xlim([-1.1,1.1])
axes[0].set_ylim([-1.1,1.1])
axes[0].set_xticks([-1,0,1])
axes[0].set_yticks([-1,0,1])
axes[0].yaxis.labelpad = -3
axes[0].xaxis.labelpad = 0
axes[0].set_xlabel(r'$\xi_1$')
axes[0].set_ylabel(r'$\xi_2$')
plt.sca(axes[1])
cuqi_samples3.plot([ idx[1] ], subplots=False)
#axes[3,1].set_yticks([])
axes[1].set_ylabel('')
axes[1].set_xlim([-1.1,1.1])
axes[1].set_ylim([-1.1,1.1])
axes[1].set_xticks([-1,0,1])
axes[1].xaxis.labelpad = 0
axes[1].set_xlabel(r'$\xi_1$')
plt.sca(axes[2])
cuqi_samples3.plot([ idx[2] ], subplots=False)
#axes[3,2].set_yticks([])
axes[2].set_ylabel('')
axes[2].set_xlim([-1.1,1.1])
axes[2].set_ylim([-1.1,1.1])
axes[2].set_xticks([-1,0,1])
axes[2].xaxis.labelpad = 0
axes[2].set_xlabel(r'$\xi_1$')
plt.sca(axes[3])
cuqi_samples3.plot([ idx[3] ], subplots=False)
#axes[3,3].set_yticks([])
axes[3].set_ylabel('')
axes[3].set_xlim([-1.1,1.1])
axes[3].set_ylim([-1.1,1.1])
axes[3].set_xticks([-1,0,1])
axes[3].xaxis.labelpad = 0
axes[3].set_xlabel(r'$\xi_1$')
plt.sca(axes[4])
cuqi_samples3.plot([ idx[4] ], subplots=False)
#axes[3,4].set_yticks([])
axes[4].set_ylabel('')
axes[4].set_xlim([-1.1,1.1])
axes[4].set_ylim([-1.1,1.1])
axes[4].set_xticks([-1,0,1])
axes[4].xaxis.labelpad = 0
axes[4].set_xlabel(r'$\xi_1$')

subfigs[3].suptitle('(d) posterior samples with 20% noise', fontsize=12)

#subfigs[0].subplots_adjust(wspace=0)
#subfigs[1].subplots_adjust(wspace=0)
#subfigs[2].subplots_adjust(wspace=0)
#subfigs[3].subplots_adjust(wspace=0)
#plt.tight_layout()

plt.savefig('./plots/samples.pdf',format='pdf')
