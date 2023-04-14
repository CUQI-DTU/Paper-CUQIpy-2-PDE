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

cuqi_samples1 = cuqi.samples.Samples(samples1, geometry=matern_geo)
cuqi_samples2 = cuqi.samples.Samples(samples2, geometry=matern_geo)
cuqi_samples3 = cuqi.samples.Samples(samples3, geometry=matern_geo)

#print('thinning samples ...')
#samples_thin1 = cuqi_samples1.burnthin(990000)
#samples_thin2 = cuqi_samples2.burnthin(990000)
#samples_thin3 = cuqi_samples3.burnthin(990000)

cm_to_in = 1/2.54
fig = plt.figure( figsize=(17.8*cm_to_in, 20*cm_to_in))#,layout='constrained')
subfigs = fig.subfigures(4, 1)

axes = subfigs[2].subplots(1,3,sharey=True)
plt.sca(axes[0])
im = cuqi_samples1.plot_mean(subplots=False)
axes[0].set_title('5% noise')
axes[0].set_xlim([-1.1,1.1])
axes[0].set_ylim([-1.1,1.1])
axes[0].set_xticks([-1,1])
axes[0].set_yticks([-1,0,1])
axes[0].yaxis.labelpad = -3
axes[0].xaxis.labelpad = -7
axes[0].set_xlabel(r'$\xi_1$')
axes[0].set_ylabel(r'$\xi_2$')
im[0].set_clim([-0.1,0.25])
plt.sca(axes[1])
im = cuqi_samples1.plot_mean(subplots=False)
axes[1].set_title('10% noise')
axes[1].set_ylabel('')
axes[1].set_xlim([-1.1,1.1])
axes[1].set_ylim([-1.1,1.1])
axes[1].set_xticks([-1,1])
axes[1].xaxis.labelpad = -7
axes[1].set_xlabel(r'$\xi_1$')
im[0].set_clim([-0.1,0.25])
plt.sca(axes[2])
im = cuqi_samples1.plot_mean(subplots=False)
axes[2].set_title('20% noise')
axes[2].set_ylabel('')
axes[2].set_xlim([-1.1,1.1])
axes[2].set_ylim([-1.1,1.1])
axes[2].set_xticks([-1,1])
axes[2].xaxis.labelpad = -7
axes[2].set_xlabel(r'$\xi_1$')
im[0].set_clim([-0.1,0.25])

subfigs[2].colorbar(im[0], fraction=0.047)

#subfigs[0].colorbar(im[0], ax=axes[2], shrink=0.6)
subfigs[2].subplots_adjust(wspace=0,right=.9,top = 0.8)
subfigs[2].suptitle(r'(c) posterior mean visualized in $\mathbf{G}_{KL}$ geometry', fontsize=12)

func_vals_np1 = []
for s in progressbar(cuqi_samples1):
    func_vals_np1.append( matern_geo.par2fun(s).vector().get_local() )
func_vals_np1 = np.array(func_vals_np1)
var_np1 = np.var(func_vals_np1,axis=0)
var_func1 = dl.Function(parameter_space)
var_func1.vector().set_local( var_np1 )

func_vals_np2 = []
for s in progressbar(cuqi_samples2):
    func_vals_np2.append( matern_geo.par2fun(s).vector().get_local() )
func_vals_np2 = np.array(func_vals_np2)
var_np2 = np.var(func_vals_np2,axis=0)
var_func2 = dl.Function(parameter_space)
var_func2.vector().set_local( var_np2 )

func_vals_np3 = []
for s in progressbar(cuqi_samples3):
    func_vals_np3.append( matern_geo.par2fun(s).vector().get_local() )
func_vals_np3 = np.array(func_vals_np3)
var_np3 = np.var(func_vals_np3,axis=0)
var_func3 = dl.Function(parameter_space)
var_func3.vector().set_local( var_np3 )

axes = subfigs[3].subplots(1,3,sharey=True)

plt.sca(axes[0])
dl.plot(var_func1)
axes[0].set_title('5% noise')
axes[0].set_xlim([-1.1,1.1])
axes[0].set_ylim([-1.1,1.1])
axes[0].set_xticks([-1,1])
axes[0].set_yticks([-1,0,1])
axes[0].yaxis.labelpad = -3
axes[0].xaxis.labelpad = -7
axes[0].set_xlabel(r'$\xi_1$')
axes[0].set_ylabel(r'$\xi_2$')
plt.sca(axes[1])
dl.plot(var_func2)
axes[1].set_title('10% noise')
axes[1].set_ylabel('')
axes[1].set_xlim([-1.1,1.1])
axes[1].set_ylim([-1.1,1.1])
axes[1].set_xticks([-1,1])
axes[1].xaxis.labelpad = -7
axes[1].set_xlabel(r'$\xi_1$')
plt.sca(axes[2])
c = dl.plot(var_func3)
axes[2].set_title('20% noise')
axes[2].set_ylabel('')
axes[2].set_xlim([-1.1,1.1])
axes[2].set_ylim([-1.1,1.1])
axes[2].set_xticks([-1,1])
axes[2].xaxis.labelpad = -7
axes[2].set_xlabel(r'$\xi_1$')

subfigs[3].colorbar(c, fraction=0.047)
#subfigs[0].subplots_adjust(wspace=0)
subfigs[3].suptitle(r'(d) point-wise variance evaluated in $\mathbf{G}_{KL}$ geometry', fontsize=12)


cuqi_samples1.geometry = domain_geometry
cuqi_samples2.geometry = domain_geometry
cuqi_samples3.geometry = domain_geometry

axes = subfigs[0].subplots(1,3,sharey=True)
plt.sca(axes[0])
im = cuqi_samples1.plot_mean(subplots=False)
axes[0].set_title('5% noise')
axes[0].set_xlim([-1.1,1.1])
axes[0].set_ylim([-1.1,1.1])
axes[0].set_xticks([-1,1])
axes[0].set_yticks([-1,0,1])
axes[0].set_ylabel(r'$y$')
axes[0].set_xlabel(r'$x$')
axes[0].yaxis.labelpad = -3
axes[0].xaxis.labelpad = -7
axes[0].set_xlabel(r'$\xi_1$')
axes[0].set_ylabel(r'$\xi_2$')
#im[0].set_clim([1,10])
plt.sca(axes[1])
im = cuqi_samples2.plot_mean(subplots=False)
axes[1].set_title('10% noise')
axes[1].set_ylabel('')
axes[1].set_xlim([-1.1,1.1])
axes[1].set_ylim([-1.1,1.1])
axes[1].set_xticks([-1,1])
axes[1].set_xlabel(r'$x$')
axes[1].xaxis.labelpad = -7
axes[1].set_xlabel(r'$\xi_1$')
#im[0].set_clim([1,10])
plt.sca(axes[2])
im = cuqi_samples3.plot_mean(subplots=False)
axes[2].set_title('20% noise')
axes[2].set_ylabel('')
axes[2].set_xlim([-1.1,1.1])
axes[2].set_ylim([-1.1,1.1])
axes[2].set_xticks([-1,1])
axes[2].set_xlabel(r'$x$')
axes[2].xaxis.labelpad = -7
axes[2].set_xlabel(r'$\xi_1$')
#im[0].set_clim([1,10])

subfigs[0].colorbar(im[0], fraction=0.047)
#subfigs[0].subplots_adjust(wspace=0)
subfigs[0].suptitle(r'(a) posterior mean visualized in $\mathbf{G}_{Heavi}$ geometry', fontsize=12)


func_vals_np1 = []
for s in progressbar(cuqi_samples1):
    func_vals_np1.append( domain_geometry.par2fun(s).vector().get_local() )
func_vals_np1 = np.array(func_vals_np1)
var_np1 = np.var(func_vals_np1,axis=0)
var_func1 = dl.Function(parameter_space)
var_func1.vector().set_local( var_np1 )

func_vals_np2 = []
for s in progressbar(cuqi_samples2):
    func_vals_np2.append( domain_geometry.par2fun(s).vector().get_local() )
func_vals_np2 = np.array(func_vals_np2)
var_np2 = np.var(func_vals_np2,axis=0)
var_func2 = dl.Function(parameter_space)
var_func2.vector().set_local( var_np2 )

func_vals_np3 = []
for s in progressbar(cuqi_samples3):
    func_vals_np3.append( domain_geometry.par2fun(s).vector().get_local() )
func_vals_np3 = np.array(func_vals_np3)
var_np3 = np.var(func_vals_np3,axis=0)
var_func3 = dl.Function(parameter_space)
var_func3.vector().set_local( var_np3 )

axes = subfigs[1].subplots(1,3,sharey=True)

plt.sca(axes[0])
dl.plot(var_func1)
axes[0].set_title('5% noise')
axes[0].set_xlim([-1.1,1.1])
axes[0].set_ylim([-1.1,1.1])
axes[0].set_xticks([-1,1])
axes[0].set_yticks([-1,0,1])
axes[0].set_ylabel(r'$y$')
axes[0].set_xlabel(r'$x$')
axes[0].yaxis.labelpad = -3
axes[0].xaxis.labelpad = -7
axes[0].set_xlabel(r'$\xi_1$')
axes[0].set_ylabel(r'$\xi_2$')
plt.sca(axes[1])
dl.plot(var_func2)
axes[1].set_title('10% noise')
axes[1].set_ylabel('')
axes[1].set_xlim([-1.1,1.1])
axes[1].set_ylim([-1.1,1.1])
axes[1].set_xticks([-1,1])
axes[1].set_xlabel(r'$x$')
axes[1].xaxis.labelpad = -7
axes[1].set_xlabel(r'$\xi_1$')
plt.sca(axes[2])
c = dl.plot(var_func3)
axes[2].set_title('20% noise')
axes[2].set_ylabel('')
axes[2].set_xlim([-1.1,1.1])
axes[2].set_ylim([-1.1,1.1])
axes[2].set_xticks([-1,1])
axes[2].set_xlabel(r'$x$')
axes[2].xaxis.labelpad = -7
axes[2].set_xlabel(r'$\xi_1$')

subfigs[1].colorbar(c, fraction=0.047)
#subfigs[0].subplots_adjust(wspace=0)
subfigs[1].suptitle(r'(b) point-wise variance evaluated in $\mathbf{G}_{Heavi}$ geometry', fontsize=12)



plt.savefig('./plots/uq.pdf',format='pdf')
exit()



fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(17.8*cm_to_in, 24*cm_to_in), layout="constrained")

plt.sca(axes[0,0])
im = cuqi_samples1.plot_mean(subplots=False)
im[0].set_clim([-0.1,0.25])
plt.sca(axes[0,1])
im = cuqi_samples2.plot_mean(subplots=False)
im[0].set_clim([-0.1,0.25])
plt.sca(axes[0,2])
im = cuqi_samples3.plot_mean(subplots=False)
im[0].set_clim([-0.1,0.25])

func_vals_np1 = []
for s in progressbar(cuqi_samples1):
    func_vals_np1.append( matern_geo.par2fun(s).vector().get_local() )
func_vals_np1 = np.array(func_vals_np1)
var_np1 = np.var(func_vals_np1,axis=0)
var_func1 = dl.Function(parameter_space)
var_func1.vector().set_local( var_np1 )

func_vals_np2 = []
for s in progressbar(cuqi_samples2):
    func_vals_np2.append( matern_geo.par2fun(s).vector().get_local() )
func_vals_np2 = np.array(func_vals_np2)
var_np2 = np.var(func_vals_np2,axis=0)
var_func2 = dl.Function(parameter_space)
var_func2.vector().set_local( var_np2 )

func_vals_np3 = []
for s in progressbar(cuqi_samples3):
    func_vals_np3.append( matern_geo.par2fun(s).vector().get_local() )
func_vals_np3 = np.array(func_vals_np3)
var_np3 = np.var(func_vals_np3,axis=0)
var_func3 = dl.Function(parameter_space)
var_func3.vector().set_local( var_np3 )

plt.sca(axes[1,0])
dl.plot(var_func1)
plt.sca(axes[1,1])
dl.plot(var_func2)
plt.sca(axes[1,2])
dl.plot(var_func3)

cuqi_samples1.geometry = domain_geometry
cuqi_samples2.geometry = domain_geometry
cuqi_samples3.geometry = domain_geometry

plt.sca(axes[2,0])
im = cuqi_samples1.plot_mean(subplots=False)
plt.sca(axes[2,1])
im = cuqi_samples2.plot_mean(subplots=False)
plt.sca(axes[2,2])
im = cuqi_samples3.plot_mean(subplots=False)

func_vals_np1 = []
for s in progressbar(cuqi_samples1):
    func_vals_np1.append( domain_geometry.par2fun(s).vector().get_local() )
func_vals_np1 = np.array(func_vals_np1)
var_np1 = np.var(func_vals_np1,axis=0)
var_func1 = dl.Function(parameter_space)
var_func1.vector().set_local( var_np1 )

func_vals_np2 = []
for s in progressbar(cuqi_samples2):
    func_vals_np2.append( domain_geometry.par2fun(s).vector().get_local() )
func_vals_np2 = np.array(func_vals_np2)
var_np2 = np.var(func_vals_np2,axis=0)
var_func2 = dl.Function(parameter_space)
var_func2.vector().set_local( var_np2 )

func_vals_np3 = []
for s in progressbar(cuqi_samples3):
    func_vals_np3.append( domain_geometry.par2fun(s).vector().get_local() )
func_vals_np3 = np.array(func_vals_np3)
var_np3 = np.var(func_vals_np3,axis=0)
var_func3 = dl.Function(parameter_space)
var_func3.vector().set_local( var_np3 )

plt.sca(axes[3,0])
dl.plot(var_func1)
plt.sca(axes[3,1])
dl.plot(var_func2)
plt.sca(axes[3,2])
dl.plot(var_func3)


plt.savefig('field.pdf',format='pdf')

