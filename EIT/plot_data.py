import numpy as np
import dolfin as dl
import cuqipy_fenics
import cuqi
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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

boundary_coord_data = np.load('boundary_coordinates.npz')
theta = boundary_coord_data['theta']
idx = boundary_coord_data['idx']

obs_data1 = np.load('./obs/obs_circular_inclusion_2_5per_noise.npz')
exact = obs_data1['b_exact']
data1 = obs_data1['data']
obs_data2 = np.load('./obs/obs_circular_inclusion_2_10per_noise.npz')
data2 = obs_data2['data']
obs_data3 = np.load('./obs/obs_circular_inclusion_2_20per_noise.npz')
data3 = obs_data3['data']

class custom_field(dl.UserExpression):
    def set_params(self,cx=np.array([0.5,-0.5,-0.3]),cy=np.array([0.5,0.6,-0.3]), r = np.array([0.2,0.1,0.3]) ):
        self.cx = cx
        self.cy = cy
        self.r2 = r**2

    def eval(self,values,x):
        if( (x[0]-self.cx[0])**2 + (x[1]-self.cy[0])**2 < self.r2[0] ):
            values[0] = 10.
        elif( (x[0]-self.cx[1])**2 + (x[1]-self.cy[1])**2 < self.r2[1] ):
            values[0] = 10.
        elif( (x[0]-self.cx[2])**2 + (x[1]-self.cy[2])**2 < self.r2[2] ):
            values[0] = 10.
        else:
            values[0] = 1.
FEM_el = parameter_space.ufl_element()
kappa_custom = custom_field(element=FEM_el)
kappa_custom.set_params()

cm_to_in = 1/2.54
fig = plt.figure( figsize=(17.8*cm_to_in, 7*cm_to_in))#,layout='constrained')
#fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(17.8*cm_to_in, 7*cm_to_in), layout="constrained")
subfigs = fig.subfigures(1, 2)

axes = subfigs[0].subplots(1,2,sharey=True)

#circ1 = patches.Circle([0,0], radius=1,color='indigo', fill=True)
circ1 = patches.Circle([0,0], radius=1,color='#450558', fill=True)
circ2 = patches.Circle([0.5,0.5], radius=0.2,color='white', fill=True)
circ3 = patches.Circle([-0.5,0.6], radius=0.1,color='white', fill=True)
circ4 = patches.Circle([-0.3,-0.3], radius=0.3,color='white', fill=True)
#circ1 = patches.Circle([0,0], radius=1,color='k', fill=False)

axes[0].add_patch(circ1)
axes[0].add_patch(circ2)
axes[0].add_patch(circ3)
axes[0].add_patch(circ4)
axes[0].set_xlim([-1.1,1.1])
axes[0].set_ylim([-1.1,1.1])
axes[0].set_xticks([-1,0,1])
axes[0].set_yticks([-1,0,1])
axes[0].set_aspect('equal')
axes[0].set_xlabel(r'$\xi_1$')
axes[0].set_ylabel(r'$\xi_2$')
axes[0].yaxis.labelpad = -3
axes[0].xaxis.labelpad = 0
axes[0].set_title(r'true conductivity')

plt.sca(axes[1])
func = dl.Function(parameter_space)
func = dl.interpolate(kappa_custom,parameter_space)
c = dl.plot(func)
axes[1].set_xlim([-1.1,1.1])
axes[1].set_ylim([-1.1,1.1])
axes[1].set_xticks([-1,0,1])
axes[1].set_yticks([-1,0,1])
axes[1].set_xlabel(r'$\xi_1$')
axes[1].yaxis.labelpad = -3
axes[1].xaxis.labelpad = 0
axes[1].set_title(r'projected conductivity')

#subfigs[0].subplots_adjust(wspace=0)
#subfigs[0].layout('tight')
subfigs[0].suptitle('(a) conductivity field', fontsize=12)
#cbaxes = inset_axes(axes[1], width="5%", height="100%",) 
#subfigs[0].colorbar(c,cax=cbaxes,anchor=[1.5,1.5])


axes = subfigs[1].subplots(1,4,sharey=True)

axes[0].plot( theta[idx], data3[0][idx] , '-')
axes[0].plot( theta[idx], exact[0][idx] , 'k-')
axes[0].set_xlim([-np.pi-0.1,np.pi+0.1])
axes[0].set_xticks([-np.pi,0])
axes[0].set_xticklabels([r'$-\pi$',r'$0$'])
axes[0].set_xlabel(r'$\theta$')
axes[0].set_ylabel(r'$u$')
axes[0].set_title(r'$k=1$')
axes[0].yaxis.labelpad = -3
axes[0].xaxis.labelpad = 0
axes[1].plot( theta[idx], data3[1][idx] , '-')
axes[1].plot( theta[idx], exact[1][idx] , 'k-')
axes[1].set_xlim([-np.pi-0.1,np.pi+0.1])
axes[1].set_xticks([-np.pi,0])
axes[1].set_xticklabels([r'$-\pi$',r'$0$'])
axes[1].set_xlabel(r'$\theta$')
axes[1].set_title(r'$k=2$')
axes[1].xaxis.labelpad = 0
axes[2].plot( theta[idx], data3[2][idx] , '-')
axes[2].plot( theta[idx], exact[2][idx] , 'k-')
axes[2].set_xlim([-np.pi-0.1,np.pi+0.1])
axes[2].set_xticks([-np.pi,0])
axes[2].set_xticklabels([r'$-\pi$',r'$0$'])
axes[2].set_xlabel(r'$\theta$')
axes[2].set_title(r'$k=3$')
axes[2].xaxis.labelpad = 0
axes[3].plot( theta[idx], data3[3][idx] , '-')
axes[3].plot( theta[idx], exact[3][idx] , 'k-')
axes[3].set_xlim([-np.pi-0.1,np.pi+0.1])
axes[3].set_xticks([-np.pi,0,np.pi])
axes[3].set_xticklabels([r'$-\pi$',r'$0$',r'$\pi$'])
axes[3].set_xlabel(r'$\theta$')
axes[3].set_title(r'$k=4$')
axes[3].xaxis.labelpad = 0

axes[0].set_zorder(1)
axes[0].legend(['noisy data','exact data'])

subfigs[1].suptitle('(b) measurements with 20% noise', fontsize=12)

subfigs[1].subplots_adjust(wspace=0, bottom=0.2, top=0.8)

#data_20_noise = np.concatenate( [data3[0][idx], data3[1][idx], data3[2][idx], data3[3][idx] ] )
#data_exact = np.concatenate( [exact[0][idx], exact[1][idx], exact[2][idx], exact[3][idx] ] ) 

#print(data3.shape)
#print(exact.shape)

#axes[2].plot( theta[idx], data1[0][idx] , '.')
#axes[2].plot( theta[idx], data2[0][idx] , '.')
#axes[2].plot( data_20_noise , '.')
#axes[2].plot( data_exact , 'k-')


plt.savefig('./plots/data.pdf',format='pdf')