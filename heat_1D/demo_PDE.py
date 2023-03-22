#%%
import matplotlib.pyplot as plt
import os
from figures_util import matplotlib_setup
import arviz as az
az.style.use('default')
import numpy as np
from cuqi.pde import TimeDependentLinearPDE
from cuqi.geometry import StepExpansion, Continuous1D

# %%
SMALL_SIZE = 7
MEDIUM_SIZE =8
BIGGER_SIZE = 9
matplotlib_setup(SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE)
# %%
# Data directory
fig_dir = './figs/'

# Figure file
fig_dir = fig_dir 
# Check if the directory exists
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

version = 'v1'
fig_file = fig_dir +   'paper_demo_PDE_'+version+'.pdf'
fig_file2 = fig_dir +   'paper_demo2_PDE_'+version+'.png'

#%%
n_grid = 100
xi_a = 1
h = xi_a/(n_grid+1)
tau_b = 0.01
cfl = 5/11
dt_approx = cfl*h**2 
num_time_steps = int(tau_b/dt_approx)+1 


# Grid for the 1D heat PDE
grid = np.linspace(h, xi_a, n_grid, endpoint=False)

# Time steps
time_steps = np.linspace(0, tau_b, num_time_steps, endpoint=True)

tau_b2 = 0.02
num_time_steps2 = int(tau_b2/dt_approx)+1 
time_steps2 = np.linspace(0, tau_b2, num_time_steps2, endpoint=True)

# FD diffusion operator
D = (np.diag(-2*np.ones(n_grid)) + np.diag(np.ones(n_grid-1), -1)
     + np.diag(np.ones(n_grid-1), 1))/h**2

source_term = np.zeros(n_grid)  # RHS=0

# PDE form, returns (D, RHS, initial_condition)
def PDE_form(g, tau):
    return (D, source_term, g)

PDE = TimeDependentLinearPDE(
    PDE_form, time_steps, grid_sol=grid)

PDE2 = TimeDependentLinearPDE(
    PDE_form, time_steps2, grid_sol=grid)

g_exact =1/30*(1-np.cos(2*np.pi*(xi_a-grid)/(xi_a)))\
            +1/30*np.exp(-2*(10*(grid-0.5))**2)+\
             1/30*np.exp(-2*(10*(grid-0.8))**2)

PDE.assemble(g_exact)
u, info = PDE.solve()
u_obs = PDE.observe(u)

#%%
#Hack to show solutions at intermediate time steps
intermediate_indices = [2, 40, 100, len(time_steps)]
u_intermediate = np.zeros((len(intermediate_indices), n_grid))

for i, idx in enumerate(intermediate_indices):
    PDE_temp = TimeDependentLinearPDE(
        PDE_form, time_steps[:idx], grid_sol=grid)
    PDE_temp.assemble(g_exact)
    u_intermediate[i, :] = PDE_temp.solve()[0]



#%%
## plot prior posterior samples and ESS
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{bm}')
cm_to_in = 1/2.54
fig, axs = plt.subplots(nrows=1, ncols=3,
                        figsize=(17.8*cm_to_in, (17.8/4)*cm_to_in),
                        layout="constrained")

colors = ['C0', 'green', 'purple', 'k', 'gray']
# 1: g
plt.sca(axs[0])
plt.annotate('(a)', xy=(0.03, 0.93), xycoords='axes fraction')
plt.plot(grid, g_exact)
plt.ylabel('$\\bm{g}$')
plt.gca().yaxis.set_label_coords(-0.21, 0.45) #-0.12, 0.4

plt.xlabel('$\\xi$')
plt.gca().xaxis.set_label_coords(.5, -.12) #-0.12, 0.4
plt.gca().set_ylim(0, .13)
plt.xlim([0,1])

# 2: u
plt.sca(axs[1])
plt.annotate('(b)', xy=(0.03, 0.93), xycoords='axes fraction')
#plt.plot(grid, u)

for i in range(len(intermediate_indices)):
    plt.plot(grid, u_intermediate[i, :], color=colors[i], linestyle='--')

times_temp = [time_steps[i-1] for i in intermediate_indices]
plt.legend(['$t={:.3f}$'.format(t) for t in times_temp], loc='upper right', ncol=2, frameon=False, bbox_to_anchor=(1, 0.95))

#plt.plot(grid, u)
plt.ylabel('$\\bm{u}$')
plt.gca().yaxis.set_label_coords(-0.21, 0.45) #-0.12, 0.4

plt.xlabel('$\\xi$')
plt.gca().xaxis.set_label_coords(.5, -.12) #-0.12, 0.4
plt.gca().set_ylim(0, .13)
plt.xlim([0,1])

# 1,3: u obs
plt.sca(axs[2])
plt.annotate('(c)', xy=(0.03, 0.93), xycoords='axes fraction')
plt.plot(grid, u_obs)
plt.ylabel('$\\bm{y}$')
plt.gca().yaxis.set_label_coords(-0.21, 0.45) #-0.12, 0.4

plt.xlabel('$\\xi$')
plt.gca().xaxis.set_label_coords(.5, -.12) #-0.12, 0.4
plt.gca().set_ylim(0, .13)
plt.xlim([0,1])

#plt.gca().set_xticklabels(['v{}'.format(i) for i in range(c1_parameters["domain_dim"])])
fig.tight_layout(pad=0, w_pad=0.25, h_pad=0)
plt.savefig(fig_file, bbox_inches='tight', pad_inches=0.01, dpi=1200)


#%% domain and range 
domain_geometry = StepExpansion(grid, n_steps=3)
range_geometry = Continuous1D(grid)

parameters = [0, 1, 0.5]
model = cuqi.model.PDEModel(PDE2, range_geometry, domain_geometry)

y = model(parameters)




#%%
## plot geometry demo 
cm_to_in = 1/2.54
fig, axs = plt.subplots(nrows=1, ncols=3,
                        figsize=(17.8*cm_to_in, (17.8/4)*cm_to_in),
                        layout="constrained")

colors = ['C0', 'green', 'purple', 'k', 'gray']


# 1: step (par)
plt.sca(axs[0])
plt.annotate('(a)', xy=(0.03, 0.93), xycoords='axes fraction')

domain_geometry.plot(parameters, plot_par=True)


plt.ylabel('$x_i$')
plt.gca().yaxis.set_label_coords(-0.15, 0.45) #-0.12, 0.4

plt.xlabel('$i$')
plt.gca().xaxis.set_label_coords(.5, -.12) #-0.12, 0.4
plt.ylim([-0.2,1.2])
#plt.xlim([0,1])
tick_ids = np.linspace(0, 2, 3, dtype=int)
plt.xticks(tick_ids, tick_ids)



# 2: step (fun)
plt.sca(axs[1])
plt.annotate('(b)', xy=(0.03, 0.93), xycoords='axes fraction')
domain_geometry.plot(parameters)
plt.ylabel('$\\bm{g}$')
plt.gca().yaxis.set_label_coords(-0.15, 0.45) #-0.12, 0.4

plt.xlabel('$\\xi$')
plt.gca().xaxis.set_label_coords(.5, -.12) #-0.12, 0.4
plt.ylim([-0.2,1.2])
plt.xlim([0,1])



# 1,3: range (func)
plt.sca(axs[2])
plt.annotate('(c)', xy=(0.03, 0.93), xycoords='axes fraction')
range_geometry.plot(y)
plt.ylabel('$\\bm{y}$')
plt.gca().yaxis.set_label_coords(-0.15, 0.45) #-0.12, 0.4

plt.xlabel('$\\xi$')
plt.gca().xaxis.set_label_coords(.5, -.12) #-0.12, 0.4
#plt.gca().set_ylim(0, .13)
plt.ylim([-0.2,1.2])
plt.xlim([0,1])

#plt.gca().set_xticklabels(['v{}'.format(i) for i in range(c1_parameters["domain_dim"])])
fig.tight_layout(pad=0, w_pad=0.25, h_pad=0)
plt.savefig(fig_file2, bbox_inches='tight', pad_inches=0.01, dpi=1200)



# %%
