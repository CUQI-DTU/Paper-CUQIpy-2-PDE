#%%
import pickle
import matplotlib
import matplotlib.pyplot as plt
import os
from load_cases import load_case
import numpy as np
from figures_util import matplotlib_setup
import arviz as az
import cuqi

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{bm}')

# %%
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
plt.rc('lines', markersize=SMALL_SIZE-3) 
# %%

# Data directory
fig_dir = './figs/'

# Figure file
fig_dir = fig_dir 
# Check if the directory exists
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

version = 'v5'
if version == 'v0':
    Nb = 100
    Nt = None
    case_files = ['./data2_cont5/paper_case17', './data2_cont6/paper_case17_b']

elif version == 'v1':
    Nb = 1000
    Nt = None
    num_var = 3
    n_ticks = 5
    case_files = ['./data2_cont6/paper_case39', './data2_cont6/paper_case40']

elif version == 'v2':
    Nb = 1000
    Nt = None
    num_var = 3
    n_ticks = 5
    case_files = ['./data2_cont6/paper_case41', './data2_cont6/paper_case42']

elif version == 'v3':
    Nb = 200
    Nt = None
    num_var = 3
    n_ticks = 5
    case_files = ['./data2_cont6/paper_case43', './data2_cont6/paper_case44']

elif version == 'v4':
    Nb = 200
    Nt = None
    num_var = 3
    n_ticks = 5
    case_files = ['./data2_cont6/paper_case45', './data2_cont6/paper_case46']

elif version == 'v5':
    Nb = 200
    Nt = None
    num_var = 3
    n_ticks = 5
    case_files = ['./data2_cont6/paper_case45', './data2_cont6/paper_case46']

else:
    raise ValueError("Unknown version")

fig_file = fig_dir + 'paper_figure5_'+version+'.pdf'
fig_file_b = fig_dir + 'paper_figure5_'+version+'_b.pdf'
fig_file_c = fig_dir + 'paper_figure5_'+version+'_c.pdf'
fig_file_d = fig_dir + 'paper_figure5_'+version+'_d.pdf'

prior_samples1, samples1, parameters1, x_exact1, y_exact1, data1 = load_case(case_files[0], load_sol_data=True, load_prior_samples=True)
prior_samples2, samples2, parameters2, x_exact2, y_exact2, data2 = load_case(case_files[1], load_sol_data=True, load_prior_samples=True)
# %% Burnthin

# %% Create the figure
cm_to_in = 1/2.54
fig, axs = plt.subplots(nrows=2, ncols=3,
                        figsize=(17.8*cm_to_in, 2/3*13.5*cm_to_in),
                        layout="constrained")

## plot prior posterior samples and ESS
colors = ['C0', 'green', 'purple', 'k', 'gray']


# 1,1:  Basis func
plt.rc('lines', markersize=SMALL_SIZE-3) 
plt.sca(axs[0,0])

a1 = cuqi.array.CUQIarray([1,0,0], geometry=parameters2['x_exact_geometry'])
a2 = cuqi.array.CUQIarray([0,1,0], geometry=parameters2['x_exact_geometry'])
a3 = cuqi.array.CUQIarray([0,0,1], geometry=parameters2['x_exact_geometry'])

a1.plot( label='$\\bm{\\chi}_1$', linestyle='--')
a2.plot( label='$\\bm{\\chi}_2$', linestyle='--')
a3.plot( label='$\\bm{\\chi}_3$', linestyle='--')



#plt.plot([0,1,2],parameters1["ESS"], 'd-', label=str(parameters1["noise_level"]*100)+"$\%$ noise") 
#plt.plot([0,1,2],parameters2["ESS"], 'd-', label=str(parameters2["noise_level"]*100)+"$\%$ noise", color='green') 
plt.annotate('(a)', xy=(0.03, 0.93), xycoords='axes fraction')
plt.legend(frameon=False)#loc='center right', bbox_to_anchor=(1., 0.27))
#plt.ylabel()
plt.ylim([-0.2, 1.2])
plt.gca().yaxis.set_label_coords(-0.18, 0.5) #-0.12, 0.4

plt.xlabel('$\\xi$')
plt.gca().xaxis.set_label_coords(.53, -.12) #-0.12, 0.4
#plt.xticks(range(0, parameters1["domain_dim"]))
#plt.gca().set_xticklabels(['v{}'.format(i) for i in range(c1_parameters["domain_dim"])])

# 1,2: prior samples
idx = 0
plt.sca(axs[0,1])
for s in prior_samples1:
    prior_samples1.geometry.plot(s, is_par=True, color=colors[idx]) 
    idx += 1
    if idx == 5:
        break
plt.ylabel('$\\bm{g}$')
plt.gca().yaxis.set_label_coords(-0.15, 0.5) #-0.12, 0.4
plt.xlabel('$\\xi$')
plt.gca().xaxis.set_label_coords(.5, -.12) #-0.12, 0.4
#plt.gca().set_ylim(-.22, .10)
plt.annotate('(b)', xy=(0.03, 0.93), xycoords='axes fraction')

# 1,3: posterior samples
idx = 0
plt.sca(axs[0,2])
for s in samples2.burnthin(1000,1000):
    samples2.geometry.plot(s, is_par=True, color=colors[idx])   
    idx += 1
    if idx == 5:
        break

plt.ylabel('$\\bm{g}$')
plt.gca().yaxis.set_label_coords(-0.15, 0.5) #-0.12, 0.4

plt.xlabel('$\\xi$')
plt.gca().xaxis.set_label_coords(.5, -.12) #-0.12, 0.4
#plt.gca().set_ylim(-.015, .17)
plt.annotate('(c)', xy=(0.03, 0.93), xycoords='axes fraction')





## 2,1: Case 3, exact solution, exact data, noisy data
#plt.sca(axs[1,0])
#plt.annotate('(d)', xy=(0.03, 0.93), xycoords='axes fraction')
#x_exact1.plot()
#y_exact1.plot()
#data1.plot()
#plt.legend(['Exact solution', 'Exact data', 'Noisy data'], bbox_to_anchor=(.629, 0), loc='lower center', ncol=1);
#plt.ylim([-0.2,1.2])
#plt.yticks([0,0.25,0.5,0.75, 1])
#plt.xlim([0,1])
##plt.ylabel('$u$')
#plt.gca().yaxis.set_label_coords(-0.18, 0.5) #-0.12, 0.4
#plt.xlabel('$\\xi$')
#plt.gca().xaxis.set_label_coords(.5, -.12) #-0.12, 0.4

## 2,2: Case 3, cont CI
#plt.sca(axs[1,1])
#plt.annotate('(e)', xy=(0.03, 0.93), xycoords='axes fraction')
#lci = samples1.burnthin(Nb,Nt).funvals.plot_ci(95, plot_par=False, exact=x_exact1)
#lci[0].set_label("Mean")
#lci[1].set_label("Exact")
#lci[2].set_label("95% CI")
#plt.legend()
#plt.ylim([-0.2,1.2])
#plt.yticks([0,0.25,0.5,0.75, 1])
#plt.xlim([0,1])
#plt.ylabel('$g(\\xi;\\mathbf{x})$')
#plt.gca().yaxis.set_label_coords(-0.18, 0.5) #-0.12, 0.4
#plt.xlabel('$\\xi$')
#plt.gca().xaxis.set_label_coords(.5, -.12) #-0.12, 0.4
#
## 2,3: Case 3, disc CI
#plt.sca(axs[1,2])
#plt.annotate('(f)', xy=(0.03, 0.93), xycoords='axes fraction')
#lci = samples1.burnthin(Nb,Nt).plot_ci(95, plot_par=True, exact=x_exact1, markersize=SMALL_SIZE-3)
#lci[0].set_label("Mean")
#lci[1].set_label("Exact")
#lci[2].set_label("95% CI")
#plt.legend(ncols=1, loc ="upper right")
##plt.ylim([-2.2,3.9])
##plt.yticks([0,0.05,0.1,0.15])
##plt.xlim([0,1])
#plt.ylabel('$X_i$')
#plt.gca().yaxis.set_label_coords(-0.15, 0.5) #-0.12, 0.4
#plt.xlabel('$i$')
#plt.gca().xaxis.set_label_coords(.53, -.12) #-0.12, 0.4
#tick_ids = np.linspace(0, num_var-1, n_ticks, dtype=int)
#plt.xticks(tick_ids, tick_ids)


# 3,1: Case 3_b, exact solution, exact data, noisy data
plt.sca(axs[1,0])
plt.annotate('(d)', xy=(0.03, 0.93), xycoords='axes fraction')
x_exact2.plot()
y_exact2.plot()
data2.plot()
plt.legend(['Exact solution', 'Exact data', 'Noisy data'], bbox_to_anchor=(.629, 0), loc='lower center', ncol=1);
plt.ylim([-0.2,1.2])
plt.yticks([0,0.25,0.5,0.75, 1])
plt.xlim([0,1])
#plt.ylabel('$u$')
plt.gca().yaxis.set_label_coords(-0.18, 0.5) #-0.12, 0.4
plt.xlabel('$\\xi$')
plt.gca().xaxis.set_label_coords(.5, -.12) #-0.12, 0.4

# 3,2: Case 3_b, cont CI
plt.sca(axs[1,1])
plt.annotate('(e)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples2.burnthin(Nb,Nt).funvals.plot_ci(95, plot_par=False, exact=x_exact2)
lci[0].set_label("Mean")
lci[1].set_label("Exact")
lci[2].set_label('$95\\%\; \\mathrm{CI}$')
plt.legend()
plt.ylim([-0.2,1.2])
plt.yticks([0,0.25,0.5,0.75, 1])
plt.xlim([0,1])
plt.ylabel('$\\bm{g}$')
#plt.ylabel('$g(\\xi;\\mathbf{x})$')
plt.gca().yaxis.set_label_coords(-0.18, 0.5) #-0.12, 0.4
plt.xlabel('$\\xi$')
plt.gca().xaxis.set_label_coords(.5, -.12) #-0.12, 0.4

# 3,3: Case 3_b, disc CI
plt.sca(axs[1,2])
plt.annotate('(f)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples2.burnthin(Nb,Nt).plot_ci(95, plot_par=True, exact=x_exact2,  markersize=SMALL_SIZE-3)
lci[0].set_label("Mean")
lci[1].set_label("Exact")
lci[2].set_label("$95\\%\; \\mathrm{CI}$")
plt.legend()
#plt.ylim([0,0.17])
#plt.yticks([0,0.05,0.1,0.15])
#plt.xlim([0,1])
plt.ylabel('$x_i$')
plt.gca().yaxis.set_label_coords(-0.15, 0.5) #-0.12, 0.4
plt.xlabel('$i$')
plt.gca().xaxis.set_label_coords(.53, -.12) #-0.12, 0.4
tick_ids = np.linspace(0, num_var-1, n_ticks, dtype=int)
plt.xticks(tick_ids, tick_ids)
plt.legend(ncols=1, loc ="upper right")
#plt.ylim([-2.2,3.9])


fig.tight_layout(pad=0, w_pad=0.1, h_pad=0.9)
plt.savefig(fig_file, bbox_inches='tight', pad_inches=0.01, dpi=1200)



#%%
## plot pair
az.style.use('arviz-grayscale')
matplotlib_setup(7, 8, 9)

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{bm}')

cm_to_in = 1/2.54
parentfig = plt.figure( figsize=(17.8*cm_to_in, 10*cm_to_in), constrained_layout=True)#,layout='constrained')

subfigs = parentfig.subfigures(1, 2, wspace=0.1, width_ratios=[.8, 1])

nrows = 2
ncols = 2
axs = subfigs[0].subplots(nrows=nrows, ncols=ncols)
pair_ideces = [0,1,2] 
samples2.geometry.variables = ['$x_{'+str(i)+'}$' for i in range(3)]
samples2.burnthin(1000,5).plot_pair(pair_ideces, ax=axs)
for ax in axs.flat:
    ax.set_rasterized(True)

for row_idx in range(nrows):
    for col_idx in range(ncols):
        if col_idx != 0:
            axs[row_idx, col_idx].yaxis.set_ticks([])
        if row_idx != nrows-1:
            axs[row_idx, col_idx].xaxis.set_ticks([])
subfigs[0].suptitle('(a) pair plot', fontsize=12)

# plot trace
az.style.use('arviz-grayscale')
matplotlib_setup(7, 8, 9)
#az.style.use('default')

cm_to_in = 1/2.54
axs =  subfigs[1].subplots(nrows=3, ncols=2)
samples2.burnthin(1000,5).plot_trace([0,1,2], axes=axs, backend_kwargs={'facecolor':'black'}, backend="matplotlib")

for ax in axs.flatten():
    ax.title.set_fontsize(BIGGER_SIZE)
    ax.xaxis.label.set_size(MEDIUM_SIZE)
    ax.yaxis.label.set_size(MEDIUM_SIZE) 
    ax.tick_params(axis='both', which='major', labelsize=SMALL_SIZE, pad=3)
    ax.tick_params(axis='both', which='minor', labelsize=SMALL_SIZE)
    
    ax.set_rasterized(True)
    #ax.title.set_text(style)
subfigs[1].suptitle('(b) trace plot', fontsize=12)
axs.flatten()[-1].set_xlabel('Iteration', color='w')
axs.flatten()[-1].xaxis.set_label_coords(0.5, -0.27)
parentfig.tight_layout(pad=0, w_pad=0, h_pad=0)


plt.savefig(fig_file_d, bbox_inches='tight', pad_inches=0.01, dpi=1200)
#plt.savefig(fig_file_c, bbox_inches='tight', pad_inches=0.01, dpi=1200)
az.style.use('default')
matplotlib_setup(7, 8, 9) 
  # %%
