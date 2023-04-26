#%%
import pickle
import matplotlib
import matplotlib.pyplot as plt
import os
from load_cases import load_case
from figures_util import matplotlib_setup
import arviz as az
az.style.use('default')
import numpy as np
import cuqi


plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{bm}')
# %%
SMALL_SIZE = 7
MEDIUM_SIZE =8
BIGGER_SIZE = 9
    
matplotlib_setup(7, 8, 9)
# %%

# Data directory
fig_dir = './figs/'


# Figure file
fig_dir = fig_dir 
# Check if the directory exists
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

version = 'v8'
fig_file = fig_dir +   'paper_figure4_'+version+'.pdf'
fig_file_b = fig_dir + 'paper_figure4_'+version+'_b.pdf'
fig_file_c = fig_dir + 'paper_figure4_'+version+'_c.pdf'
fig_file_d = fig_dir + 'paper_figure4_'+version+'_d.pdf'

# %% Burnthin
if version == 'v4':
    Nb = 25000
    Nt =  None#1000# None
    Nb_2row = 1000
    Nb_3row = 1000
    Nb_4row = 1000

    case_files = ['./data2_cont6/paper_case2_b6_2', './data2_cont6/paper_case2_b3_2', './data2_cont6/paper_case2_b3_3', './data2_cont6/paper_case2_b3_5']

elif version == 'v5':
    Nb = 25000
    Nt =  10 # None#1000# None
    Nb_2row = 25000
    Nb_3row = 25000
    Nb_4row = 10000

    case_files = ['./data2_cont6/paper_case20','./data2_cont6/paper_case21', './data2_cont6/paper_case22', './data2_cont6/paper_case23'] 

elif version == 'v6':
    Nb = 25000
    Nt =  1 # None#1000# None
    Nb_2row = 10000
    Nb_3row = 10000
    Nb_4row = 10000

    case_files = ['./data2_cont6/paper_case24','./data2_cont6/paper_case25', './data2_cont6/paper_case26', './data2_cont6/paper_case29']

elif version == 'v7':
    Nb = 25000
    Nt =  None # None#1000# None
    Nb_2row = 10000
    Nb_3row = 10000
    Nb_4row = 10000
    n_ticks = 5
    num_var = 20

    case_files = ['./data2_cont6/paper_case35','./data2_cont6/paper_case36', './data2_cont6/paper_case37', './data2_cont6/paper_case38']

elif version == 'v7_test':
    Nb = 25000
    Nt =  500 # None#1000# None
    Nb_2row = 10000
    Nb_3row = 10000
    Nb_4row = 10000

    case_files = ['./data2_cont6/paper_case35','./data2_cont6/paper_case36', './data2_cont6/paper_case37', './data2_cont6/paper_case38']

elif version == 'v8':
    Nb = 25000
    Nt =  None # None#1000# None
    Nb_2row = 10000
    Nb_3row = 10000
    Nb_4row = 10000
    n_ticks = 5
    num_var = 20
        
    case_files = ['./data2_cont6/paper_case35','./data2_cont6/paper_case36', './data2_cont6/paper_case37', './data2_cont6/paper_case38']

else:
    raise ValueError('Unknown version')
# %% Create the figure
cm_to_in = 1/2.54
fig, axs = plt.subplots(nrows=4, ncols=3,
                        figsize=(17.8*cm_to_in, 17.8*cm_to_in),
                        layout="constrained")
prior_samples_3row, samples_3row, parameters_3row, _, _, _ = load_case(case_files[2], load_sol_data=True, load_prior_samples=True)

colors = ['C0', 'green', 'purple', 'k', 'gray']
# 1,1:  Basis func
#plt.rc('lines', markersize=SMALL_SIZE-3) 
plt.sca(axs[0,0])
geo =parameters_3row['x_exact_geometry'] 
a1_array = np.zeros(geo.par_dim)
a1_array[0] = 1
a1 = cuqi.array.CUQIarray(a1_array, geometry=geo)

a2_array = np.zeros(geo.par_dim)
a2_array[1] = 1
a2 = cuqi.array.CUQIarray(a2_array, geometry=geo)

a3_array = np.zeros(geo.par_dim)
a3_array[2] = 1
a3 = cuqi.array.CUQIarray(a3_array, geometry=geo)

a4_array = np.zeros(geo.par_dim)
a4_array[3] = 1
a4 = cuqi.array.CUQIarray(a4_array, geometry=geo)


a1.plot( label='$\\bm{e}_1$', linestyle='--', color=colors[0])
a2.plot( label='$\\bm{e}_2$', linestyle='--', color=colors[1])
a3.plot( label='$\\bm{e}_3$', linestyle='--', color=colors[2])
a4.plot( label='$\\bm{e}_4$', linestyle='--', color=colors[3])



#plt.plot([0,1,2],parameters1["ESS"], 'd-', label=str(parameters1["noise_level"]*100)+"$\%$ noise") 
#plt.plot([0,1,2],parameters2["ESS"], 'd-', label=str(parameters2["noise_level"]*100)+"$\%$ noise", color='green') 
plt.annotate('(a)', xy=(0.03, 0.93), xycoords='axes fraction')
plt.legend(frameon=False, ncols=2)#loc='center right', bbox_to_anchor=(1., 0.27))
#plt.ylabel()
plt.ylim([-0.05, 0.15])
plt.gca().yaxis.set_label_coords(-0.18, 0.5) #-0.12, 0.4

plt.xlim([0, 1])
plt.xlabel('$\\xi$')
plt.gca().xaxis.set_label_coords(.53, -.12) #-0.12, 0.4
#plt.xticks(range(0, parameters1["domain_dim"]))
#plt.gca().set_xticklabels(['v{}'.format(i) for i in range(c1_parameters["domain_dim"])])





# 1,1: Case 2, prior samples
prior_samples_1row, _, _, _, _, _ = load_case(case_files[0], load_sol_data=True, load_prior_samples=True)

idx = 0
plt.sca(axs[0,1])
plt.annotate('(b)', xy=(0.03, 0.93), xycoords='axes fraction')
for s in prior_samples_1row:
    prior_samples_3row.geometry.plot(s, is_par=True, color=colors[idx]) 
    idx += 1
    if idx == 5:
        break
plt.ylabel('$\\bm{g}$')
#plt.gca().yaxis.set_label_coords(-0.21, 0.45) #-0.12, 0.4
plt.gca().yaxis.set_label_coords(-0.16, 0.5)

plt.xlabel('$\\xi$')
plt.gca().xaxis.set_label_coords(.5, -.12) #-0.12, 0.4
plt.gca().set_ylim(-.175, .175)
plt.xlim([0,1])


# 1,2: Case 2, posterior samples


idx = 0
plt.sca(axs[0,2])
plt.annotate('(c)', xy=(0.03, 0.93), xycoords='axes fraction')
for s in samples_3row.burnthin(1000,5000):
    samples_3row.geometry.plot(s, is_par=True, color=colors[idx])   
    idx += 1
    if idx == 5:
        break
plt.ylabel('$\\bm{g}$')
#plt.gca().yaxis.set_label_coords(-0.21, 0.45) #-0.12, 0.4
plt.gca().yaxis.set_label_coords(-0.16, 0.5)
plt.xlabel('$\\xi$')
plt.gca().xaxis.set_label_coords(.5, -.12) #-0.12, 0.4
plt.gca().set_ylim(-.015, .17)
plt.xlim([0,1])

## 1,3: Case 2, ESS
#plt.rc('lines', markersize=SMALL_SIZE-3) 
#plt.sca(axs[0,2])
#plt.plot(parameters_1row["ESS"], 'd-',
#         label=str(parameters_1row["noise_level"]*100)+"$\%$ noise")
#
#plt.plot(parameters_2row["ESS"], 'd-', 
#         label=str(parameters_2row["noise_level"]*100)+"$\%$ noise", color='green') 
#
#plt.plot(parameters_3row["ESS"], 'd-',
#         label=str(parameters_3row["noise_level"]*100)+"$\%$ noise", color='k')
#
#plt.plot(parameters_4row["ESS"], 'd-',
#         label=str(parameters_4row["noise_level"]*100)+"$\%$ noise*", color='gray') 
#
#plt.annotate('(c)', xy=(0.03, 0.93), xycoords='axes fraction')
#plt.legend(frameon=False)#loc='center right', bbox_to_anchor=(1., 0.27))
#plt.ylabel('ESS($X_i$)')
#plt.gca().yaxis.set_label_coords(-0.19, 0.45) #-0.12, 0.4
#
#plt.xlabel('$i$')
#plt.gca().xaxis.set_label_coords(.5, -.12) #-0.12, 0.4
#
#plt.xticks(range(0, parameters_1row["domain_dim"], 2))
#plt.xlim([0,20])
##plt.gca().set_xticklabels(['v{}'.format(i) for i in range(c1_parameters["domain_dim"])])
#fig.tight_layout(pad=0, w_pad=0.25, h_pad=0)
#plt.savefig(fig_file_b, bbox_inches='tight', pad_inches=0.01, dpi=1200)
#
#
#
# 1,1: Case 1, exact solution, exact data, noisy data
prior_samples_1row, samples_1row, parameters_1row, x_exact, y_exact, data = load_case(case_files[0], load_sol_data=True, load_prior_samples=True)
plt.sca(axs[1,0])
plt.annotate('(d)', xy=(0.03, 0.93), xycoords='axes fraction')
x_exact.plot()
y_exact.plot()
data.plot()
plt.legend(['Exact solution', 'Exact data', 'Noisy data']);
plt.ylim([0,0.17])
plt.yticks([0,0.05,0.1,0.15])
plt.xlim([0,1])
#plt.ylabel('$u$')
plt.gca().yaxis.set_label_coords(-0.16, 0.5)
plt.xlabel('$\\xi$')
plt.gca().xaxis.set_label_coords(0.5, -0.14)

# 1,2: Case 1, cont CI
plt.sca(axs[1,1])
plt.annotate('(e)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples_1row.burnthin(Nb,Nt).funvals.plot_ci(95, plot_par=False, exact=x_exact)
lci[0].set_label("Mean")
lci[1].set_label("Exact")
lci[2].set_label("$95\\%\;\mathrm{CI}$")
plt.legend()
plt.ylim([0,0.17])
plt.yticks([0,0.05,0.1,0.15])
plt.xlim([0,1])
plt.ylabel('$\\bm{g}$')
plt.gca().yaxis.set_label_coords(-0.16, 0.5)
plt.xlabel('$\\xi$')
plt.gca().xaxis.set_label_coords(0.5, -0.14)

# 1,3: Case 1, disc CI
plt.sca(axs[1,2])
plt.annotate('(f)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples_1row.burnthin(Nb,Nt).plot_ci(95, plot_par=True, exact=x_exact, markersize=SMALL_SIZE-3)
lci[0].set_label("Mean")
lci[1].set_label("Exact")
lci[2].set_label("$95\\%\;\mathrm{CI}$")
plt.legend(loc = 'lower right')
#plt.ylim([0,0.17])
#plt.yticks([0,0.05,0.1,0.15])
#plt.xlim([0,1])
plt.ylabel('$x_i$')
plt.gca().yaxis.set_label_coords(-0.09, 0.5)
plt.xlabel('$i$')
plt.gca().xaxis.set_label_coords(0.5, -0.14)
tick_ids = np.linspace(0, num_var-1, n_ticks, dtype=int)
plt.xticks(tick_ids, tick_ids)#[parameters_1row['x_exact_geometry'].variables[i] for i in tick_ids])

## 2,1: Case 2, exact solution, exact data, noisy data
#prior_samples_2row, samples_2row, parameters_2row, x_exact, y_exact, data = load_case(case_files[1], load_sol_data=True, load_prior_samples=True)
#plt.sca(axs[1,0])
#plt.annotate('(d)', xy=(0.03, 0.93), xycoords='axes fraction')
#x_exact.plot()
#y_exact.plot()
#data.plot()
#plt.legend(['Exact solution', 'Exact data', 'Noisy data']);
#plt.ylim([0,0.17])
#plt.yticks([0,0.05,0.1,0.15])
#plt.xlim([0,1])
##plt.ylabel('$u$')
#plt.gca().yaxis.set_label_coords(-0.16, 0.5)
#plt.xlabel('$\\xi$')
#plt.gca().xaxis.set_label_coords(0.5, -0.14)
#
## 2,2: Case 2, cont CI
#plt.sca(axs[1,1])
#plt.annotate('(e)', xy=(0.03, 0.93), xycoords='axes fraction')
#lci = samples_2row.burnthin(Nb_2row,Nt).funvals.plot_ci(95, plot_par=False, exact=x_exact)
#lci[0].set_label("Mean")
#lci[1].set_label("Exact")
#lci[2].set_label("95% CI")
#plt.legend()
#plt.ylim([0,0.17])
#plt.yticks([0,0.05,0.1,0.15])
#plt.xlim([0,1])
#plt.ylabel('$g(\\xi;\\mathbf{x})$')
#plt.gca().yaxis.set_label_coords(-0.16, 0.5)
#plt.xlabel('$\\xi$')
#plt.gca().xaxis.set_label_coords(0.5, -0.14)
#
## 2,3: Case 2, disc CI
#plt.sca(axs[1,2])
#plt.annotate('(f)', xy=(0.03, 0.93), xycoords='axes fraction')
#lci = samples_2row.burnthin(Nb_2row,Nt).plot_ci(95, plot_par=True, exact=x_exact, markersize=SMALL_SIZE-3)
#lci[0].set_label("Mean")
#lci[1].set_label("Exact")
#lci[2].set_label("95% CI")
#plt.legend(loc = 'lower right')
##plt.ylim([0,0.17])
##plt.yticks([0,0.05,0.1,0.15])
##plt.xlim([0,1])
#plt.ylabel('$X_i$')
#plt.gca().yaxis.set_label_coords(-0.09, 0.5)
#plt.xlabel('$i$')
#plt.gca().xaxis.set_label_coords(0.5, -0.14)
#tick_ids = np.linspace(0, num_var-1, n_ticks, dtype=int)
#plt.xticks(tick_ids, tick_ids)

# 3,1: Case 3, exact solution, exact data, noisy data
prior_samples_3row, samples_3row, parameters_3row, x_exact, y_exact, data = load_case(case_files[2], load_sol_data=True, load_prior_samples=True)
plt.sca(axs[2,0])
plt.annotate('(g)', xy=(0.03, 0.93), xycoords='axes fraction')
x_exact.plot()
y_exact.plot()
data.plot()
plt.legend(['Exact solution', 'Exact data', 'Noisy data']);
plt.ylim([0,0.17])
plt.yticks([0,0.05,0.1,0.15])
plt.xlim([0,1])
#plt.ylabel('$u$')
plt.gca().yaxis.set_label_coords(-0.16, 0.5)
plt.xlabel('$\\xi$')
plt.gca().xaxis.set_label_coords(0.5, -0.14)

# 3,2: Case 3, cont CI
plt.sca(axs[2,1])
plt.annotate('(h)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples_3row.burnthin(Nb_3row,Nt).funvals.plot_ci(95, plot_par=False, exact=x_exact)
lci[0].set_label("Mean")
lci[1].set_label("Exact")
lci[2].set_label("$95\\%\;\mathrm{CI}$")
plt.legend()
plt.ylim([0,0.17])
plt.yticks([0,0.05,0.1,0.15])
plt.xlim([0,1])
plt.ylabel('$\\bm{g}$')
plt.gca().yaxis.set_label_coords(-0.16, 0.5)
plt.xlabel('$\\xi$')
plt.gca().xaxis.set_label_coords(0.5, -0.14)

# 3,3: Case 3, disc CI
plt.sca(axs[2,2])
plt.annotate('(i)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples_3row.burnthin(Nb_3row,Nt).plot_ci(95, plot_par=True, exact=x_exact, markersize=SMALL_SIZE -3)
lci[0].set_label("Mean")
lci[1].set_label("Exact")
lci[2].set_label("$95\\%\;\mathrm{CI}$")
plt.legend(loc = 'lower right')
#plt.ylim([0,0.17])
#plt.yticks([0,0.05,0.1,0.15])
#plt.xlim([0,1])
plt.ylabel('$x_i$')
plt.gca().yaxis.set_label_coords(-0.09, 0.5)
plt.xlabel('$i$')
plt.gca().xaxis.set_label_coords(0.5, -0.14)
tick_ids = np.linspace(0, num_var-1, n_ticks, dtype=int)
plt.xticks(tick_ids, tick_ids)


# 4,1: Case 4, exact solution, exact data, noisy data
prior_samples_4row, samples_4row, parameters_4row, x_exact, y_exact, data = load_case(case_files[3], load_sol_data=True, load_prior_samples=True)
plt.sca(axs[3,0])
plt.annotate('(j)', xy=(0.03, 0.93), xycoords='axes fraction')
x_exact.plot()
y_exact.plot()
data.plot()
plt.legend(['Exact solution', 'Exact data', 'Noisy data']);
plt.ylim([0,0.17])
plt.yticks([0,0.05,0.1,0.15])
plt.xlim([0,1])
#plt.ylabel('$u$')
plt.gca().yaxis.set_label_coords(-0.16, 0.5)
plt.xlabel('$\\xi$')
plt.gca().xaxis.set_label_coords(0.5, -0.14)

# 4,2: Case 4, cont CI
plt.sca(axs[3,1])
plt.annotate('(k)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples_4row.burnthin(Nb_4row,Nt).funvals.plot_ci(95, plot_par=False, exact=x_exact)
lci[0].set_label("Mean")
lci[1].set_label("Exact")
lci[2].set_label("$95\\%\;\mathrm{CI}$")
plt.legend()
plt.ylim([0,0.17])
plt.yticks([0,0.05,0.1,0.15])
plt.xlim([0,1])
plt.ylabel('$\\bm{g}$')
plt.gca().yaxis.set_label_coords(-0.16, 0.5)
plt.xlabel('$\\xi$')
plt.gca().xaxis.set_label_coords(0.5, -0.14)

# 4,3: Case 4, disc CI
plt.sca(axs[3,2])
plt.annotate('(l)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples_4row.burnthin(Nb_4row,Nt).plot_ci(95, plot_par=True, exact=x_exact, markersize=SMALL_SIZE -3)
lci[0].set_label("Mean")
lci[1].set_label("Exact")
lci[2].set_label("$95\\%\;\mathrm{CI}$")
plt.legend(loc = 'lower right')
#plt.ylim([0,0.17])
#plt.yticks([0,0.05,0.1,0.15])
#plt.xlim([0,1])
plt.ylabel('$x_i$')
plt.gca().yaxis.set_label_coords(-0.09, 0.5)
plt.xlabel('$i$')
plt.gca().xaxis.set_label_coords(0.5, -0.14)
tick_ids = np.linspace(0, num_var-1, n_ticks, dtype=int)
plt.xticks(tick_ids, tick_ids)



fig.tight_layout(pad=0, w_pad=0.1, h_pad=0.9)
plt.savefig(fig_file, bbox_inches='tight', pad_inches=0.01, dpi=1200)

#%%
## plot prior posterior samples and ESS
cm_to_in = 1/2.54
fig, axs = plt.subplots(nrows=1, ncols=3,
                        figsize=(17.8*cm_to_in, (17.8/3.4)*cm_to_in),
                        layout="constrained")

colors = ['C0', 'green', 'purple', 'k', 'gray']
# 1,1: Case 2, prior samples
idx = 0
plt.sca(axs[0])
plt.annotate('(a)', xy=(0.03, 0.93), xycoords='axes fraction')
for s in prior_samples_1row:
    prior_samples_1row.geometry.plot(s, is_par=True, color=colors[idx]) 
    idx += 1
    if idx == 5:
        break
plt.ylabel('$g(\\xi;\\mathbf{x})$')
plt.gca().yaxis.set_label_coords(-0.21, 0.45) #-0.12, 0.4

plt.xlabel('$\\xi$')
plt.gca().xaxis.set_label_coords(.5, -.12) #-0.12, 0.4
plt.gca().set_ylim(-.175, .175)
plt.xlim([0,1])
# 1,2: Case 2, posterior samples
idx = 0
plt.sca(axs[1])
plt.annotate('(b)', xy=(0.03, 0.93), xycoords='axes fraction')
for s in samples_3row.burnthin(1000,5000):
    samples_1row.geometry.plot(s, is_par=True, color=colors[idx])   
    idx += 1
    if idx == 5:
        break
plt.ylabel('$\\bm{g}$')
plt.gca().yaxis.set_label_coords(-0.21, 0.45) #-0.12, 0.4

plt.xlabel('$\\xi$')
plt.gca().xaxis.set_label_coords(.5, -.12) #-0.12, 0.4
plt.gca().set_ylim(-.015, .17)
plt.xlim([0,1])

# 1,3: Case 2, ESS
plt.rc('lines', markersize=SMALL_SIZE-3) 
plt.sca(axs[2])
plt.plot(parameters_1row["ESS"], 'd-',
         label=str(parameters_1row["noise_level"]*100)+"$\%$ noise")

plt.plot(parameters_2row["ESS"], 'd-', 
         label=str(parameters_2row["noise_level"]*100)+"$\%$ noise", color='green') 

plt.plot(parameters_3row["ESS"], 'd-',
         label=str(parameters_3row["noise_level"]*100)+"$\%$ noise", color='k')

plt.plot(parameters_4row["ESS"], 'd-',
         label=str(parameters_4row["noise_level"]*100)+"$\%$ noise*", color='gray') 

plt.annotate('(c)', xy=(0.03, 0.93), xycoords='axes fraction')
plt.legend(frameon=False)#loc='center right', bbox_to_anchor=(1., 0.27))
plt.ylabel('ESS($X_i$)')
plt.gca().yaxis.set_label_coords(-0.19, 0.45) #-0.12, 0.4

plt.xlabel('$i$')
plt.gca().xaxis.set_label_coords(.5, -.12) #-0.12, 0.4

plt.xticks(range(0, parameters_1row["domain_dim"], 2))
plt.xlim([0,20])
#plt.gca().set_xticklabels(['v{}'.format(i) for i in range(c1_parameters["domain_dim"])])
fig.tight_layout(pad=0, w_pad=0.25, h_pad=0)
plt.savefig(fig_file_b, bbox_inches='tight', pad_inches=0.01, dpi=1200)
#%%
## plot pair
#cm_to_in = 1/2.54
#nrows = 3
#ncols = 3
#fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
#                        figsize=(17.8*cm_to_in, 17.8*cm_to_in),
#                        layout="constrained")
#pair_ideces = [0,1,2,15] 
#samples_3row.geometry.variables = ['$X_{'+str(i)+'}$' for i in range(20)]
#samples_3row.burnthin(Nb_3row).plot_pair(pair_ideces, ax=axs)
#for ax in axs.flat:
#    ax.set_rasterized(True)
#
#for row_idx in range(nrows):
#    for col_idx in range(ncols):
#        if col_idx != 0:
#            axs[row_idx, col_idx].yaxis.set_ticks([])
#        if row_idx != nrows-1:
#            axs[row_idx, col_idx].xaxis.set_ticks([])
#
#fig.tight_layout(pad=0, w_pad=0, h_pad=0)
#plt.savefig(fig_file_c, bbox_inches='tight', pad_inches=0.01, dpi=1200)
#
#
#
##%% plot trace
#az.style.use('arviz-grayscale')
#matplotlib_setup(7, 8, 9)
##az.style.use('default')
#
#cm_to_in = 1/2.54
#fig, axs = plt.subplots(nrows=4, ncols=2,
#                        figsize=(17.8*cm_to_in, 12*cm_to_in),
#                        layout="constrained")
#samples_3row.burnthin(1000,5).plot_trace([0,1,2,15], axes=axs, backend_kwargs={'facecolor':'black'}, backend="matplotlib")
#fig.tight_layout(pad=0, w_pad=0.1, h_pad=0.5)
#
#for ax in axs.flatten():
#    ax.title.set_fontsize(BIGGER_SIZE)
#    ax.xaxis.label.set_size(MEDIUM_SIZE)
#    ax.yaxis.label.set_size(MEDIUM_SIZE) 
#    ax.tick_params(axis='both', which='major', labelsize=SMALL_SIZE, pad=3)
#    ax.tick_params(axis='both', which='minor', labelsize=SMALL_SIZE)
#    
#    ax.set_rasterized(True)
#    #ax.title.set_text(style)
#plt.savefig(fig_file_d, bbox_inches='tight', pad_inches=0.01, dpi=1200)
#az.style.use('default')
#matplotlib_setup(7, 8, 9) 
#  # %%
#import numpy as np
#for key in parameters_1row:
#    if np.any(parameters_1row[key] != parameters_2row[key]) or\
#        np.any(parameters_1row[key] != parameters_3row[key]) or\
#            np.any(parameters_1row[key] != parameters_4row[key]):	
#    
#        print("###",key,"###")
#        print(parameters_1row[key], parameters_2row[key], parameters_3row[key], parameters_4row[key])
#
## %%
#
