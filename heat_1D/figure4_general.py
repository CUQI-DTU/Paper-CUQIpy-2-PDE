#%%
import pickle
import matplotlib
import matplotlib.pyplot as plt
import os
from load_cases import load_case

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


# %%

# Data directory
fig_dir = './figs/'


# Figure file
fig_dir = fig_dir 
# Check if the directory exists
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

version = 'v6'
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
else:
    raise ValueError('Unknown version')
# %% Create the figure
cm_to_in = 1/2.54
fig, axs = plt.subplots(nrows=4, ncols=3,
                        figsize=(17.8*cm_to_in, 17.8*cm_to_in),
                        layout="constrained")

# 1,1: Case 2, exact solution, exact data, noisy data
prior_samples_1row, samples_1row, parameters_1row, x_exact, y_exact, data = load_case(case_files[0], load_sol_data=True, load_prior_samples=True)
plt.sca(axs[0,0])
plt.annotate('(a)', xy=(0.03, 0.93), xycoords='axes fraction')
x_exact.plot()
y_exact.plot()
data.plot()
plt.legend(['Exact solution', 'Exact data', 'Noisy data']);
plt.ylim([0,0.17])
plt.yticks([0,0.05,0.1,0.15])
plt.xlim([0,1])
plt.ylabel('$g(x)$')
plt.xlabel('$x$')
plt.gca().xaxis.set_label_coords(0.5, -0.08)

# 1,2: Case 2, cont CI
plt.sca(axs[0,1])
plt.annotate('(b)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples_1row.burnthin(Nb,Nt).funvals.plot_ci(95, plot_par=False, exact=x_exact)
lci[0].set_label("95% CI")
lci[1].set_label("Exact")
lci[2].set_label("Mean")
plt.legend()
plt.ylim([0,0.17])
plt.yticks([0,0.05,0.1,0.15])
plt.xlim([0,1])
plt.xlabel('$x$')
plt.gca().xaxis.set_label_coords(0.5, -0.08)

# 1,2: Case 2, disc CI
plt.sca(axs[0,2])
plt.annotate('(c)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples_1row.burnthin(Nb,Nt).plot_ci(95, plot_par=True, exact=x_exact, markersize=SMALL_SIZE-3)
lci[0].set_label("Mean")
lci[1].set_label("Exact")
lci[2].set_label("95% CI")
plt.legend()
#plt.ylim([0,0.17])
#plt.yticks([0,0.05,0.1,0.15])
#plt.xlim([0,1])
plt.xlabel('$x$')
plt.gca().xaxis.set_label_coords(0.5, -0.08)

# 2,1: Case 10, exact solution, exact data, noisy data
prior_samples_2row, samples_2row, parameters_2row, x_exact, y_exact, data = load_case(case_files[1], load_sol_data=True, load_prior_samples=True)
plt.sca(axs[1,0])
plt.annotate('(d)', xy=(0.03, 0.93), xycoords='axes fraction')
x_exact.plot()
y_exact.plot()
data.plot()
plt.legend(['Exact solution', 'Exact data', 'Noisy data']);
plt.ylim([0,0.17])
plt.yticks([0,0.05,0.1,0.15])
plt.xlim([0,1])
plt.ylabel('$g(x)$')
plt.xlabel('$x$')
plt.gca().xaxis.set_label_coords(0.5, -0.08)

# 1,2: Case 10, cont CI
plt.sca(axs[1,1])
plt.annotate('(e)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples_2row.burnthin(Nb_2row,Nt).funvals.plot_ci(95, plot_par=False, exact=x_exact)
lci[0].set_label("Mean")
lci[1].set_label("Exact")
lci[2].set_label("95% CI")
plt.legend()
plt.ylim([0,0.17])
plt.yticks([0,0.05,0.1,0.15])
plt.xlim([0,1])
plt.xlabel('$x$')
plt.gca().xaxis.set_label_coords(0.5, -0.08)

# 1,2: Case 10, disc CI
plt.sca(axs[1,2])
plt.annotate('(f)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples_2row.burnthin(Nb_2row,Nt).plot_ci(95, plot_par=True, exact=x_exact, markersize=SMALL_SIZE-3)
lci[0].set_label("Mean")
lci[1].set_label("Exact")
lci[2].set_label("95% CI")
plt.legend()
#plt.ylim([0,0.17])
#plt.yticks([0,0.05,0.1,0.15])
#plt.xlim([0,1])
plt.xlabel('$x$')
plt.gca().xaxis.set_label_coords(0.5, -0.08)

# 3,1: Case 13, exact solution, exact data, noisy data
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
plt.ylabel('$g(x)$')
plt.xlabel('$x$')
plt.gca().xaxis.set_label_coords(0.5, -0.08)

# 3,2: Case 13, cont CI
plt.sca(axs[2,1])
plt.annotate('(h)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples_3row.burnthin(Nb_3row,Nt).funvals.plot_ci(95, plot_par=False, exact=x_exact)
lci[0].set_label("Mean")
lci[1].set_label("Exact")
lci[2].set_label("95% CI")
plt.legend()
plt.ylim([0,0.17])
plt.yticks([0,0.05,0.1,0.15])
plt.xlim([0,1])
plt.xlabel('$x$')
plt.gca().xaxis.set_label_coords(0.5, -0.08)

# 3,3: Case 10, disc CI
plt.sca(axs[2,2])
plt.annotate('(i)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples_3row.burnthin(Nb_3row,Nt).plot_ci(95, plot_par=True, exact=x_exact, markersize=SMALL_SIZE -3)
lci[0].set_label("Mean")
lci[1].set_label("Exact")
lci[2].set_label("95% CI")
plt.legend()
#plt.ylim([0,0.17])
#plt.yticks([0,0.05,0.1,0.15])
#plt.xlim([0,1])
plt.xlabel('$x$')
plt.gca().xaxis.set_label_coords(0.5, -0.08)



# 4,1: Case 13, exact solution, exact data, noisy data
prior_samples_4row, samples_4row, parameters_4row, x_exact, y_exact, data = load_case(case_files[3], load_sol_data=True, load_prior_samples=True)
plt.sca(axs[3,0])
plt.annotate('(g)', xy=(0.03, 0.93), xycoords='axes fraction')
x_exact.plot()
y_exact.plot()
data.plot()
plt.legend(['Exact solution', 'Exact data', 'Noisy data']);
plt.ylim([0,0.17])
plt.yticks([0,0.05,0.1,0.15])
plt.xlim([0,1])
plt.ylabel('$g(x)$')
plt.xlabel('$x$')
plt.gca().xaxis.set_label_coords(0.5, -0.08)

# 4,2: Case 13, cont CI
plt.sca(axs[3,1])
plt.annotate('(h)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples_4row.burnthin(Nb_4row,Nt).funvals.plot_ci(95, plot_par=False, exact=x_exact)
lci[0].set_label("Mean")
lci[1].set_label("Exact")
lci[2].set_label("95% CI")
plt.legend()
plt.ylim([0,0.17])
plt.yticks([0,0.05,0.1,0.15])
plt.xlim([0,1])
plt.xlabel('$x$')
plt.gca().xaxis.set_label_coords(0.5, -0.08)

# 4,3: Case 10, disc CI
plt.sca(axs[3,2])
plt.annotate('(i)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples_4row.burnthin(Nb_4row,Nt).plot_ci(95, plot_par=True, exact=x_exact, markersize=SMALL_SIZE -3)
lci[0].set_label("Mean")
lci[1].set_label("Exact")
lci[2].set_label("95% CI")
plt.legend()
#plt.ylim([0,0.17])
#plt.yticks([0,0.05,0.1,0.15])
#plt.xlim([0,1])
plt.xlabel('$x$')
plt.gca().xaxis.set_label_coords(0.5, -0.08)


fig.tight_layout(pad=0, w_pad=0.1, h_pad=0.9)
plt.savefig(fig_file, bbox_inches='tight', pad_inches=0.01, dpi=1200)

#%%
## plot prior posterior samples and ESS
cm_to_in = 1/2.54
fig, axs = plt.subplots(nrows=1, ncols=3,
                        figsize=(17.8*cm_to_in, (17.8/3)*cm_to_in),
                        layout="constrained")

colors = ['C0', 'green', 'purple', 'k', 'gray']
# 1,1: Case 2, prior samples
idx = 0
plt.sca(axs[0])
for s in prior_samples_1row:
    prior_samples_1row.geometry.plot(s, is_par=True, color=colors[idx]) 
    idx += 1
    if idx == 5:
        break

# 1,2: Case 2, posterior samples
idx = 0
plt.sca(axs[1])
for s in samples_1row.burnthin(1000,5000):
    samples_1row.geometry.plot(s, is_par=True, color=colors[idx])   
    idx += 1
    if idx == 5:
        break

# 1,3: Case 2, ESS
plt.rc('lines', markersize=SMALL_SIZE-3) 
plt.sca(axs[2])
plt.plot(parameters_1row["ESS"], 'o-',
         label=str(parameters_1row["Ns"])+' '
         + str(parameters_1row["noise_level"])+' '
               + (str(parameters_1row["decay"]) if "decay" in parameters_1row.keys() else str(parameters_1row['x_exact_geometry'].decay_rate)))

plt.plot(parameters_2row["ESS"], '*-', 
         label=str(parameters_2row["Ns"])+' '
         + str(parameters_2row["noise_level"])+' '
               + (str(parameters_2row["decay"]) if "decay" in parameters_2row.keys() else str(parameters_2row['x_exact_geometry'].decay_rate)), color='green') 

plt.plot(parameters_3row["ESS"], 'd-',
         label=str(parameters_3row["Ns"])+' '
         + str(parameters_3row["noise_level"])+' '
               + (str(parameters_3row["decay"]) if "decay" in parameters_3row.keys() else str(parameters_3row['x_exact_geometry'].decay_rate)), color='k')

plt.plot(parameters_4row["ESS"], 'd-',
         label=str(parameters_4row["Ns"])+' '
         + str(parameters_4row["noise_level"])+' '
               + (str(parameters_4row["decay"]) if "decay" in parameters_4row.keys() else str(parameters_4row['x_exact_geometry'].decay_rate)), color='gray') 

plt.annotate('(c)', xy=(0.03, 0.93), xycoords='axes fraction')
plt.legend()#loc='center right', bbox_to_anchor=(1., 0.27))
plt.ylabel('ESS')
plt.gca().yaxis.set_label_coords(-0.12, 0.45) #-0.12, 0.4
plt.xticks(range(0, parameters_1row["domain_dim"], 2))
#plt.gca().set_xticklabels(['v{}'.format(i) for i in range(c1_parameters["domain_dim"])])
plt.savefig(fig_file_b, bbox_inches='tight', pad_inches=0.01, dpi=1200)
#%%
## plot pair 
cm_to_in = 1/2.54
fig, axs = plt.subplots(nrows=3, ncols=3,
                        figsize=(17.8*cm_to_in, 17.8*cm_to_in),
                        layout="constrained")
samples_3row.burnthin(Nb).plot_pair([0,1,2,15], ax=axs)
for ax in axs.flat:
    ax.set_rasterized(True)
fig.tight_layout(pad=0, w_pad=0.1, h_pad=0.9)
plt.savefig(fig_file_c, bbox_inches='tight', pad_inches=0.01, dpi=1200)


#%%

#%% plot trace
cm_to_in = 1/2.54
fig, axs = plt.subplots(nrows=4, ncols=2,
                        figsize=(17.8*cm_to_in, 12*cm_to_in),
                        layout="constrained")
samples_1row.plot_trace([0,1,5,15], axes=axs)
fig.tight_layout(pad=0, w_pad=0.1, h_pad=0.9)
plt.savefig(fig_file_d, bbox_inches='tight', pad_inches=0.01, dpi=1200)
# %%
import numpy as np
for key in parameters_1row:
    if np.any(parameters_1row[key] != parameters_2row[key]) or\
        np.any(parameters_1row[key] != parameters_3row[key]) or\
            np.any(parameters_1row[key] != parameters_4row[key]):	
    
        print("###",key,"###")
        print(parameters_1row[key], parameters_2row[key], parameters_3row[key], parameters_4row[key])

# %%
