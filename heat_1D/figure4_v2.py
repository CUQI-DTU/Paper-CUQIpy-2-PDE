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

fig_file = fig_dir + 'paper_figure4.pdf'
fig_file_b = fig_dir + 'paper_figure4_b.pdf'
fig_file_c = fig_dir + 'paper_figure4_c.pdf'
fig_file_d = fig_dir + 'paper_figure4_d.pdf'


# %% Burnthin
Nb = 100
Nt = None
# %% Create the figure
cm_to_in = 1/2.54
fig, axs = plt.subplots(nrows=3, ncols=3,
                        figsize=(17.8*cm_to_in, 13.5*cm_to_in),
                        layout="constrained")

# 1,1: Case 2, exact solution, exact data, noisy data
prior_samples2, _, _, x_exact, y_exact, data = load_case('./data2_cont3/paper_case2', load_sol_data=True, load_prior_samples=True)
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
_, samples2, parameters2, _, _, _ = load_case('./data2/paper_case2', load_sol_data=False, load_prior_samples=False)
plt.sca(axs[0,1])
plt.annotate('(b)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples2.burnthin(Nb,Nt).funvals.plot_ci(95, plot_par=False, exact=x_exact)
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
lci = samples2.burnthin(Nb,Nt).plot_ci(95, plot_par=True, exact=x_exact, markersize=SMALL_SIZE-3)
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
prior_samples10, _, _, x_exact, y_exact, data = load_case('./data2_cont3/paper_case10', load_sol_data=True, load_prior_samples=True)
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
_, samples10, parameters10, _, _, _ = load_case('./data2_cont2/paper_case10', load_sol_data=False)
plt.sca(axs[1,1])
plt.annotate('(e)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples10.burnthin(Nb,Nt).funvals.plot_ci(95, plot_par=False, exact=x_exact)
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
lci = samples10.burnthin(Nb,Nt).plot_ci(95, plot_par=True, exact=x_exact, markersize=SMALL_SIZE-3)
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
prior_samples13, _, _, x_exact, y_exact, data = load_case('./data2_cont3/paper_case13', load_sol_data=True, load_prior_samples=True)
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

# 1,2: Case 10, cont CI
_, samples13, parameters13, _, _, _ = load_case('./data2_cont2/paper_case13', load_sol_data=False)
plt.sca(axs[2,1])
plt.annotate('(h)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples13.burnthin(Nb,Nt).funvals.plot_ci(95, plot_par=False, exact=x_exact)
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
plt.sca(axs[2,2])
plt.annotate('(i)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples13.burnthin(Nb,Nt).plot_ci(95, plot_par=True, exact=x_exact, markersize=SMALL_SIZE -3)
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
for s in prior_samples2:
    prior_samples2.geometry.plot(s, is_par=True, color=colors[idx]) 
    idx += 1
    if idx == 5:
        break

# 1,2: Case 2, posterior samples
idx = 0
plt.sca(axs[1])
for s in samples2.burnthin(1000,5000):
    samples2.geometry.plot(s, is_par=True, color=colors[idx])   
    idx += 1
    if idx == 5:
        break

# 1,3: Case 2, ESS
plt.rc('lines', markersize=SMALL_SIZE-3) 
plt.sca(axs[2])
plt.plot(parameters2["ESS"], 'o-', label='$1\%$ noise') 
plt.plot(parameters10["ESS"], '*-', label='$5\%$ noise', color='green') 
plt.plot(parameters13["ESS"], 'd-', label='partial data', color='k') 
plt.annotate('(c)', xy=(0.03, 0.93), xycoords='axes fraction')
plt.legend()#loc='center right', bbox_to_anchor=(1., 0.27))
plt.ylabel('ESS')
plt.gca().yaxis.set_label_coords(-0.12, 0.45) #-0.12, 0.4
plt.xticks(range(0, parameters2["domain_dim"], 2))
#plt.gca().set_xticklabels(['v{}'.format(i) for i in range(c1_parameters["domain_dim"])])
plt.savefig(fig_file_b, bbox_inches='tight', pad_inches=0.01, dpi=1200)
#%%
## plot pair 
cm_to_in = 1/2.54
fig, axs = plt.subplots(nrows=3, ncols=3,
                        figsize=(17.8*cm_to_in, 17.8*cm_to_in),
                        layout="constrained")
samples13.burnthin(Nb).plot_pair([0,1,2,15], ax=axs)
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
samples2.plot_trace([0,1,5,15], axes=axs)
fig.tight_layout(pad=0, w_pad=0.1, h_pad=0.9)
plt.savefig(fig_file_d, bbox_inches='tight', pad_inches=0.01, dpi=1200)
# %%
