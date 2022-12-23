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

fig_file = fig_dir + 'paper_figure1_v3.pdf'



# %% Burnthin
Nb = 100
Nt = None
# %% Create the figure
cm_to_in = 1/2.54
fig, axs = plt.subplots(nrows=3, ncols=3,
                        figsize=(17.8*cm_to_in, 13.5*cm_to_in),
                        layout="constrained")

# 1,1: Case 3, exact solution, exact data, noisy data
prior_samples3, samples3, parameters3, x_exact, y_exact, data = load_case('./data2_cont3/paper_case3', load_sol_data=True, load_prior_samples=True)
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

# 1,2: Case 3, cont CI
plt.sca(axs[0,1])
plt.annotate('(b)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples3.burnthin(Nb,Nt).funvals.plot_ci(95, plot_par=False, exact=x_exact)
lci[0].set_label("95% CI")
lci[1].set_label("Exact")
lci[2].set_label("Mean")
plt.legend()
plt.ylim([0,0.17])
plt.yticks([0,0.05,0.1,0.15])
plt.xlim([0,1])
plt.xlabel('$x$')
plt.gca().xaxis.set_label_coords(0.5, -0.08)

# 1,3: Case 3, disc CI
plt.sca(axs[0,2])
plt.annotate('(c)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples3.burnthin(Nb,Nt).plot_ci(95, plot_par=True, exact=x_exact)
lci[0].set_label("95% CI")
lci[1].set_label("Exact")
lci[2].set_label("Mean")
plt.legend()
#plt.ylim([0,0.17])
#plt.yticks([0,0.05,0.1,0.15])
#plt.xlim([0,1])
plt.xlabel('$x$')
plt.gca().xaxis.set_label_coords(0.5, -0.08)


# 2,1: Case 3_b, exact solution, exact data, noisy data
prior_samples3_c, samples3_c, parameters3_c, x_exact, y_exact, data = load_case('./data2_cont5/paper_case3_c', load_sol_data=True, load_prior_samples=True)
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

# 2,2: Case 3_b, cont CI
plt.sca(axs[1,1])
plt.annotate('(e)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples3_c.burnthin(Nb,Nt).funvals.plot_ci(95, plot_par=False, exact=x_exact)
lci[0].set_label("95% CI")
lci[1].set_label("Exact")
lci[2].set_label("Mean")
plt.legend()
plt.ylim([0,0.17])
plt.yticks([0,0.05,0.1,0.15])
plt.xlim([0,1])
plt.xlabel('$x$')
plt.gca().xaxis.set_label_coords(0.5, -0.08)

# 2,3: Case 3_b, disc CI
plt.sca(axs[1,2])
plt.annotate('(f)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples3_c.burnthin(Nb,Nt).plot_ci(95, plot_par=True, exact=x_exact)
lci[0].set_label("95% CI")
lci[1].set_label("Exact")
lci[2].set_label("Mean")
plt.legend()
#plt.ylim([0,0.17])
#plt.yticks([0,0.05,0.1,0.15])
#plt.xlim([0,1])
plt.xlabel('$x$')
plt.gca().xaxis.set_label_coords(0.5, -0.08)

## plot prior posterior samples and ESS
colors = ['C0', 'green', 'purple', 'k', 'gray']
# 3,1: prior samples
idx = 0
plt.sca(axs[2,0])
for s in prior_samples3:
    prior_samples3.geometry.plot(s, is_par=True, color=colors[idx]) 
    idx += 1
    if idx == 5:
        break

# 3,2: posterior samples
idx = 0
plt.sca(axs[2,1])
for s in samples3.burnthin(1000,1000):
    samples3.geometry.plot(s, is_par=True, color=colors[idx])   
    idx += 1
    if idx == 5:
        break

# 3,3:  ESS
plt.rc('lines', markersize=SMALL_SIZE-3) 
plt.sca(axs[2,2])
plt.plot(parameters3["ESS"], 'o-', label='$1\%$ noise') 
plt.plot(parameters3_c["ESS"], '*-', label='$5\%$ noise', color='green') 
plt.annotate('(i)', xy=(0.03, 0.93), xycoords='axes fraction')
plt.legend()#loc='center right', bbox_to_anchor=(1., 0.27))
plt.ylabel('ESS')
plt.gca().yaxis.set_label_coords(-0.12, 0.45) #-0.12, 0.4
plt.xticks(range(0, parameters3["domain_dim"], 2))
#plt.gca().set_xticklabels(['v{}'.format(i) for i in range(c1_parameters["domain_dim"])])


fig.tight_layout(pad=0, w_pad=0.1, h_pad=0.9)
plt.savefig(fig_file, bbox_inches='tight', pad_inches=0.01, dpi=1200)



