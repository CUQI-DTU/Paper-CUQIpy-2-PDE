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


# %% Burnthin
Nb = 100
Nt = None
# %% Create the figure
cm_to_in = 1/2.54
fig, axs = plt.subplots(nrows=3, ncols=3,
                        figsize=(17.8*cm_to_in, 13.5*cm_to_in),
                        layout="constrained")

# 1,1: Case 2, exact solution, exact data, noisy data
_, _, _, x_exact, y_exact, data = load_case('./data2_cont3/paper_case2', load_sol_data=True, load_samples=False)
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
_, samples, parameters, _, _, _ = load_case('./data2/paper_case2', load_sol_data=False, load_samples=False)
plt.sca(axs[0,1])
plt.annotate('(b)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples.burnthin(Nb,Nt).funvals.plot_ci(95, plot_par=False, exact=x_exact)
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
lci = samples.burnthin(Nb,Nt).plot_ci(95, plot_par=True)#, exact=x_exact)
lci[0].set_label("95% CI")
#lci[1].set_label("Exact")
lci[2].set_label("Mean")
plt.legend()
#plt.ylim([0,0.17])
#plt.yticks([0,0.05,0.1,0.15])
#plt.xlim([0,1])
plt.xlabel('$x$')
plt.gca().xaxis.set_label_coords(0.5, -0.08)

# 2,1: Case 10, exact solution, exact data, noisy data
_, _, _, x_exact, y_exact, data = load_case('./data2_cont3/paper_case10', load_sol_data=True)
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
_, samples, parameters, _, _, _ = load_case('./data2_cont2/paper_case10', load_sol_data=False)
plt.sca(axs[1,1])
plt.annotate('(e)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples.burnthin(Nb,Nt).funvals.plot_ci(95, plot_par=False, exact=x_exact)
lci[0].set_label("95% CI")
lci[1].set_label("Exact")
lci[2].set_label("Mean")
plt.legend()
plt.ylim([0,0.17])
plt.yticks([0,0.05,0.1,0.15])
plt.xlim([0,1])
plt.xlabel('$x$')
plt.gca().xaxis.set_label_coords(0.5, -0.08)

# 1,2: Case 10, disc CI
plt.sca(axs[1,2])
plt.annotate('(f)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples.burnthin(Nb,Nt).plot_ci(95, plot_par=True)#, exact=x_exact)
lci[0].set_label("95% CI")
#lci[1].set_label("Exact")
lci[2].set_label("Mean")
plt.legend()
#plt.ylim([0,0.17])
#plt.yticks([0,0.05,0.1,0.15])
#plt.xlim([0,1])
plt.xlabel('$x$')
plt.gca().xaxis.set_label_coords(0.5, -0.08)

# 3,1: Case 13, exact solution, exact data, noisy data
_, _, _, x_exact, y_exact, data = load_case('./data2_cont3/paper_case13', load_sol_data=True)
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
_, samples, parameters, _, _, _ = load_case('./data2_cont2/paper_case13', load_sol_data=False)
plt.sca(axs[2,1])
plt.annotate('(h)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples.burnthin(Nb,Nt).funvals.plot_ci(95, plot_par=False, exact=x_exact)
lci[0].set_label("95% CI")
lci[1].set_label("Exact")
lci[2].set_label("Mean")
plt.legend()
plt.ylim([0,0.17])
plt.yticks([0,0.05,0.1,0.15])
plt.xlim([0,1])
plt.xlabel('$x$')
plt.gca().xaxis.set_label_coords(0.5, -0.08)

# 1,2: Case 10, disc CI
plt.sca(axs[2,2])
plt.annotate('(i)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = samples.burnthin(Nb,Nt).plot_ci(95, plot_par=True)#, exact=x_exact)
lci[0].set_label("95% CI")
#lci[1].set_label("Exact")
lci[2].set_label("Mean")
plt.legend()
#plt.ylim([0,0.17])
#plt.yticks([0,0.05,0.1,0.15])
#plt.xlim([0,1])
plt.xlabel('$x$')
plt.gca().xaxis.set_label_coords(0.5, -0.08)


fig.tight_layout(pad=0, w_pad=0.1, h_pad=0.9)
plt.savefig(fig_file, bbox_inches='tight', pad_inches=0.01, dpi=1200)
"""

# 1,2: Case 1, cont CI
plt.sca(axs[0,1])
plt.annotate('(b)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = c1_samples.burnthin(Nb,Nt).plot_ci(95, plot_par=False)#, exact=x_exact)
lci[0].set_label("95% CI")
#lci[1].set_label("Exact")
lci[2].set_label("Mean")
plt.legend()
plt.ylim([0,0.17])
plt.yticks([0,0.05,0.1,0.15])
plt.xlim([0,1])
plt.xlabel('$x$')
plt.gca().xaxis.set_label_coords(0.5, -0.08)


# 1,3: Case 2, cont CI
plt.sca(axs[0,2])
plt.annotate('(c)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = c2_samples.burnthin(Nb, Nt).plot_ci(95, plot_par=False)#, exact=x_exact)
lci[0].set_label("95% CI")
#lci[1].set_label("Exact")
lci[2].set_label("Mean")
plt.legend()
plt.ylim([0,0.17])
plt.yticks([0,0.05,0.1,0.15])
plt.xlim([0,1])
plt.xlabel('$x$')
plt.gca().xaxis.set_label_coords(0.5, -0.08)


# 2,1: Case 1&2, ESS
plt.rc('lines', markersize=SMALL_SIZE-2) 
plt.sca(axs[1,0])
plt.semilogy(c1_parameters["ESS"], '-', label='Continuous') #CWMH
plt.semilogy(c2_parameters["ESS"], '-', label='KL', color='green') #PCN
plt.annotate('(d)', xy=(0.03, 0.93), xycoords='axes fraction')
plt.legend()#loc='center right', bbox_to_anchor=(1., 0.27))
plt.ylabel('ESS')
plt.gca().yaxis.set_label_coords(-0.12, 0.45) #-0.12, 0.4
plt.xticks(range(0, c1_parameters["domain_dim"], 10))
#plt.gca().set_xticklabels(['v{}'.format(i) for i in range(c1_parameters["domain_dim"])])


# 2,2: Case 2, funval CI
plt.sca(axs[1,1])
plt.annotate('(e)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = c2_samples.burnthin(Nb, Nt).funvals.plot_ci(95)#, plot_par=True, exact=x_exact)
lci[0].set_label("95% CI")
#lci[1].set_label("Exact")
lci[2].set_label("Mean")
plt.legend(loc='lower left')

# 2,3: Case 2, Disc CI
plt.sca(axs[1,2])
c1_samples.burnthin(Nb, Nt).plot_trace([0], axes = axs[1:2,1:])



fig.tight_layout(pad=0, w_pad=0.1, h_pad=0.9)
plt.savefig(fig_file, bbox_inches='tight', pad_inches=0.01, dpi=1200)


"""

