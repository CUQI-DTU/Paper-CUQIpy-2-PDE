#%%
import pickle
import matplotlib
import matplotlib.pyplot as plt
import os

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
data_dir = './data2/'

# Figure file
fig_dir = data_dir + 'figs/'
# Check if the directory exists
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

fig_file = fig_dir + 'paper_figure1.pdf'

# Cases:
c1_dir = data_dir + 'paper_case1/'
c2_dir = data_dir + 'paper_case2/'

# Load data
c1_samples = pickle.load(open(c1_dir + 'samples.pkl', 'rb'))
c1_parameters = pickle.load(open(c1_dir + 'parameters.pkl', 'rb'))

c2_samples = pickle.load(open(c2_dir + 'samples.pkl', 'rb'))
c2_parameters = pickle.load(open(c2_dir + 'parameters.pkl', 'rb'))

# %% Burnthin
Nb = 100
Nt = 1000
# %% Create the figure
cm_to_in = 1/2.54
fig, axs = plt.subplots(nrows=2, ncols=3,
                        figsize=(17.8*cm_to_in, 9*cm_to_in),
                        layout="constrained")

# 1,1: Case 1, exact solution, exact data, noisy data
plt.sca(axs[0,0])
plt.annotate('(a)', xy=(0.03, 0.93), xycoords='axes fraction')
#x_exact.plot()
#y_exact.plot()
#data.plot()
#plt.legend(['Exact solution', 'Exact data', 'Noisy data']);
#plt.ylim([0,0.17])
#plt.yticks([0,0.05,0.1,0.15])
#plt.xlim([0,1])
#plt.ylabel('$g(x)$')
#plt.xlabel('$x$')
#plt.gca().xaxis.set_label_coords(0.5, -0.08)



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
plt.annotate('(f)', xy=(0.03, 0.93), xycoords='axes fraction')
lci = c2_samples.burnthin(Nb, Nt).plot_ci(95, plot_par=True)
lci[0].set_label("95% CI")
#lci[1].set_label("Exact")
lci[2].set_label("Mean")
plt.legend()


fig.tight_layout(pad=0, w_pad=0.1, h_pad=0.9)
plt.savefig(fig_file, bbox_inches='tight', pad_inches=0.01, dpi=1200)


