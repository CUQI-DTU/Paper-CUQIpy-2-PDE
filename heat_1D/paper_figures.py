#%%
import matplotlib.pyplot as plt
import os
from figures_util import matplotlib_setup
import arviz as az
az.style.use('default')
import numpy as np
from cuqi.pde import TimeDependentLinearPDE
from cuqi.geometry import StepExpansion, Continuous1D
from cuqi.distribution import Gaussian, JointDistribution
import cuqi

# method to plot figure 2 for the Heat 1D example
def plot_figure2(fig_dir, version,
                 g_custom, u_custom, y_custom, u_intermediate,
                 grid, tau, intermediate_times):
    # Matplotlib setup
    SMALL_SIZE = 7
    MEDIUM_SIZE =8
    BIGGER_SIZE = 9
    matplotlib_setup(SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE)
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{bm}')
    
    # Check if the directory exists
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    # Figure file
    fig_file = fig_dir + 'paper_demo_PDE_'+version+'.pdf'

    # Create figure        
    cm_to_in = 1/2.54
    fig, axs = plt.subplots(nrows=1, ncols=3,
                            figsize=(17.8*cm_to_in, (17.8/4)*cm_to_in),
                            layout="constrained")
    
    colors = ['C0', 'green', 'purple', 'k', 'gray']
    # a) plot g_custom
    plt.sca(axs[0])
    plt.annotate('(a)', xy=(0.03, 0.93), xycoords='axes fraction')
    plt.plot(grid, g_custom)
    plt.ylabel('$\\bm{g}^\\mathrm{custom}$')
    plt.gca().yaxis.set_label_coords(-0.21, 0.45)
    
    plt.xlabel('$\\xi$')
    plt.gca().xaxis.set_label_coords(.5, -.12)
    plt.gca().set_ylim(0, .13)
    plt.xlim([0,1])
    
    # b) plot u_custom
    plt.sca(axs[1])
    plt.annotate('(b)', xy=(0.03, 0.93), xycoords='axes fraction')
    
    number_of_time_steps_to_plot = u_intermediate.shape[1]
    for i in range(number_of_time_steps_to_plot):
        if i==number_of_time_steps_to_plot-1:
            plt.plot(
                grid, u_intermediate[:, i], color='#1f77b4', linestyle='-')
        else:
            plt.plot(
                grid, u_intermediate[:, i], color=colors[i], linestyle='--')

    plt.legend(['${:.2g}$'.format(t) for t in intermediate_times],
               loc='upper right', ncol=2, frameon=False,
               bbox_to_anchor=(1, 0.95))
    
    plt.ylabel('$\\bm{u}^\\mathrm{custom}$')
    plt.gca().yaxis.set_label_coords(-0.21, 0.45) 
    
    plt.xlabel('$\\xi$')
    plt.gca().xaxis.set_label_coords(.5, -.12) 
    plt.gca().set_ylim(0, .13)
    plt.xlim([0,1])
    
    # c) plot y_custom
    plt.sca(axs[2])
    plt.annotate('(c)', xy=(0.03, 0.93), xycoords='axes fraction')
    plt.plot(grid, y_custom)
    plt.ylabel('$\\bm{y}^\\mathrm{custom}$')
    plt.gca().yaxis.set_label_coords(-0.21, 0.45) 
    
    plt.xlabel('$\\xi$')
    plt.gca().xaxis.set_label_coords(.5, -.12) 
    plt.gca().set_ylim(0, .13)
    plt.xlim([0,1])
    
    # Save figure
    fig.tight_layout(pad=0, w_pad=0.25, h_pad=0)
    plt.savefig(fig_file, bbox_inches='tight', pad_inches=0.01, dpi=1200)


# method to plot figure 3 for the Heat 1D example
def plot_figure3(fig_dir, version, x_step, y_step):

    # Matplotlib setup
    SMALL_SIZE = 7
    MEDIUM_SIZE =8
    BIGGER_SIZE = 9
    matplotlib_setup(SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE)
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{bm}')
    
    # Check if the directory exists
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir) 
    
    # Create figure
    cm_to_in = 1/2.54
    fig, axs = plt.subplots(nrows=1, ncols=3,
                            figsize=(17.8*cm_to_in, (17.8/4)*cm_to_in),
                            layout="constrained")
    
    # Figure file
    fig_file = fig_dir + 'paper_demo2_PDE_'+version+'.pdf'
    
    # 1: plot x_step (function representation)
    plt.sca(axs[0])
    plt.annotate('(a)', xy=(0.03, 0.93), xycoords='axes fraction')
    x_step.plot()
    plt.ylabel('$\\bm{g}^{\\mathrm{step}}$')
    plt.gca().yaxis.set_label_coords(-0.15, 0.45)
    
    plt.xlabel('$\\xi$')
    plt.gca().xaxis.set_label_coords(.5, -.12)
    plt.ylim([-0.2,1.2])
    plt.xlim([0,1])
    
    # 2: plot x_step (parameter representation)
    plt.sca(axs[1])
    plt.annotate('(b)', xy=(0.03, 0.93), xycoords='axes fraction')
    x_step.plot(plot_par=True)
    
    plt.ylabel('$x^{\\mathrm{step}}_i$')
    plt.gca().yaxis.set_label_coords(-0.15, 0.45)
    
    plt.xlabel('$i$')
    plt.gca().xaxis.set_label_coords(.5, -.12)
    plt.ylim([-0.2,1.2])
    tick_ids = np.linspace(0, 2, 3, dtype=int)
    plt.xticks(tick_ids, tick_ids)

    # 3: Plot y_step
    plt.sca(axs[2])
    plt.annotate('(c)', xy=(0.03, 0.93), xycoords='axes fraction')
    y_step.plot()
    plt.ylabel('$\\bm{y}^{\\mathrm{step}}$')
    plt.gca().yaxis.set_label_coords(-0.15, 0.45)
    
    plt.xlabel('$\\xi$')
    plt.gca().xaxis.set_label_coords(.5, -.12)
    plt.ylim([-0.2,1.2])
    plt.xlim([0,1])

    # Save figure
    fig.tight_layout(pad=0, w_pad=0.25, h_pad=0)
    plt.savefig(fig_file, bbox_inches='tight', pad_inches=0.01, dpi=1200)    


# method to plot figure 4 for the Heat 1D example
def plot_figure4(fig_dir, version, G_step, 
                 prior_samples, posterior_samples,
                 x_step, y_step, y_obs): 
    
    # Matplotlib setup 
    SMALL_SIZE = 7
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 9
    matplotlib_setup(SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE)
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{bm}')

    Ns = posterior_samples.Ns # number of samples
    Nb = int(1/250*Ns) # burn-in
    Nt = None # thinning
    num_var = 3 # number of variables
    n_ticks = 5 # number of ticks 

    # Check if the directory exists
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir) 

    fig_file = fig_dir + 'paper_figure4_'+version+'.pdf'

    # Create the figure
    cm_to_in = 1/2.54
    fig, axs = plt.subplots(nrows=2, ncols=3,
                            figsize=(17.8*cm_to_in, 2/3*13.5*cm_to_in),
                            layout="constrained")

    # potting colors
    colors = ['C0', 'green', 'purple', 'k', 'gray']
    
    # (a)  Basis functions
    plt.rc('lines', markersize=SMALL_SIZE-3) 
    plt.sca(axs[0,0])
    
    a1 = cuqi.array.CUQIarray([1,0,0], geometry=G_step)
    a2 = cuqi.array.CUQIarray([0,1,0], geometry=G_step)
    a3 = cuqi.array.CUQIarray([0,0,1], geometry=G_step)
    
    a1.plot( label='$\\bm{\\chi}_1$', linestyle='--')
    a2.plot( label='$\\bm{\\chi}_2$', linestyle='--')
    a3.plot( label='$\\bm{\\chi}_3$', linestyle='--')
    
    plt.annotate('(a)', xy=(0.03, 0.93), xycoords='axes fraction')
    plt.legend(frameon=False)
    plt.ylim([-0.2, 1.2])
    plt.gca().yaxis.set_label_coords(-0.18, 0.5) 
    
    plt.xlabel('$\\xi$')
    plt.gca().xaxis.set_label_coords(.53, -.12)

    # (b) prior samples
    idx = 0
    plt.sca(axs[0,1])
    prior_samples.plot(range(5))
    plt.ylabel('$\\bm{g}$')
    plt.gca().yaxis.set_label_coords(-0.15, 0.5) 
    plt.xlabel('$\\xi$')
    plt.gca().xaxis.set_label_coords(.5, -.12)
    plt.annotate('(b)', xy=(0.03, 0.93), xycoords='axes fraction')

    # (c) posterior samples
    idx = 0
    plt.sca(axs[0,2])
    posterior_samples.burnthin(int(1/50*Ns), int(1/50*Ns)).plot(range(5))
    
    plt.ylabel('$\\bm{g}$')
    plt.gca().yaxis.set_label_coords(-0.15, 0.5)
    
    plt.xlabel('$\\xi$')
    plt.gca().xaxis.set_label_coords(.5, -.12) 
    plt.annotate('(c)', xy=(0.03, 0.93), xycoords='axes fraction')

    # (d) Exact solution, exact data, noisy data
    plt.sca(axs[1,0])
    plt.annotate('(d)', xy=(0.03, 0.93), xycoords='axes fraction')
    x_step.plot()
    y_step.plot()
    y_obs.plot()
    plt.legend(['Exact solution', 'Exact data', 'Noisy data'],
               bbox_to_anchor=(.629, 0), loc='lower center', ncol=1);
    plt.ylim([-0.2,1.2])
    plt.yticks([0,0.25,0.5,0.75, 1])
    plt.xlim([0,1])
    plt.gca().yaxis.set_label_coords(-0.18, 0.5) 
    plt.xlabel('$\\xi$')
    plt.gca().xaxis.set_label_coords(.5, -.12) 

    # (e) continuous CI
    plt.sca(axs[1,1])
    plt.annotate('(e)', xy=(0.03, 0.93), xycoords='axes fraction')
    lci = posterior_samples.burnthin(Nb,Nt).funvals.plot_ci(95,
                                                            plot_par=False,
                                                            exact=x_step)
    lci[0].set_label("Mean")
    lci[1].set_label("Exact")
    lci[2].set_label('$95\\%\; \\mathrm{CI}$')
    plt.legend()
    plt.ylim([-0.2,1.2])
    plt.yticks([0,0.25,0.5,0.75, 1])
    plt.xlim([0,1])
    plt.ylabel('$\\bm{g}$')
    plt.gca().yaxis.set_label_coords(-0.18, 0.5)
    plt.xlabel('$\\xi$')
    plt.gca().xaxis.set_label_coords(.5, -.12)

    # (f) discrete CI
    plt.sca(axs[1,2])
    plt.annotate('(f)', xy=(0.03, 0.93), xycoords='axes fraction')
    lci = posterior_samples.burnthin(Nb,Nt).plot_ci(95, plot_par=True,
                                                    exact=x_step,
                                                    markersize=SMALL_SIZE-3)
    lci[0].set_label("Mean")
    lci[1].set_label("Exact")
    lci[2].set_label("$95\\%\; \\mathrm{CI}$")
    plt.legend()
    plt.ylabel('$x_i$')
    plt.gca().yaxis.set_label_coords(-0.15, 0.5)
    plt.xlabel('$i$')
    plt.gca().xaxis.set_label_coords(.53, -.12)
    tick_ids = np.linspace(0, num_var-1, n_ticks, dtype=int)
    plt.xticks(tick_ids, tick_ids)
    plt.legend(ncols=1, loc ="upper right")
    
    # save figure
    fig.tight_layout(pad=0, w_pad=0.1, h_pad=0.9)
    plt.savefig(fig_file, bbox_inches='tight', pad_inches=0.01, dpi=1200)

#  method to plot figure 5 for the Heat 1D example
def plot_figure5(fig_dir, version, G_KL, 
                 prior_samples, 
                 case1_data,
                 case2_data,
                 case3_data,
                ): 
    #case data contains:  exact, y, y_obs, posterior_samples
    #
    # Matplotlib setup
    SMALL_SIZE = 7
    MEDIUM_SIZE =8
    BIGGER_SIZE = 9

    Ns = case1_data[-1].Ns # number of samples
        
    matplotlib_setup(7, 8, 9)

    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{bm}')
    
    # Check if the directory exists
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    fig_file = fig_dir +   'paper_figure5_'+version+'.pdf'

    Nb = int(0.5*Ns) # burn-in
    Nt =  None # thinning
    Nb_3row = int(1/5*Ns) # burn-in for 3-row plots
    Nb_4row = int(1/5*Ns) # burn-in for 4-row plots
    n_ticks = 5
    num_var = 20
    
    # Create the figure
    cm_to_in = 1/2.54
    fig, axs = plt.subplots(nrows=4, ncols=3,
                            figsize=(17.8*cm_to_in, 17.8*cm_to_in),
                            layout="constrained")
    
    # color array
    colors = ['C0', 'green', 'purple', 'k', 'gray']

    # (a)
    plt.sca(axs[0,0])
    geo = G_KL 
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
    
    plt.annotate('(a)', xy=(0.03, 0.93), xycoords='axes fraction')
    plt.legend(frameon=False, ncols=2)
    plt.ylim([-0.05, 0.15])
    plt.gca().yaxis.set_label_coords(-0.18, 0.5)
    
    plt.xlim([0, 1])
    plt.xlabel('$\\xi$')
    plt.gca().xaxis.set_label_coords(.53, -.12)
    
    
    # (b)  
    idx = 0
    plt.sca(axs[0,1])
    plt.annotate('(b)', xy=(0.03, 0.93), xycoords='axes fraction')
    prior_samples.plot(range(5))
    plt.ylabel('$\\bm{g}$')
    plt.gca().yaxis.set_label_coords(-0.16, 0.5)
    
    plt.xlabel('$\\xi$')
    plt.gca().xaxis.set_label_coords(.5, -.12) #-0.12, 0.4
    plt.gca().set_ylim(-.175, .175)
    plt.xlim([0,1])
    
    
    # (c)
    idx = 0
    plt.sca(axs[0,2])
    plt.annotate('(c)', xy=(0.03, 0.93), xycoords='axes fraction')
    case2_data[-1].burnthin(int(1/50*Ns), int(1/10*Ns)).plot(range(5))
    plt.ylabel('$\\bm{g}$')
    plt.gca().yaxis.set_label_coords(-0.16, 0.5)
    plt.xlabel('$\\xi$')
    plt.gca().xaxis.set_label_coords(.5, -.12) 
    plt.gca().set_ylim(-.015, .17)
    plt.xlim([0,1])
    

    # (d)
    plt.sca(axs[1,0])
    plt.annotate('(d)', xy=(0.03, 0.93), xycoords='axes fraction')
    case1_data[0].plot()
    case1_data[1].plot()
    case1_data[2].plot()
    plt.legend(['Exact solution', 'Exact data', 'Noisy data']);
    plt.ylim([0,0.17])
    plt.yticks([0,0.05,0.1,0.15])
    plt.xlim([0,1])
    plt.gca().yaxis.set_label_coords(-0.16, 0.5)
    plt.xlabel('$\\xi$')
    plt.gca().xaxis.set_label_coords(0.5, -0.14)
    
    # (e)
    plt.sca(axs[1,1])
    plt.annotate('(e)', xy=(0.03, 0.93), xycoords='axes fraction')
    lci = case1_data[-1].burnthin(Nb, Nt).funvals.plot_ci(
        95, plot_par=False, exact=case1_data[0])
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
    
    # (f)
    plt.sca(axs[1,2])
    plt.annotate('(f)', xy=(0.03, 0.93), xycoords='axes fraction')
    lci = case1_data[-1].burnthin(Nb, Nt).plot_ci(
        95, plot_par=True, exact=case1_data[0], markersize=SMALL_SIZE-3)
    lci[0].set_label("Mean")
    lci[1].set_label("Exact")
    lci[2].set_label("$95\\%\;\mathrm{CI}$")
    plt.legend(loc = 'lower right')
    plt.ylabel('$x_i$')
    plt.gca().yaxis.set_label_coords(-0.09, 0.5)
    plt.xlabel('$i$')
    plt.gca().xaxis.set_label_coords(0.5, -0.14)
    tick_ids = np.linspace(0, num_var-1, n_ticks, dtype=int)
    plt.xticks(tick_ids, tick_ids)
    
    
    # (g)
    plt.sca(axs[2,0])
    plt.annotate('(g)', xy=(0.03, 0.93), xycoords='axes fraction')
    case2_data[0].plot()
    case2_data[1].plot()
    case2_data[2].plot()
    plt.legend(['Exact solution', 'Exact data', 'Noisy data']);
    plt.ylim([0,0.17])
    plt.yticks([0,0.05,0.1,0.15])
    plt.xlim([0,1])
    plt.gca().yaxis.set_label_coords(-0.16, 0.5)
    plt.xlabel('$\\xi$')
    plt.gca().xaxis.set_label_coords(0.5, -0.14)
    
    # (h)
    plt.sca(axs[2,1])
    plt.annotate('(h)', xy=(0.03, 0.93), xycoords='axes fraction')
    lci = case2_data[-1].burnthin(Nb_3row, Nt).funvals.plot_ci(
        95, plot_par=False, exact=case2_data[0])
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
    
    # (i)
    plt.sca(axs[2,2])
    plt.annotate('(i)', xy=(0.03, 0.93), xycoords='axes fraction')
    lci = case2_data[-1].burnthin(Nb_3row, Nt).plot_ci(
        95, plot_par=True, exact=case2_data[0], markersize=SMALL_SIZE -3)
    lci[0].set_label("Mean")
    lci[1].set_label("Exact")
    lci[2].set_label("$95\\%\;\mathrm{CI}$")
    plt.legend(loc = 'lower right')
    plt.ylabel('$x_i$')
    plt.gca().yaxis.set_label_coords(-0.09, 0.5)
    plt.xlabel('$i$')
    plt.gca().xaxis.set_label_coords(0.5, -0.14)
    tick_ids = np.linspace(0, num_var-1, n_ticks, dtype=int)
    plt.xticks(tick_ids, tick_ids)
    

    # (j)
    plt.sca(axs[3,0])
    plt.annotate('(j)', xy=(0.03, 0.93), xycoords='axes fraction')
    case3_data[0].plot()
    case3_data[1].plot()
    case3_data[2].plot()
    plt.legend(['Exact solution', 'Exact data', 'Noisy data']);
    plt.ylim([0,0.17])
    plt.yticks([0,0.05,0.1,0.15])
    plt.xlim([0,1])
    plt.gca().yaxis.set_label_coords(-0.16, 0.5)
    plt.xlabel('$\\xi$')
    plt.gca().xaxis.set_label_coords(0.5, -0.14)
    
    # (k)
    plt.sca(axs[3,1])
    plt.annotate('(k)', xy=(0.03, 0.93), xycoords='axes fraction')
    lci = case3_data[-1].burnthin(Nb_4row, Nt).funvals.plot_ci(
        95, plot_par=False, exact=case3_data[0])
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
    
    # (l)
    plt.sca(axs[3,2])
    plt.annotate('(l)', xy=(0.03, 0.93), xycoords='axes fraction')
    lci = case3_data[-1].burnthin(Nb_4row, Nt).plot_ci(
        95, plot_par=True, exact=case3_data[0], markersize=SMALL_SIZE -3)
    lci[0].set_label("Mean")
    lci[1].set_label("Exact")
    lci[2].set_label("$95\\%\;\mathrm{CI}$")
    plt.legend(loc = 'lower right')
    plt.ylabel('$x_i$')
    plt.gca().yaxis.set_label_coords(-0.09, 0.5)
    plt.xlabel('$i$')
    plt.gca().xaxis.set_label_coords(0.5, -0.14)
    tick_ids = np.linspace(0, num_var-1, n_ticks, dtype=int)
    plt.xticks(tick_ids, tick_ids)

    # Save figure
    fig.tight_layout(pad=0, w_pad=0.1, h_pad=0.9)
    plt.savefig(fig_file, bbox_inches='tight', pad_inches=0.01, dpi=1200)
