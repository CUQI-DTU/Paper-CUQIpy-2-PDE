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
    plt.gca().yaxis.set_label_coords(-0.21, 0.45) #-0.12, 0.4
    
    plt.xlabel('$\\xi$')
    plt.gca().xaxis.set_label_coords(.5, -.12) #-0.12, 0.4
    plt.gca().set_ylim(0, .13)
    plt.xlim([0,1])
    
    # b) plot u_custom
    plt.sca(axs[1])
    plt.annotate('(b)', xy=(0.03, 0.93), xycoords='axes fraction')
    
    for i in range(len(u_intermediate)):
        if i==len(u_intermediate)-1:
            plt.plot(grid, u_intermediate[i, :], color='#1f77b4', linestyle='-')
        else:
            plt.plot(grid, u_intermediate[i, :], color=colors[i], linestyle='--')

    plt.legend(['${:.2g}$'.format(t) for t in intermediate_times], loc='upper right', ncol=2, frameon=False, bbox_to_anchor=(1, 0.95))
    
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
                 prior_samples, posterior_samples, Ns_factor,
                 x_step, y_step, y_obs): 
    
    # Matplotlib setup 
    SMALL_SIZE = 7
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 9
    matplotlib_setup(SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE)

    Nb = int(200*Ns_factor) # burn-in
    Nt = None # thinning
    num_var = 3 # number of variables
    n_ticks = 5 # number of ticks 

    fig_file = fig_dir + 'paper_figure5_'+version+'.pdf'

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
    # No need to iterate over all samples in an upcoming release
    for s in prior_samples:
        prior_samples.geometry.plot(s, is_par=True, color=colors[idx]) 
        idx += 1
        if idx == 5:
            break
    plt.ylabel('$\\bm{g}$')
    plt.gca().yaxis.set_label_coords(-0.15, 0.5) 
    plt.xlabel('$\\xi$')
    plt.gca().xaxis.set_label_coords(.5, -.12)
    plt.annotate('(b)', xy=(0.03, 0.93), xycoords='axes fraction')

    # (c) posterior samples
    idx = 0
    plt.sca(axs[0,2])
    for s in posterior_samples.burnthin(int(1000*Ns_factor),
                                        int(1000*Ns_factor)):
        posterior_samples.geometry.plot(s, is_par=True, color=colors[idx])   
        idx += 1
        if idx == 5:
            break
    
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
