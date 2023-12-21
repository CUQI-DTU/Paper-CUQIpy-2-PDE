import matplotlib.pyplot as plt
import dolfin as dl
import numpy as np
import matplotlib.patches as patches

def plot_figure10(g_true, b_exact, y_obs):
    """Function to plot figure 10 in the paper."""
    cm_to_in = 1/2.54
    #fig = plt.figure( figsize=(17.8*cm_to_in, 5*cm_to_in),layout='constrained')
    #subfigs = fig.subfigures(1)
    f, axes = plt.subplots(1,3, figsize=(17.8*cm_to_in, 6*cm_to_in), sharey=True)
    
    t = np.linspace(0,1,251)
    labels = np.linspace(0,1,5)
    
    plt.sca(axes[0])
    plt.plot(t,y_obs[:,0])
    plt.plot(t,b_exact[:,0])
    axes[0].legend([r'noisy data',r'exact data'], loc=1)
    axes[0].set_xticks(labels)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlim([-.05,1.05])
    axes[0].set_ylim([-0.05,.55])
    axes[0].set_xlabel(r'$\tau$')
    axes[0].set_ylabel(r'$u(\xi_L)$')
    axes[0].yaxis.labelpad = -3
    axes[0].xaxis.labelpad = 0
    axes[0].set_title(r'(a) data from left boundaries')
    axes[0].grid()
    
    plt.sca(axes[2])
    plt.plot(t,y_obs[:,1])
    plt.plot(t,b_exact[:,1])
    axes[2].legend([r'noisy data',r'exact data'], loc=1)
    axes[2].set_xticks(labels)
    axes[2].set_xticklabels(labels)
    axes[2].set_xlim([-.05,1.05])
    axes[2].set_ylim([-0.05,.55])
    axes[2].set_xlabel(r'$\tau$')
    axes[2].set_ylabel(r'$u(\xi_R)$')
    axes[2].yaxis.labelpad = -3
    axes[2].xaxis.labelpad = 0
    axes[2].set_title(r'(c) data from right boundaries')
    axes[2].grid()
    
    t = np.linspace(0,1,121)
    labels = np.linspace(0,1,5)
    
    plt.sca(axes[1])
    plt.plot(t,g_true)
    axes[1].set_xticks(labels)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlim([-.05,1.05])
    axes[1].set_ylim([-0.05,.55])
    axes[1].set_xlabel(r'$\xi$')
    axes[1].set_ylabel(r'$u(\xi)$')
    axes[1].yaxis.labelpad = -3
    axes[1].xaxis.labelpad = 0
    axes[1].set_title(r'(b) initial pressure profile')
    axes[1].grid()
    
    plt.tight_layout()

def plot_figure11(g_true, samples_full, samples_half): 
    """Function to plot figure 11 in the paper."""
    cm_to_in = 1/2.54

    f, axes = plt.subplots(1,2, figsize=(17.8*cm_to_in, 6*cm_to_in), sharey=True)
    
    labels = np.linspace(0,1,5)
    
    plt.sca(axes[0])
    samples_full.funvals.vector.plot_ci(95, exact=g_true, subplots=False)
    axes[0].legend([r'95% CI',r'Mean',r'Exact'], loc=1)
    axes[0].set_xticks(labels)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlim([-.05,1.05])
    axes[0].set_ylim([-0.3,0.7])
    axes[0].set_xlabel(r'$\xi$')
    axes[0].set_ylabel(r'$g$')
    axes[0].yaxis.labelpad = -3
    axes[0].xaxis.labelpad = 0
    axes[0].set_title(r'(a) data from both boundaries')
    axes[0].grid()
    
    plt.sca(axes[1])
    samples_half.funvals.vector.plot_ci(95, exact=g_true, subplots=False)
    axes[1].legend([r'95% CI',r'Mean',r'Exact'], loc=1)
    axes[1].set_xticks(labels)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlim([-.05,1.05])
    axes[1].set_ylim([-0.3,0.7])
    axes[1].set_xlabel(r'$\xi$')
    axes[1].yaxis.labelpad = -3
    axes[1].xaxis.labelpad = 0
    axes[1].set_title(r'(b) data from one boundaries')
    axes[1].grid()
    
    plt.tight_layout()


def plot_figure12(samples_full, samples_half):
    """Function to plot figure 12 in the paper."""
    cm_to_in = 1/2.54

    f, axes = plt.subplots(1,2, figsize=(17.8*cm_to_in, 5*cm_to_in), sharey=True)
    
    labels = list(range(0,36,7))
    
    plt.sca(axes[0])
    samples_full.plot_ci(95, plot_par=True, marker='.')
    axes[0].legend([r'Mean',r'95% CT'], loc=4)
    axes[0].set_xticks(labels)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlim([-1,25])
    axes[0].set_ylim([-2.5,2.5])
    axes[0].grid()
    axes[0].set_xlabel(r'$i$')
    axes[0].set_ylabel(r'$x_i$')
    axes[0].yaxis.labelpad = -3
    axes[0].xaxis.labelpad = 0
    axes[0].set_title(r'(a) data from both boundaries')
    plt.sca(axes[1])
    samples_half.plot_ci(95, plot_par=True, marker='.')
    axes[1].legend([r'Mean',r'95% CT'], loc=4)
    axes[1].set_xticks(labels)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlim([-1,25])
    axes[1].set_ylim([-2.5,2.5])
    axes[1].grid()
    axes[1].set_xlabel(r'$i$')
    axes[1].yaxis.labelpad = -3
    axes[1].xaxis.labelpad = 0
    axes[1].set_title(r'(b) data from one boundary')
    
    plt.tight_layout()