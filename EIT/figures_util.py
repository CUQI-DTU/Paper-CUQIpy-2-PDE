import matplotlib.pyplot as plt
import dolfin as dl
import numpy as np
import matplotlib.patches as patches

def plot_figure6(parameter_space, exact_data, data):
    # Expression for the exact conductivity
    class custom_field(dl.UserExpression):
        def set_params(self,cx=np.array([0.5,-0.5,-0.3]),
                       cy= np.array([0.5,0.6,-0.3]),
                       r = np.array([0.2,0.1,0.3]) ):
            self.cx = cx
            self.cy = cy
            self.r2 = r**2
    
        def eval(self,values,x):
            if( (x[0]-self.cx[0])**2 + (x[1]-self.cy[0])**2 < self.r2[0] ):
                values[0] = 10.
            elif( (x[0]-self.cx[1])**2 + (x[1]-self.cy[1])**2 < self.r2[1] ):
                values[0] = 10.
            elif( (x[0]-self.cx[2])**2 + (x[1]-self.cy[2])**2 < self.r2[2] ):
                values[0] = 10.
            else:
                values[0] = 1.
    FEM_el = parameter_space.ufl_element()
    kappa_custom = custom_field(element=FEM_el)
    kappa_custom.set_params()
    
    cm_to_in = 1/2.54
    fig = plt.figure( figsize=(17.8*cm_to_in, 7*cm_to_in))
    subfigs = fig.subfigures(1, 2)
    
    axes = subfigs[0].subplots(1,2,sharey=True)
    
    circ1 = patches.Circle([0,0], radius=1,color='#450558', fill=True)
    circ2 = patches.Circle([0.5,0.5], radius=0.2,color='white', fill=True)
    circ3 = patches.Circle([-0.5,0.6], radius=0.1,color='white', fill=True)
    circ4 = patches.Circle([-0.3,-0.3], radius=0.3,color='white', fill=True)

    # Read theta and idx used for plotting data:
    boundary_coord_data = np.load('./obs/boundary_coordinates.npz')
    theta = boundary_coord_data['theta']
    idx = boundary_coord_data['idx']

    
    axes[0].add_patch(circ1)
    axes[0].add_patch(circ2)
    axes[0].add_patch(circ3)
    axes[0].add_patch(circ4)
    axes[0].set_xlim([-1.1,1.1])
    axes[0].set_ylim([-1.1,1.1])
    axes[0].set_xticks([-1,0,1])
    axes[0].set_yticks([-1,0,1])
    axes[0].set_aspect('equal')
    axes[0].set_xlabel(r'$\xi_1$')
    axes[0].set_ylabel(r'$\xi_2$')
    axes[0].yaxis.labelpad = -3
    axes[0].xaxis.labelpad = 0
    axes[0].set_title(r'true conductivity')
    
    plt.sca(axes[1])
    func = dl.Function(parameter_space)
    func = dl.interpolate(kappa_custom,parameter_space)
    c = dl.plot(func)
    axes[1].set_xlim([-1.1,1.1])
    axes[1].set_ylim([-1.1,1.1])
    axes[1].set_xticks([-1,0,1])
    axes[1].set_yticks([-1,0,1])
    axes[1].set_xlabel(r'$\xi_1$')
    axes[1].yaxis.labelpad = -3
    axes[1].xaxis.labelpad = 0
    axes[1].set_title(r'projected conductivity')
    
    subfigs[0].suptitle('(a) conductivity field', fontsize=12)    
    axes = subfigs[1].subplots(1,4,sharey=True)
    axes[0].plot( theta[idx], data[0][idx] , '-')
    axes[0].plot( theta[idx], exact_data[0][idx] , 'k-')
    axes[0].set_xlim([-np.pi-0.1,np.pi+0.1])
    axes[0].set_xticks([-np.pi,0])
    axes[0].set_xticklabels([r'$-\pi$',r'$0$'])
    axes[0].set_xlabel(r'$\theta$')
    axes[0].set_ylabel(r'$u$')
    axes[0].set_title(r'$k=1$')
    axes[0].yaxis.labelpad = -3
    axes[0].xaxis.labelpad = 0
    axes[1].plot( theta[idx], data[1][idx] , '-')
    axes[1].plot( theta[idx], exact_data[1][idx] , 'k-')
    axes[1].set_xlim([-np.pi-0.1,np.pi+0.1])
    axes[1].set_xticks([-np.pi,0])
    axes[1].set_xticklabels([r'$-\pi$',r'$0$'])
    axes[1].set_xlabel(r'$\theta$')
    axes[1].set_title(r'$k=2$')
    axes[1].xaxis.labelpad = 0
    axes[2].plot( theta[idx], data[2][idx] , '-')
    axes[2].plot( theta[idx], exact_data[2][idx] , 'k-')
    axes[2].set_xlim([-np.pi-0.1,np.pi+0.1])
    axes[2].set_xticks([-np.pi,0])
    axes[2].set_xticklabels([r'$-\pi$',r'$0$'])
    axes[2].set_xlabel(r'$\theta$')
    axes[2].set_title(r'$k=3$')
    axes[2].xaxis.labelpad = 0
    axes[3].plot( theta[idx], data[3][idx] , '-')
    axes[3].plot( theta[idx], exact_data[3][idx] , 'k-')
    axes[3].set_xlim([-np.pi-0.1,np.pi+0.1])
    axes[3].set_xticks([-np.pi,0,np.pi])
    axes[3].set_xticklabels([r'$-\pi$',r'$0$',r'$\pi$'])
    axes[3].set_xlabel(r'$\theta$')
    axes[3].set_title(r'$k=4$')
    axes[3].xaxis.labelpad = 0
    
    axes[0].set_zorder(1)
    axes[0].legend(['noisy data','exact data'])
    
    subfigs[1].suptitle('(b) measurements with 20% noise', fontsize=12)
    
    subfigs[1].subplots_adjust(wspace=0, bottom=0.2, top=0.8)


def plot_figure7(prior_samples, cuqi_samples1, cuqi_samples2, cuqi_samples3):
    cm_to_in = 1/2.54
    fig = plt.figure( figsize=(17.8*cm_to_in, 10.5*cm_to_in),layout='constrained')
    subfigs = fig.subfigures(2, 1)
    
    axes = subfigs[0].subplots(1,4,sharey=True)
    plt.sca(axes[0])
    prior_samples.plot([ 0 ], subplots=False)
    axes[0].set_xlim([-1.1,1.1])
    axes[0].set_ylim([-1.1,1.1])
    axes[0].set_xticks([-1,0,1])
    axes[0].set_yticks([-1,0,1])
    axes[0].yaxis.labelpad = -3
    axes[0].xaxis.labelpad = 0
    axes[0].set_xlabel(r'$\xi_1$')
    axes[0].set_ylabel(r'$\xi_2$')

    plt.sca(axes[1])
    cuqi_samples1.plot([ 0 ], subplots=False)
    axes[1].set_ylabel('')
    axes[1].set_xlim([-1.1,1.1])
    axes[1].set_ylim([-1.1,1.1])
    axes[1].set_xticks([-1,0,1])
    axes[1].xaxis.labelpad = 0
    axes[1].set_xlabel(r'$\xi_1$')

    plt.sca(axes[2])
    cuqi_samples2.plot([ 0 ], subplots=False)
    axes[2].set_ylabel('')
    axes[2].set_xlim([-1.1,1.1])
    axes[2].set_ylim([-1.1,1.1])
    axes[2].set_xticks([-1,0,1])
    axes[2].xaxis.labelpad = 0
    axes[2].set_xlabel(r'$\xi_1$')

    plt.sca(axes[3])
    cuqi_samples3.plot([ 0 ], subplots=False)
    #prior_samples.plot([ 3 ], subplots=False)
    #axes[0,3].set_yticks([])
    axes[3].set_ylabel('')
    axes[3].set_xlim([-1.1,1.1])
    axes[3].set_ylim([-1.1,1.1])
    axes[3].set_xticks([-1,0,1])
    axes[3].xaxis.labelpad = 0
    axes[3].set_xlabel(r'$\xi_1$')
    
    axes = subfigs[1].subplots(1,4,sharey=True)
    plt.sca(axes[0])
    prior_samples.plot([ 2 ], subplots=False)
    axes[0].set_xlim([-1.1,1.1])
    axes[0].set_ylim([-1.1,1.1])
    axes[0].set_xticks([-1,0,1])
    axes[0].set_yticks([-1,0,1])
    axes[0].yaxis.labelpad = -3
    axes[0].xaxis.labelpad = 0
    axes[0].set_xlabel(r'$\xi_1$')
    axes[0].set_ylabel(r'$\xi_2$')

    plt.sca(axes[1])
    cuqi_samples1.plot([ -10 ], subplots=False)
    axes[1].set_ylabel('')
    axes[1].set_xlim([-1.1,1.1])
    axes[1].set_ylim([-1.1,1.1])
    axes[1].set_xticks([-1,0,1])
    axes[1].xaxis.labelpad = 0
    axes[1].set_xlabel(r'$\xi_1$')

    plt.sca(axes[2])
    cuqi_samples2.plot([ -10 ], subplots=False)
    axes[2].set_ylabel('')
    axes[2].set_xlim([-1.1,1.1])
    axes[2].set_ylim([-1.1,1.1])
    axes[2].set_xticks([-1,0,1])
    axes[2].xaxis.labelpad = 0
    axes[2].set_xlabel(r'$\xi_1$')
    
    plt.sca(axes[3])
    cuqi_samples3.plot([ -10 ], subplots=False)
    axes[3].set_ylabel('')
    axes[3].set_xlim([-1.1,1.1])
    axes[3].set_ylim([-1.1,1.1])
    axes[3].set_xticks([-1,0,1])
    axes[3].xaxis.labelpad = 0
    axes[3].set_xlabel(r'$\xi_1$')


def plot_figure8(cuqi_samples1, cuqi_samples2, cuqi_samples3):

    cm_to_in = 1/2.54
    fig = plt.figure( figsize=(17.8*cm_to_in, 20*cm_to_in))#,layout='constrained')
    subfigs = fig.subfigures(4, 1)

    # Use KL geometry for plotting
    domain_geometry = cuqi_samples1.geometry
    cuqi_samples1.geometry = domain_geometry.geometry
    cuqi_samples2.geometry = domain_geometry.geometry
    cuqi_samples3.geometry = domain_geometry.geometry

    axes = subfigs[2].subplots(1,3,sharey=True)
    plt.sca(axes[0])
    im = cuqi_samples1.plot_mean(subplots=False)
    axes[0].set_title('5% noise')
    axes[0].set_xlim([-1.1,1.1])
    axes[0].set_ylim([-1.1,1.1])
    axes[0].set_xticks([-1,1])
    axes[0].set_yticks([-1,0,1])
    axes[0].yaxis.labelpad = -3
    axes[0].xaxis.labelpad = -7
    axes[0].set_xlabel(r'$\xi_1$')
    axes[0].set_ylabel(r'$\xi_2$')
    im[0].set_clim([-0.1,0.25])
    plt.sca(axes[1])
    im = cuqi_samples1.plot_mean(subplots=False)
    axes[1].set_title('10% noise')
    axes[1].set_ylabel('')
    axes[1].set_xlim([-1.1,1.1])
    axes[1].set_ylim([-1.1,1.1])
    axes[1].set_xticks([-1,1])
    axes[1].xaxis.labelpad = -7
    axes[1].set_xlabel(r'$\xi_1$')
    im[0].set_clim([-0.1,0.25])
    plt.sca(axes[2])
    im = cuqi_samples1.plot_mean(subplots=False)
    axes[2].set_title('20% noise')
    axes[2].set_ylabel('')
    axes[2].set_xlim([-1.1,1.1])
    axes[2].set_ylim([-1.1,1.1])
    axes[2].set_xticks([-1,1])
    axes[2].xaxis.labelpad = -7
    axes[2].set_xlabel(r'$\xi_1$')
    im[0].set_clim([-0.1,0.25])
    subfigs[2].colorbar(im[0], fraction=0.047)
    subfigs[2].subplots_adjust(wspace=0,right=.9,top = 0.8)
    subfigs[2].suptitle(r'(c) posterior mean visualized in $\mathbf{G}_{KL}$ geometry', fontsize=12)
        
    axes = subfigs[3].subplots(1,3,sharey=True)
    plt.sca(axes[0])
    c = cuqi_samples1.funvals.vector.plot_variance(subplots=False)
    axes[0].set_title('5% noise')
    axes[0].set_xlim([-1.1,1.1])
    axes[0].set_ylim([-1.1,1.1])
    axes[0].set_xticks([-1,1])
    axes[0].set_yticks([-1,0,1])
    axes[0].yaxis.labelpad = -3
    axes[0].xaxis.labelpad = -7
    axes[0].set_xlabel(r'$\xi_1$')
    axes[0].set_ylabel(r'$\xi_2$')
    plt.sca(axes[1])
    c = cuqi_samples2.funvals.vector.plot_variance(subplots=False)
    axes[1].set_title('10% noise')
    axes[1].set_ylabel('')
    axes[1].set_xlim([-1.1,1.1])
    axes[1].set_ylim([-1.1,1.1])
    axes[1].set_xticks([-1,1])
    axes[1].xaxis.labelpad = -7
    axes[1].set_xlabel(r'$\xi_1$')
    plt.sca(axes[2])
    c = cuqi_samples3.funvals.vector.plot_variance(subplots=False)
    axes[2].set_title('20% noise')
    axes[2].set_ylabel('')
    axes[2].set_xlim([-1.1,1.1])
    axes[2].set_ylim([-1.1,1.1])
    axes[2].set_xticks([-1,1])
    axes[2].xaxis.labelpad = -7
    axes[2].set_xlabel(r'$\xi_1$')
    subfigs[3].colorbar(c[0], fraction=0.047)
    subfigs[3].suptitle(
        r'(d) point-wise variance evaluated in $\mathbf{G}_{KL}$ geometry',
        fontsize=12)

    # Use Heaviside geometry for plotting
    cuqi_samples1.geometry = domain_geometry
    cuqi_samples2.geometry = domain_geometry
    cuqi_samples3.geometry = domain_geometry
    
    axes = subfigs[0].subplots(1,3,sharey=True)
    plt.sca(axes[0])
    im = cuqi_samples1.plot_mean(subplots=False)
    axes[0].set_title('5% noise')
    axes[0].set_xlim([-1.1,1.1])
    axes[0].set_ylim([-1.1,1.1])
    axes[0].set_xticks([-1,1])
    axes[0].set_yticks([-1,0,1])
    axes[0].set_ylabel(r'$y$')
    axes[0].set_xlabel(r'$x$')
    axes[0].yaxis.labelpad = -3
    axes[0].xaxis.labelpad = -7
    axes[0].set_xlabel(r'$\xi_1$')
    axes[0].set_ylabel(r'$\xi_2$')
    plt.sca(axes[1])
    im = cuqi_samples2.plot_mean(subplots=False)
    axes[1].set_title('10% noise')
    axes[1].set_ylabel('')
    axes[1].set_xlim([-1.1,1.1])
    axes[1].set_ylim([-1.1,1.1])
    axes[1].set_xticks([-1,1])
    axes[1].set_xlabel(r'$x$')
    axes[1].xaxis.labelpad = -7
    axes[1].set_xlabel(r'$\xi_1$')
    plt.sca(axes[2])
    im = cuqi_samples3.plot_mean(subplots=False)
    axes[2].set_title('20% noise')
    axes[2].set_ylabel('')
    axes[2].set_xlim([-1.1,1.1])
    axes[2].set_ylim([-1.1,1.1])
    axes[2].set_xticks([-1,1])
    axes[2].set_xlabel(r'$x$')
    axes[2].xaxis.labelpad = -7
    axes[2].set_xlabel(r'$\xi_1$')
    subfigs[0].colorbar(im[0], fraction=0.047)
    subfigs[0].suptitle(
        r'(a) posterior mean visualized in $\mathbf{G}_{Heavi}$ geometry',
        fontsize=12)
    
    axes = subfigs[1].subplots(1,3,sharey=True)
    plt.sca(axes[0])
    c = cuqi_samples1.funvals.vector.plot_variance(subplots=False)
    axes[0].set_title('5% noise')
    axes[0].set_xlim([-1.1,1.1])
    axes[0].set_ylim([-1.1,1.1])
    axes[0].set_xticks([-1,1])
    axes[0].set_yticks([-1,0,1])
    axes[0].set_ylabel(r'$y$')
    axes[0].set_xlabel(r'$x$')
    axes[0].yaxis.labelpad = -3
    axes[0].xaxis.labelpad = -7
    axes[0].set_xlabel(r'$\xi_1$')
    axes[0].set_ylabel(r'$\xi_2$')
    plt.sca(axes[1])
    c = cuqi_samples2.funvals.vector.plot_variance(subplots=False)
    axes[1].set_title('10% noise')
    axes[1].set_ylabel('')
    axes[1].set_xlim([-1.1,1.1])
    axes[1].set_ylim([-1.1,1.1])
    axes[1].set_xticks([-1,1])
    axes[1].set_xlabel(r'$x$')
    axes[1].xaxis.labelpad = -7
    axes[1].set_xlabel(r'$\xi_1$')
    plt.sca(axes[2])
    c= cuqi_samples3.funvals.vector.plot_variance(subplots=False)
    axes[2].set_title('20% noise')
    axes[2].set_ylabel('')
    axes[2].set_xlim([-1.1,1.1])
    axes[2].set_ylim([-1.1,1.1])
    axes[2].set_xticks([-1,1])
    axes[2].set_xlabel(r'$x$')
    axes[2].xaxis.labelpad = -7
    axes[2].set_xlabel(r'$\xi_1$')
    
    subfigs[1].colorbar(c[0], fraction=0.047)
    subfigs[1].suptitle(
        r'(b) point-wise variance evaluated in $\mathbf{G}_{Heavi}$ geometry',
        fontsize=12)
    

def plot_figure9(cuqi_samples1, cuqi_samples2, cuqi_samples3):

    cm_to_in = 1/2.54
    f, axes = plt.subplots(
        1,3, figsize=(17.8*cm_to_in, 5*cm_to_in), sharey=True)
    
    labels = list(range(0,36,7))
    
    plt.sca(axes[0])
    cuqi_samples1.plot_ci(95, plot_par=True, marker='.')
    axes[0].legend([r'Mean',r'95% CT'], loc=4)
    axes[0].set_xticks(labels)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlim([-1,36])
    axes[0].set_ylim([-5,3])
    axes[0].grid()
    axes[0].set_xlabel(r'$i$')
    axes[0].set_ylabel(r'$x_i$')
    axes[0].yaxis.labelpad = -3
    axes[0].xaxis.labelpad = 0
    axes[0].set_title(r'(a) 5% noise')

    plt.sca(axes[1])
    cuqi_samples2.plot_ci(95, plot_par=True, marker='.')
    axes[1].legend([r'Mean',r'95% CT'], loc=4)
    axes[1].set_xticks(labels)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlim([-1,36])
    axes[1].set_ylim([-5,3])
    axes[1].grid()
    axes[1].set_xlabel(r'$i$')
    axes[1].yaxis.labelpad = -3
    axes[1].xaxis.labelpad = 0
    axes[1].set_title(r'(b) 10% noise')

    plt.sca(axes[2])
    cuqi_samples3.plot_ci(95, plot_par=True, marker='.')
    axes[2].legend([r'Mean',r'95% CT'], loc=4)
    axes[2].set_xticks(labels)
    axes[2].set_xticklabels(labels)
    axes[2].set_xlim([-1,36])
    axes[2].set_ylim([-3,5])
    axes[2].grid()
    axes[2].set_xlabel(r'$i$')
    axes[2].yaxis.labelpad = -3
    axes[2].xaxis.labelpad = 0
    axes[2].set_title(r'(c) 20% noise')
    
    plt.tight_layout()
    