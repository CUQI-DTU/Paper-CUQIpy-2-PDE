import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
#from matern import matern

#from progressbar import progressbar

from dolfin import *

class init_cond(UserExpression):
    def __init__(self, center=np.array([0.5]), radius=0.1, **kwargs):
        self.center = center
        self.r = radius
        super().__init__(**kwargs)
    def eval(self, values, x):
        if( (x[0] > 0.5) and (x[0] < 0.7)  ):
            r = x[0] - 0.6
            values[0] = 0.5*(1-1/(1 + np.exp(-100*r)))
        elif( (x[0] > 0.1) and (x[0] < 0.3)  ):
            r = x[0] - 0.2
            values[0] = 0.5*(1/(1 + np.exp(-100*r)))
        elif( (x[0] > 0.85) and (x[0] < 0.99)  ):
            r = x[0] - 0.9
            values[0] = 0.25*(1-1/(1 + np.exp(-100*r)))
        elif( (x[0] > 0.71) and (x[0] < 0.85)  ):
            r = x[0] - 0.8
            values[0] = 0.25*(1/(1 + np.exp(-100*r)))



class wave():
    def __init__(self):
        # defining the mesh
        self.mesh = UnitIntervalMesh(120)
        #self.mesh = RectangleMesh(Point(-5, -5), Point(5, 5), 120, 120)

        # defining the function space
        self.V = FunctionSpace(self.mesh,'CG', 1)

        FEM_el = self.V.ufl_element()
        init = init_cond(element=FEM_el)

        # saving the initial condition
        #f = Function(self.V)
        #f = interpolate( init, self.V )
        #file = File('temp.pvd')
        #file << f
        #exit()

        # problem paramters
#        self.c2 = 5000**2
#        self.T = 0.001
#        self.dt = 0.0000001
#        self.num_time_steps  = 1500#int(self.T/self.dt)+1

        self.c2 = 1
        self.T = 1
        self.dt = 0.004
        self.num_time_steps  = int(self.T/self.dt)+1

        # defining test and trial spaces
        u0 = Constant('0.0')
        self.t = TestFunction(self.V)
        self.u = TrialFunction(self.V)
        self.v = TrialFunction(self.V)

        # functions storing the values of previous time step
        self.u_past = Function( self.V )
        self.u_past = interpolate(init, self.V )

        #np.savez('./obs/init_pressure.npz', init_pressure=self.u_past.vector().get_local()[::-1])
        #plt.plot(self.u_past.vector().get_local())
        #plt.savefig('init.pdf',format='pdf',dpi=300)

        self.u_past_numpy = self.u_past.vector().get_local()
        self.v_past = Function( self.V )

        # weak forms for the two equations
        self.a1 = self.v*self.t*dx 
        self.L1 = self.v_past*self.t*dx - self.dt/2*self.c2*inner( grad(self.u_past), grad(self.t) )*dx - self.dt/2*self.v_past*self.t*ds

        self.a2 = self.u*self.t*dx 
        self.L2 = self.u_past*self.t*dx + self.dt*self.v_past*self.t*dx

        self.temp = Function( self.V )

        self.compute_boundary_coordinates()
        A = assemble(self.a1)
        self.solver = LUSolver(A)

    # This is a second order syplectic time integrator to numerically preserve the Hamiltonian
    def stormer_verlet_step(self):
        b1 = assemble(self.L1)
        self.solver.solve(self.temp.vector(), b1)
        self.v_past.vector().set_local( self.temp.vector().get_local() )

        temp = self.u_past.vector().get_local() + self.dt*self.v_past.vector().get_local()
        self.u_past.vector().set_local( temp )

        b1 = assemble(self.L1)
        self.solver.solve(self.temp.vector(), b1)
        self.v_past.vector().set_local( self.temp.vector().get_local() )

    # This applies the time-stepping routine
    def time_stepping(self):
        # Here we save the LU decomposition of the mass matrix (share in)
        A = assemble(self.a1)
        self.solver = LUSolver(A)

#        sol = Function(self.V)

        # This is to save the solution to file
#        path = './solution/sol.pvd'
#        file = File(path)

        sol = []
        for i in progressbar( range(self.num_time_steps) ):
            self.stormer_verlet_step()

            # uncomment to save solution to file
            sol.append( self.u_past.vector().get_local() )

        sol = np.array(sol)
        print(sol.shape)

        plt.imshow(sol)
        plt.colorbar()
        plt.savefig('fig.pdf',format='pdf',dpi=300)

    def set_initial_pressure(self,p0):
        self.u_past.vector().set_local( p0.vector().get_local() )
        self.v_past.vector().set_local( np.zeros_like( p0.vector().get_local() ) )

    def forward_full(self, p0):
        self.u_past.vector().set_local( p0.vector().get_local() )
        self.v_past.vector().set_local( np.zeros_like( p0.vector().get_local() ) )

        return self.read_time_full_boundary()

    def forward_half(self, p0):
        self.u_past.vector().set_local( p0.vector().get_local() )
        self.v_past.vector().set_local( np.zeros_like( p0.vector().get_local() ) )

        return self.read_time_half_boundary()

    def read_time_full_boundary(self):
        obs = []
        for i in range(self.num_time_steps):
            self.stormer_verlet_step()
            obs.append( self.extract_full_boundary() )
        return np.array(obs)

    def read_time_half_boundary(self):
        obs = []
        for i in range(self.num_time_steps):
            self.stormer_verlet_step()
            obs.append( self.extract_half_boundary() )
        return np.array(obs)

    def extract_full_boundary(self):
        obs = []
        for i in range( self.full_boundary.shape[0] ):
            obs.append( self.u_past(self.full_boundary[i]) )
        return np.array(obs)

    def extract_half_boundary(self):
        obs = []
        for i in range( self.half_boundary.shape[0] ):
            obs.append( self.u_past(self.half_boundary[i]) )
        return np.array(obs)

    def compute_boundary_coordinates(self):
        self.full_boundary = np.array( [0.001, 0.999] )
        self.half_boundary = np.array( [0.001] )

    def save_observations(self):
        FEM_el = self.V.ufl_element()
        init = init_cond(element=FEM_el)

        temp = Function(self.V)
        temp = interpolate( init, self.V )

        self.u_past.vector().set_local( temp.vector().get_local() )
        self.v_past.vector().set_local( np.zeros_like( self.v_past.vector().get_local() ) )
        plt.figure()
        x = np.linspace(0,1,121)
        plt.plot(x,self.u_past.vector().get_local()[::-1])
        #plt.savefig('./plots/true_pressure.pdf',format='pdf',dpi=300)

        b_exact = self.read_time_full_boundary()
        plt.figure()
        plt.plot(b_exact)
        plt.savefig('observation.pdf')
        #exit()
        SNR = 20
        sigma = np.linalg.norm(b_exact)/SNR
        noise_vec = np.random.standard_normal( b_exact.shape )
        noise_vec /= np.linalg.norm(noise_vec)
        data = b_exact + sigma*noise_vec
        sigma2 = sigma*sigma
        np.savez( './obs/full_boundary_5per.npz', data=data, b_exact=b_exact, noise_vec=noise_vec, sigma2=sigma2 )

        self.u_past.vector().set_local( temp.vector().get_local() )
        self.v_past.vector().set_local( np.zeros_like( self.v_past.vector().get_local() ) )

        b_exact = self.read_time_half_boundary()
        plt.figure()
        plt.plot(b_exact)
        plt.savefig('observation2.pdf')
        #exit()
        SNR = 20
        sigma = np.linalg.norm(b_exact)/SNR
        noise_vec = np.random.standard_normal( b_exact.shape )
        noise_vec /= np.linalg.norm(noise_vec)
        data = b_exact + sigma*noise_vec
        sigma2 = sigma*sigma
        np.savez( './obs/half_boundary_5per.npz', data=data, b_exact=b_exact, noise_vec=noise_vec, sigma2=sigma2 )

if __name__ == '__main__':
    problem = wave()
    #problem.time_stepping()
    #obs = problem.read_time_full_boundary().reshape(251,-1)
    #plt.figure()
    #plt.plot( obs[:334] )
    #plt.plot(obs[334:])
    #plt.savefig('obs.pdf',format='pdf',dpi=300)
    problem.save_observations()