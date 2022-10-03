import scipy.stats as scp
import numpy as np
import matplotlib
from progressbar import progressbar

class Random_Walk():
    def __init__(self, pi_like, x0, history_length=100):
        self.pi_like = pi_like
        self.x0 = x0
        self.ndim = self.x0.shape[0]

        self.target_last = pi_like(x0)
        self.sample_last = self.x0

        self.all_samples = [ self.x0 ]
        self.all_targets = [ self.target_last ]
        self.all_accepts = [ 1 ]

        self.update = pCN_update( pi_like, self.sample_last)

        self.history_length = history_length
        self.star_acc = 0.234    # target acceptance rate RW
        self.num_adapt = 0
        self.beta = 0.01

    def tune(self):
        av_acc = np.mean( self.all_accepts[-self.history_length:] )
        zeta = 1/np.sqrt(self.num_adapt+1)
        self.beta = np.exp(np.log(self.beta) + zeta*(av_acc-self.star_acc))
        self.num_adapt += 1
        #print('here')
        #print(self.beta)
        #input()

    def sample(self, Ns):
        for i in progressbar( range(1,Ns) ):
            sample, target, acc = self.update.step(self.beta)

            if(i%self.history_length == 0):
                self.tune()

            self.sample_last = sample
            self.target_last = target

            self.all_samples.append( sample )
            self.all_targets.append( target )
            self.all_accepts.append( acc )

    def print_stat(self):
        print( np.array( self.all_accepts ).mean() )

    def give_stats(self):
        return np.array(self.all_samples)

class pCN_update:
    def __init__(self, pi_target,x_old):
        self.pi_target = pi_target
        self.x_old = x_old
        self.target_old = self.pi_target( self.x_old )
        self.dim = self.x_old.shape[0]

    def set_pi_target(self, pi_target):
        self.pi_target

    def step(self,beta):
        x = np.sqrt( 1 - beta**2 ) * self.x_old + beta*np.random.standard_normal( self.dim )

        target = self.pi_target( x )
        ratio = np.exp(target - self.target_old)
        alpha = min(1., ratio)
        uu = np.random.uniform(0,1)
        if (uu <= alpha):
            acc = 1
            self.x_old = x
            self.target_old = target
        else:
            acc = 0

        return self.x_old, self.target_old, acc



