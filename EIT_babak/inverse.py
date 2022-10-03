import numpy as np
import matplotlib.pyplot as plt
#from poisson import poisson
from mcmc import Random_Walk

def run_mcmc():
    obs_data = np.load('./obs/obs1.npz')
    
    y_true = obs_data['obs_vec']
    noise_vec = obs_data['noise_vec']
    sigma = np.linalg.norm(y_true)/1000
    sigma2 = sigma*sigma
    y_obs = y_true + sigma*noise_vec

    problem = poisson()
    problem.precomute_rhs()

    log_like = lambda p: - ( 0.5*np.sum(  (problem.forward(p) - y_obs)**2)/sigma2 )

    x0 = np.ones(64)
    sampler = Random_Walk(log_like, x0)

    sampler.sample(100000)
    sampler.print_stat()
    samples = sampler.give_stats()

    np.savez('./stat/stat1.npz', samples=samples)

def post_process():
    stat_data = np.load('./stat/stat1.npz')
    samples = stat_data['samples']

    plt.plot(samples[:,2])
    plt.show()

if __name__ == '__main__':
    #run_mcmc()
    post_process()
