import numpy as np

class matern():
    def __init__(self, path, dim=64):
        matern_data = np.load(path)
        self.eig_val = matern_data['l'][:dim]
        self.eig_vec = matern_data['e'][:,:dim]
        self.dim = dim

    def set_levels(self, c_minus=1., c_plus=10.):
        self.c_minus = c_minus
        self.c_plus = c_plus

    def heavy(self, x):
        return self.c_minus*0.5*(1 + np.sign(x)) + self.c_plus*0.5*(1 - np.sign(x))

    def assemble(self, p):
        #return np.ones(833)
        #return 1 + np.exp( self.eig_vec@( np.sqrt(self.eig_val)*p ) )
        return self.heavy( self.eig_vec@( np.sqrt(self.eig_val)*p ) )
        #return self.eig_vec@( np.sqrt(self.eig_val)*p )

if __name__ == '__main__':
    rand_field = matern('matern_basis.npz')
    v = rand_field.assemble( np.random.standard_normal(rand_field.dim) )
    print(v.shape)