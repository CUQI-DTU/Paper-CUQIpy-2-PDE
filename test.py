import numpy as np
import dolfin as dl
import cuqipy_fenics
import cuqi
import matplotlib.pyplot as plt
from progressbar import progressbar

class x_boundary(dl.UserExpression):
    def eval(self, values, x):
        values[0] = x[0]

class y_boundary(dl.UserExpression):
    def eval(self, values, x):
        values[0] = x[1]

#%% 1.1 loading mesh
mesh = dl.Mesh("mesh.xml")

#%% 1.2 Define function spaces 
parameter_space = dl.FunctionSpace(mesh, "CG", 1)
FEM_el = parameter_space.ufl_element()

x_func = x_boundary(element=FEM_el)
y_func = y_boundary(element=FEM_el)

u0 = dl.Constant('0.0')
boundary = lambda x, on_boundary: on_boundary
zero_bc = dl.DirichletBC(parameter_space, u0, boundary)

x_bnd = dl.DirichletBC(parameter_space, x_func, boundary)
y_bnd = dl.DirichletBC(parameter_space, y_func, boundary)

dummy = dl.Function(parameter_space)
dummy.vector().set_local( np.ones_like( dummy.vector().get_local() ) )
zero_bc.apply( dummy.vector() )
bnd_idx = np.argwhere( dummy.vector().get_local() == 0 ).flatten()

func = dl.Function(parameter_space)
x_bnd.apply( func.vector() )
x_coords = func.vector().get_local()[bnd_idx]

y_bnd.apply( func.vector() )
y_coords = func.vector().get_local()[bnd_idx]

boundary_polar_coord = np.arctan2(y_coords , x_coords)
idx = np.argsort( boundary_polar_coord )

np.savez( 'boundary_coordinates.npz', theta=boundary_polar_coord, idx=idx )




