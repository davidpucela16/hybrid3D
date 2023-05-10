#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 18:52:07 2023

@author: pdavid
"""

import os 
path=os.path.dirname(__file__)
os.chdir(path)
path_src=os.path.join(path, '../')
os.chdir(path_src)



import numpy as np

import matplotlib.pyplot as plt
import scipy as sp
import scipy.sparse.linalg
import math

import pdb

import matplotlib.pylab as pylab
plt.style.use('default')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10,10),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large', 
         'font.size': 24,
         'lines.linewidth': 2,
         'lines.markersize': 15}
pylab.rcParams.update(params)


from assembly import Assembly_diffusion_3D_interior, Assembly_diffusion_3D_boundaries
from mesh import cart_mesh_3D
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve as dir_solve
from scipy.sparse.linalg import bicg
import numpy as np
import matplotlib.pyplot as plt

import math
from assembly_1D import full_adv_diff_1D
from mesh_1D import mesh_1D
import pdb

from hybrid_set_up_noboundary import hybrid_set_up, visualization_3D

from neighbourhood import get_neighbourhood, get_uncommon


BC_type=np.array(["Neumann", "Neumann", "Neumann","Neumann","Neumann","Neumann"])
BC_type=np.array(["Dirichlet", "Dirichlet","Neumann","Neumann", "Dirichlet","Dirichlet"])
BC_value=np.array([0,0,0,0,0,0])
L_vessel=240
cells_3D=10
n=2
L_3D=np.array([L_vessel, 3*L_vessel, L_vessel])
mesh=cart_mesh_3D(L_3D,cells_3D)

mesh.assemble_boundary_vectors()


# - This is the validation of the 1D transport eq without reaction

D = 1
K=np.array([0.0001,1,0.0001])

U = np.array([2,2,2])/L_vessel
alpha=5
R_vessel=L_vessel/alpha
R_1D=np.zeros(3)+R_vessel

startVertex=np.array([0,1,2])
endVertex=np.array([1,2,3])
pos_vertex=np.array([[L_vessel/2, 0, L_vessel/2],
                     [L_vessel/2, L_vessel,L_vessel/2],
                     [L_vessel/2, 2*L_vessel, L_vessel/2],
                     [L_vessel/2, L_vessel*3,L_vessel/2]
                     ])

vertex_to_edge=[[0],[0,1], [1,2], [2]]
diameters=np.array([2*R_vessel, 2*R_vessel, 2*R_vessel])

cells_per_vessel=25
h=np.zeros(3)+L_vessel/cells_per_vessel

net=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h,1)
net.U=U
net.D=D
net.pos_arrays(mesh)

BCs_1D=np.array([[0,1],
                 [3,0]])

prob=hybrid_set_up(mesh, net, BC_type, BC_value,n,1, np.zeros(len(diameters))+K, BCs_1D)
mesh.get_ordered_connect_matrix()

#%%
# =============================================================================
# prob.Assembly_problem()
# prob.Solve_problem()
# 
# =============================================================================


#%%
import pstats
import cProfile
if __name__ == '__main__':
    # run the profiler on the code and save results to a file
    cProfile.run('prob.Assembly_B()', filename=path + '/profile_results.prof')



stats = pstats.Stats(path + '/profile_results.prof')

stats.strip_dirs().sort_stats('cumulative').print_stats(100)







