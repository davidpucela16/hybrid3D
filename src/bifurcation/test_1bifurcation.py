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
          'figure.figsize': (15,15),
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
from Green import get_source_potential
import pdb

from hybrid_set_up_noboundary import hybrid_set_up, visualization_3D

from neighbourhood import get_neighbourhood, get_uncommon





L_vessel=240
#%%
# - This is the validation of the 1D transport eq without reaction
alpha=50
R_vessel=L_vessel/alpha
R_1D=np.array([1,0.5,0.5])*R_vessel
D_1D = 10
K=np.array([1,1,1])*R_1D/R_vessel

U = np.array([2,2,2])/L_vessel*10

U[1]=U[0]/2*(R_1D[0]/R_1D[1])**2
U[2]=U[1]


startVertex=np.array([0,1,1])
endVertex=np.array([1,2,3])
aux=L_vessel*15**0.5/4
pos_vertex=np.array([[L_vessel/2, 0, L_vessel/2],
                     [L_vessel/2, L_vessel,L_vessel/2],
                     [L_vessel/2, L_vessel+aux, L_vessel*3/4],
                     [L_vessel/2, L_vessel+aux,L_vessel/4]
                     ])

vertex_to_edge=[[0],[0,1,2], [1], [2]]
diameters=2*R_1D

cells_per_vessel=10
h=np.zeros(3)+L_vessel/cells_per_vessel

net=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h,D_1D)
net.U=U
net.D=D_1D


#%% - Assembly of 3D data
cells_3D=10
n=2
L_3D=np.array([L_vessel, 2*L_vessel, L_vessel])
mesh=cart_mesh_3D(L_3D,cells_3D)

mesh.assemble_boundary_vectors()

net.pos_arrays(mesh)

#%% - Set boundary conditions
BCs_1D=np.array([[0,1],
                 [2,0],
                 [3,0]])

BC_type=np.array(["Neumann", "Neumann", "Neumann","Neumann","Neumann","Neumann"])
BC_type=np.array(["Dirichlet", "Dirichlet","Neumann","Neumann", "Dirichlet","Dirichlet"])
BC_value=np.array([0,0,0,0,0,0])

#%% - Intra vascular problem
prob=hybrid_set_up(mesh, net, BC_type, BC_value,n,1, np.zeros(len(diameters))+K, BCs_1D)
mesh.get_ordered_connect_matrix()

prob.Assembly_I()


tot_cell=cells_per_vessel*len(startVertex)
x=net.pos_s[:net.cells[0],1]
Pe_0=U[0]*L_vessel/D_1D
Pe_1=U[1]*L_vessel/D_1D
A,B,C,d=np.linalg.solve(np.array([[1,1,0,0],
                                  [np.exp(Pe_0), 1, -1, -1],
                                  [0,0,np.exp(Pe_1), 1],
                                  [np.exp(Pe_0)*Pe_0, 0, -Pe_1, 0]]),
                        np.array([1,0,0,0]))


analytical_0 = lambda s : A*np.exp(Pe_0*s/L_vessel)+B
analytical_1 = lambda s : C*np.exp(Pe_1*s/L_vessel)+d


full_analytical=np.concatenate((analytical_0(x), analytical_1(x), analytical_1(x)))

text_1 = 'vessel 0'
text_2 = 'vessel 1'
text_3 = 'vessel 2'
props = dict(boxstyle='round', facecolor='white', alpha=0.5)

plt.title("Converged estimation of the flux")
plt.plot(full_analytical, label="analytical")

plt.plot(dir_solve(prob.I_matrix, -prob.III_ind_array), label="numerical")
plt.text(0, 0.6, text_1,  fontsize=40,
        verticalalignment='top', bbox=props)

plt.text(10, 0.6, text_2,  fontsize=40,
        verticalalignment='top', bbox=props)

plt.text(20, 0.6, text_3,  fontsize=40,
        verticalalignment='top', bbox=props)
plt.legend()
plt.show()

#%%
prob.Assembly_problem()

print("If all BCs are newton the sum of all coefficients divided by the length of the network should be close to 1", np.sum(prob.B_matrix.toarray())/net.L)


#%% - We solve the 2D problem in a 3D setting to be able to validate. Ideally, there is no transport in the y direction
C_v_array=np.ones(len(net.pos_s)) #Though this neglects completely the contribution from the dipoles

L1=sp.sparse.hstack((prob.A_matrix,prob.B_matrix))
L2=sp.sparse.hstack((prob.D_matrix, prob.E_matrix))

Li=sp.sparse.vstack((L1,L2))

ind=np.concatenate((prob.I_ind_array, prob.F_matrix.dot(C_v_array)))

sol=dir_solve(Li, -ind)


plt.plot(net.pos_s[:,1],sol[-prob.S:], label='hybrid reaction')
plt.title("q(s) with C_v=1")
plt.legend()
plt.show()

prob.s=sol[:-prob.S]
prob.q=sol[-prob.S:]


phi_bar=np.array([])
for i in range(prob.S):
    phi_bar=np.append(phi_bar, -prob.q[i]/prob.K[net.source_edge[i]]+1)
    

plt.plot(prob.q, label="q")
plt.plot(phi_bar, label="phi_bar")
plt.legend()

#%%
a=visualization_3D([0, L_vessel], 50, prob, 12, 0.5, np.array([0,L_vessel/2,0]))


# =============================================================================
# #%%  - Validation with 2D code
# 
# Lin_matrix=sp.sparse.hstack((prob.A_matrix, prob.B_matrix))
# Lin_matrix=sp.sparse.vstack((Lin_matrix, sp.sparse.hstack((prob.D_matrix, prob.E_matrix))))
# 
# prob.C_v_array=np.ones(len(net.pos_s))
# 
# 
# 
# sol=dir_solve(Lin_matrix, -np.concatenate((prob.I_ind_array, prob.F_matrix.dot(prob.C_v_array))))
# 
# prob.s=sol[:prob.F]
# prob.q=sol[prob.F:]
# plt.plot(prob.q)
# 
# 
# =============================================================================
#%%

prob.Assembly_problem()
prob.H_matrix/=4
Lin_matrix=prob.reAssembly_matrices()
prob.Full_linear_matrix=Lin_matrix
prob.Solve_problem()

phi_bar=np.array([])
for i in range(prob.S):
    phi_bar=np.append(phi_bar, -prob.q[i]/prob.K[net.source_edge[i]]+1)
    

plt.plot(prob.q, label="q")
plt.plot(phi_bar, label="phi_bar")
plt.legend()

a=visualization_3D([0, L_vessel], 50, prob, 12, 0.5, np.array([0,L_vessel/2,0]))
