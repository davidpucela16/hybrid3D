#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 18:47:23 2023

@author: pdavid
"""
import os 
path=os.path.dirname(__file__)
path_src=os.path.join(path, '../')
os.chdir(path_src)

import numpy as np
from assembly import assemble_transport_1D
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



#%% - 

from assembly import Assembly_diffusion_3D_interior, Assembly_diffusion_3D_boundaries
from mesh import cart_mesh_3D
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve as dir_solve
from scipy.sparse.linalg import bicg
import numpy as np
import matplotlib.pyplot as plt

import math

from mesh_1D import mesh_1D
from Green import get_source_potential
import pdb

from hybrid_set_up_noLoc import hybrid_set_up, visualization_3D

from neighbourhood import get_neighbourhood, get_uncommon
#%



BC_type=np.array(["Neumann", "Neumann", "Neumann","Neumann","Neumann","Neumann"])
BC_type=np.array(["Dirichlet", "Dirichlet","Neumann","Neumann", "Dirichlet","Dirichlet"])
BC_value=np.array([0,0,0,0,0,0])

cells=5
L=np.array([240,480,240])
mesh=cart_mesh_3D(L,cells)

mesh.assemble_boundary_vectors()

# - This is the validation of the 1D transport eq without reaction
U = 0.1
D = 1
K=1
L_vessel = L[1]

alpha=100
R=L[0]/alpha
cells_1D = 40



startVertex=np.array([0])
endVertex=np.array([1])
pos_vertex=np.array([[L[0]/2, 0.1, L[0]/2],[L[0]/2, L_vessel-0.1,L[0]/2]])
vertex_to_edge=[[0],[1]]
diameters=np.array([2*R])
h=np.array([L_vessel])/cells_1D

net=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h,1)
net.U=U
net.D=D
net.pos_arrays(mesh)

prob=hybrid_set_up(mesh, net, BC_type, BC_value, 1, np.zeros(len(diameters))+K)

mesh.get_ordered_connect_matrix()
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

res=50
#lim=[mesh.h/2, L[0]-mesh.h/2]
lim=[0,L[0]]
mid=L[0]/2-mesh.h/2



#%%  - Validation with 2D code

Lin_matrix=sp.sparse.hstack((prob.A_matrix, prob.B_matrix))
Lin_matrix=sp.sparse.vstack((Lin_matrix, sp.sparse.hstack((prob.D_matrix, prob.E_matrix))))

prob.C_v_array=np.ones(len(net.pos_s))



sol=dir_solve(Lin_matrix, -np.concatenate((prob.I_ind_array, prob.F_matrix.dot(prob.C_v_array))))

prob.s=sol[:prob.F]
prob.q=sol[prob.F:]
plt.plot(prob.q)

#%%
a=visualization_3D([0, L[0]], 50, prob, 12, 0.5)

#%%

import matplotlib.pylab as pylab
from hybrid2d.Reconstruction_functions import coarse_cell_center_rec
from hybrid2d.Small_functions import get_MRE, plot_sketch
from hybrid2d.Testing import (
    FEM_to_Cartesian,
    Testing,
    extract_COMSOL_data,
    save_csv,
    save_var_name_csv,
)

from hybrid2d.Module_Coupling import assemble_SS_2D_FD

# 0-Set up the sources
# 1-Set up the domain

Da_t = 10
D = 1
K0 = K
L = 240

cells = 5
h_coarse = L / cells

# Definition of the Cartesian Grid
x_coarse = np.linspace(h_coarse / 2, L - h_coarse / 2, int(np.around(L / h_coarse)))
y_coarse = x_coarse

# V-chapeau definition
directness = 2
print("directness=", directness)

S = 1
Rv = L / alpha + np.zeros(S)
pos_s = np.array([[0.5, 0.5]]) * L

# ratio=int(40/cells)*2
ratio = int(100 * h_coarse // L / 4) * 2

print("h coarse:", h_coarse)
K_eff = K0 / (np.pi * Rv**2)


C_v_array = np.ones(S)

# =============================================================================
# BC_value = np.array([0, 0.2, 0, 0.2])
# BC_type = np.array(["Periodic", "Periodic", "Neumann", "Dirichlet"])
# =============================================================================
BC_value = np.array([0, 0,0,0])
BC_type = np.array(["Dirichlet", "Dirichlet", "Dirichlet", "Dirichlet"])

t = Testing(
    pos_s, Rv, cells, L, K_eff, D, directness, ratio, C_v_array, BC_type, BC_value
)

s_Multi_cart_linear, q_Multi_linear = t.Multi()

Multi_rec_linear, _, _ = t.Reconstruct_Multi(0, 1)

c = 0
plt.plot(t.x_fine, t.array_phi_field_x_Multi[c], label="Multi")
plt.xlabel("x")
plt.legend()
plt.title("linear 2D reference")
plt.show()

plt.plot(t.y_fine, t.array_phi_field_y_Multi[c], label="Multi")
plt.xlabel("y")
plt.legend()
plt.title("linear 2D reference")
plt.show()

    


#%%
plt.plot(np.linspace(0,L*(1-1/res), res)+L[0]/2/res,a.data[5,int(res/2),:])
plt.plot(t.x_fine, t.array_phi_field_x_Multi[c], label="Multi")


