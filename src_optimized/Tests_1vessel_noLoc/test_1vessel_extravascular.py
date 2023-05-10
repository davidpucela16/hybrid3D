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

cells=10
L=np.array([240,480,240])
mesh=cart_mesh_3D(L,cells)

mesh.assemble_boundary_vectors()

# - This is the validation of the 1D transport eq without reaction
U = 0.1
D = 1
K=1
L_vessel = L[1]

alpha=100
R=L_vessel/alpha
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

#%% First let's try no source

sol=dir_solve(prob.A_matrix, -prob.I_ind_array)

sol=sol.reshape(mesh.cells)


#%% Second let's test the mass balance part of the matrix i.e. A and B. 
#We don't really test anything here other than it looks right

prob.q=np.ones(len(net.pos_s))*3
#prob.q=np.arange(len(net.pos_s))/len(net.pos_s)
sol=dir_solve(prob.A_matrix, -prob.B_matrix.dot(prob.q)-prob.I_ind_array)

prob.s=sol

phi_bar=np.zeros(len(net.pos_s))

prob.C_v_array=np.random.random(len(net.pos_s))
for i in range(len(net.pos_s)):
    a,b,c,d,_,_=prob.interpolate(net.pos_s[i])
    kernel_s=csc_matrix((a,(np.zeros(len(b)), b)), shape=(1, prob.F))
    kernel_q=csc_matrix((c,(np.zeros(len(d)), d)), shape=(1, prob.S))
    
    phi_bar[i]=kernel_s.dot(prob.s) + kernel_q.dot(prob.q)

prob.C_v_array=prob.q/K+phi_bar

plt.plot(phi_bar, label='phi_bar')
plt.plot(prob.C_v_array, label='C_v')
plt.legend()
plt.show()
    
#%%

a=visualization_3D([0, L[0]], 50, prob, 12, 1)

#%%

plt.plot(a.data[5,25,:])
plt.plot(a.data[5,:,25])

#%%

sol=dir_solve(prob.Full_linear_matrix, -prob.Full_ind_array)

prob.s=sol[:prob.F]
prob.q=sol[prob.F:-prob.S]
prob.Cv=sol[-prob.S:]

plt.plot(prob.q)
plt.show()

#%%
a=visualization_3D([0, L[0]], 50, prob, 12, 0.05)


#%%
C_v_array=np.ones(len(net.pos_s)) #Though this neglects completely the contribution from the dipoles

L1=sp.sparse.hstack((prob.A_matrix,prob.B_matrix))
L2=sp.sparse.hstack((prob.D_matrix, prob.E_matrix))

Li=sp.sparse.vstack((L1,L2))

ind=np.concatenate((prob.I_ind_array, prob.F_matrix.dot(C_v_array)))

sol=dir_solve(Li, -ind)


plt.plot(net.pos_s[:,1],sol[-prob.S:], label='hybrid reaction')
plt.legend()
plt.show()

prob.s=sol[:-prob.S]
prob.q=sol[-prob.S:]

res=100
#lim=[mesh.h/2, L[0]-mesh.h/2]
lim=[0,L[0]]
mid=L[0]/2-mesh.h/2

cor=np.array([[lim[0],mid,lim[0]],[lim[0],mid,lim[1]],[lim[1],mid,lim[0]],[lim[1],mid,lim[1]]])

import time
start = time.time()
a,b=prob.get_coord_reconst_chat(cor, res, num_processes=12)
end = time.time()
print(end - start)

plt.imshow(b.reshape(100,100))
plt.colorbar()
plt.show()

plt.plot(b.reshape(res,res)[50])
plt.show()

#%%
cor=np.array([[mid,lim[0],lim[0]],[mid,lim[0],lim[1]],[mid,lim[1],lim[0]],[mid,lim[1],lim[1]]])

import time
start = time.time()
a,b=prob.get_coord_reconst_chat(cor, res, num_processes=12)
end = time.time()
print(end - start)

plt.imshow(b.reshape(100,100))
plt.colorbar()
plt.show()

plt.plot(b.reshape(res,res)[50])



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


# Metabolism Parameters
M = Da_t * D / L**2
phi_0 = 0.4
conver_residual = 5e-5
stabilization = 0.5

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


p = np.linspace(0, 1, 100)
if np.min(p - M * (1 - phi_0 / (phi_0 + p))) < 0:
    print("There is an error in the metabolism")


C_v_array = np.ones(S)

# =============================================================================
# BC_value = np.array([0, 0.2, 0, 0.2])
# BC_type = np.array(["Periodic", "Periodic", "Neumann", "Dirichlet"])
# =============================================================================
BC_value = np.array([0, 0,0,0])
BC_type = np.array(["Dirichlet", "Dirichlet", "Dirichlet", "Dirichlet"])

# What comparisons are we making
COMSOL_reference = 1
non_linear = 1
Peaceman_reference = 0


array_of_cells = np.arange(18) * 2 + 3
# array_of_cells=(np.arange(10))*4+3


#%%

t = Testing(
    pos_s, Rv, cells, L, K_eff, D, directness, ratio, C_v_array, BC_type, BC_value
)

s_Multi_cart_linear, q_Multi_linear = t.Multi()
#%%
Multi_rec_linear, _, _ = t.Reconstruct_Multi(0, 1)

#%%
c = 0
plt.plot(t.x_fine, t.array_phi_field_x_Multi[c], label="Multi")
plt.xlabel("x")
plt.legend()
plt.title("linear")
plt.show()

plt.plot(t.y_fine, t.array_phi_field_y_Multi[c], label="Multi")
plt.xlabel("y")
plt.legend()
plt.title("linear")
plt.show()

    


#%%



