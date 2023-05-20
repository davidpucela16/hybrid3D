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


from assembly import AssemblyDiffusion3DInterior, AssemblyDiffusion3DBoundaries
from mesh import cart_mesh_3D
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve as dir_solve
from scipy.sparse.linalg import bicg
import numpy as np
import matplotlib.pyplot as plt

import math
from assembly_1D import FullAdvectionDiffusion1D
from mesh_1D import mesh_1D
from GreenFast import GetSourcePotential
import pdb

from hybridFast import hybrid_set_up, AssemblyBArraysFast
from post_processing import Visualization3D

from neighbourhood import GetNeighbourhood, GetUncommon

from numba.typed import List

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

cells_per_vessel=8
h_approx=L_vessel/cells_per_vessel

net=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h_approx,D_1D)
net.U=U
net.D=D_1D


#%% - Assembly of 3D data
cells_3D=5
n=3
L_3D=np.array([L_vessel, 2*L_vessel, L_vessel])
mesh=cart_mesh_3D(L_3D,cells_3D)

mesh.AssemblyBoundaryVectors()

net.PositionalArraysFast(mesh)

#%% - Set boundary conditions
BCs_1D=np.array([[0,1],
                 [2,0],
                 [3,0]])

BC_type=np.array(["Neumann", "Neumann", "Neumann","Neumann","Neumann","Neumann"])
BC_type=np.array(["Dirichlet", "Dirichlet","Neumann","Neumann", "Dirichlet","Dirichlet"])
BC_value=np.array([0,0,0,0,0,0])

#%% - Intra vascular problem
prob=hybrid_set_up(mesh, net, BC_type, BC_value,n,1, np.zeros(len(diameters))+K, BCs_1D)
mesh.GetOrderedConnectivityMatrix()

prob.AssemblyI()


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

plt.text(cells_per_vessel, 0.6, text_2,  fontsize=40,
        verticalalignment='top', bbox=props)

plt.text(2*cells_per_vessel, 0.6, text_3,  fontsize=40,
        verticalalignment='top', bbox=props)
plt.legend()
plt.show()

#%%

prob=hybrid_set_up(mesh, net, BC_type, BC_value,n,1, np.zeros(len(diameters))+K, BCs_1D)
mesh.GetOrderedConnectivityMatrix()
AssemblyBArraysFast(List(mesh.ordered_connect_matrix), mesh.size_mesh, n,1,
                                mesh.cells_x, mesh.cells_y, mesh.cells_z, mesh.pos_cells, mesh.h, 
                                net.s_blocks, net.tau, net.h, net.pos_s, net.source_edge)
mat_path=path+ '/matrices'

#%%

DEF_slow=prob.AssemblyDEF()

#%%

DEF_fast=prob.AssemblyDEFFast(mat_path)

#%%
def CompareRowByRow(A,B):
    for i in range(A.shape[0]):
        a=np.where(A[i]!=B[i])[0]
        if np.any(a):
            print("Different values on row{}".format(i))
            print(A[i,a])
            print(B[i,a])

#%% - Verify InterpolatePhiBarBlock -> evaluates each block separately
from Second_eq_functions import InterpolatePhiBarBlock, RetrieveBlockPhiBar
from small_functions import AppendSparse

for i in prob.mesh_1D.uni_s_blocks:
    #list_arrays=InterpolatePhiBarBlock(i,prob.n, prob.mesh_3D.cells_x, prob.mesh_3D.cells_y, prob.mesh_3D.cells_z, prob.mesh_3D.h,prob.mesh_3D.pos_cells,prob.mesh_1D.s_blocks, prob.mesh_1D.source_edge,prob.mesh_1D.tau, prob.mesh_1D.pos_s, prob.mesh_1D.h, prob.mesh_1D.R, 1, prob.mesh_1D.sources_per_block, prob.mesh_1D.quant_sources_per_block)
    
    D=np.zeros([3,0])
    Gij=np.zeros([3,0])
    #Manually obtaining the reference kernesl
    for j in np.where(prob.mesh_1D.s_blocks==i)[0]:
        kernel_s,col_s,kernel_q, col_q=prob.Interpolate(prob.mesh_1D.pos_s[j])
        D=AppendSparse(D, kernel_s,np.zeros(len(col_s))+j, col_s)
        Gij=AppendSparse(Gij, kernel_q,np.zeros(len(col_q))+j, col_q)
    D_ref=csc_matrix((D[0], (D[1], D[2])), shape=(prob.S, prob.mesh_3D.size_mesh))
    Gij_ref=csc_matrix((Gij[0], (Gij[1], Gij[2])), shape=(prob.S, prob.S)) 
    
    #Through the function
    list_arrays=RetrieveBlockPhiBar(mat_path, i)
    D_fast=csc_matrix((list_arrays[0], (list_arrays[1], list_arrays[2])), shape=(prob.S, prob.mesh_3D.size_mesh))
    Gij_fast=csc_matrix((list_arrays[3], (list_arrays[4], list_arrays[5])), shape=(prob.S, prob.S))
    
    
    for i in range(prob.S):
        a=np.where(D_ref.toarray()[i]!=D_fast.toarray()[i])[0]
        if np.any(a):
            print("s Different values on row{}".format(i))
            print(D_ref[i,a])
            print(D_fast[i,a])
            
    for i in range(prob.S):
        a=np.where(Gij_ref.toarray()[i]!=Gij_fast.toarray()[i])[0]
        if np.any(a):
            print("q Different values on row{}".format(i))
            print(Gij_ref[i,a])
            print(Gij_fast[i,a])





#%% - Now compare the full matrix row by row
D=np.zeros([3,0])
E=np.zeros([3,0])
c=0
for i in prob.mesh_1D.s_blocks:
    #list_arrays=InterpolatePhiBarBlock(i,prob.n, prob.mesh_3D.cells_x, prob.mesh_3D.cells_y, prob.mesh_3D.cells_z, prob.mesh_3D.h,prob.mesh_3D.pos_cells,prob.mesh_1D.s_blocks, prob.mesh_1D.source_edge,prob.mesh_1D.tau, prob.mesh_1D.pos_s, prob.mesh_1D.h, prob.mesh_1D.R, 1, prob.mesh_1D.sources_per_block, prob.mesh_1D.quant_sources_per_block)
    #Manually obtaining the reference kernesl
    kernel_s,col_s,kernel_q, col_q=prob.Interpolate(prob.mesh_1D.pos_s[c])
    D=AppendSparse(D, kernel_s,np.zeros(len(col_s))+c, col_s)
    E=AppendSparse(E, kernel_q,np.zeros(len(col_q))+c, col_q)
    c+=1

D_ref=csc_matrix((D[0], (D[1], D[2])), shape=(prob.S, prob.mesh_3D.size_mesh))
E_ref=csc_matrix((E[0], (E[1], E[2])), shape=(prob.S, prob.S)) 

#Now we compare the esimation of the s part of the interpolation phi bar
for i in range(prob.S):
    a=np.where(D_ref.toarray()[i]!=prob.phi_bar_s.toarray()[i])[0]
    if np.any(a):
        print("s Different values on row{}".format(i))
        print(D_ref[i,a])
        print(prob.phi_bar_s[i,a])

#%%        
print(np.allclose(E_ref.toarray(),prob.Gij.toarray(), atol=1e-18))



