#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 17:12:55 2023

@author: pdavid

In this file Im gonna try to verify that the rapid term does what it is supposed to 
and try to perform the first simulations
"""


def get_three_slices():
    """Function that returns already reshaped, a field going through
    three equidistant slices perpendicular to two different axis. Therefore, 
    it returns 6 slices"""
    
    return



#%% - 
import os 
os.chdir(os.path.dirname(__file__))
from assembly import Assembly_diffusion_3D_interior, Assembly_diffusion_3D_boundaries
from mesh import cart_mesh_3D
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve as dir_solve
from scipy.sparse.linalg import bicg
import numpy as np
from mesh_1D import mesh_1D
from Green import get_source_potential
import pdb
#%%



BC_type=np.array(["Neumann", "Dirichlet", "Neumann","Neumann","Neumann","Neumann"])
BC_value=np.array([1,0,0,0,0,0])

sq_cells=20

L=np.array([10,10,10])

mesh=cart_mesh_3D(L,sq_cells)

a=Assembly_diffusion_3D_interior(mesh)
b=Assembly_diffusion_3D_boundaries(mesh, BC_type, BC_value)

A_matrix=csc_matrix((a[2], (a[0], a[1])), shape=(sq_cells**3,sq_cells**3)) + csc_matrix((b[2], (b[0], b[1])), shape=(sq_cells**3,sq_cells**3))



#%%

startVertex=np.array([0])
endVertex=np.array([1])
pos_vertex=np.array([[L[0]/2, L[0]/2,2*L[0]/5],[L[0]/2,L[0]/2,3*L[0]/5]])
vertex_to_edge=[[0],[0,1,2],[1,],[2]]
diameters=np.array([0.15])
h=np.array([0.05])


a=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h)
a.pos_arrays(mesh)

#%%

#%%
z=np.linspace(0,L[0],100)
y=np.zeros(100)+L[0]/2
x=np.zeros(100)+L[0]/2

array=np.zeros(100)

for i in np.arange(100)[::-1]:
    temp,_,_=a.kernel_point(np.array([x[i],y[i], z[i]]), a.uni_s_blocks, get_source_potential, np.array([1]), 1)
    array[i]=np.sum(temp)

