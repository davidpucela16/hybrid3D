#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 17:21:46 2023

@author: pdavid
"""

import numpy as np 
import matplotlib.pyplot as plt



def grad_Green(x, a, b):
    rb=x-b
    ra=x-a
    L=np.linalg.norm(b-a)
    tau=(b-a)/L
    
    first=(rb/np.linalg.norm(rb)-tau)/(L+rb+np.dot(tau,ra))
    second=(ra/np.linalg.norm(ra)-tau)/(ra+np.dot(tau,ra))
    
    return(first-second)

def grad_Green_approx(x,a,b):
    L=np.linalg.norm(b-a)
    c=(b-a)/2
    
    return()

x=-np.linspace(0,1,100)
y=np.linspace(0,1,100)

a=np.array([0,-0.1])
b=np.array([1,-0.1])

A=np.empty((100,100))

for i in range(100):
    for j in range(100):
        A[i,j]=np.linalg.norm(grad_Green(np.array([i,j]), a, b))
        
plt.imshow(A, origin='lower', vmin=0, vmax=0.01)
plt.colorbar()


#%% - 
import os 
os.chdir(os.path.dirname(__file__))
from assembly import Assembly_diffusion_3D_interior, Assembly_diffusion_3D_boundaries
from mesh import cart_mesh_3D
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve as dir_solve
from scipy.sparse.linalg import bicg

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

sol=dir_solve(A_matrix, b[-1])
#%%
sol=bicg(A_matrix, b[-1])
#%%
pp=mesh.get_x_slice(L[0]/2).astype(int)
qq=mesh.get_y_slice(L[0]/2).astype(int)
rr=mesh.get_z_slice(L[0]/2).astype(int)

#%%
plt.imshow(sol[pp].reshape(sq_cells, sq_cells), origin='lower')
plt.colorbar()
plt.show
#%%
plt.imshow(sol[qq].reshape(sq_cells, sq_cells), origin='lower')
plt.colorbar()
plt.show
#%%
plt.imshow(sol[rr].reshape(sq_cells, sq_cells), origin='lower')
plt.colorbar()
plt.show



#%% - Some tests regarding single layer potential 
import os 
directory_script = os.path.dirname(__file__)
os.chdir(directory_script)
from Green import Green_line_orig, Green_line
from mesh import cart_mesh_3D

L=10


a=cart_mesh_3D(np.array([L,L,L]), 50)
a.assemble_boundary_vectors()

ra=np.array([5.1,4.1,5.1])
rb=np.array([5.1,6.1,5.1])
#%%
A=np.array([])
for k in range(a.size_mesh):
    A=np.append(A, Green_line_orig((a.get_coords(k),ra,rb),1 ))
    
#%%


plt.imshow(A[a.get_z_slice(5)].reshape(50,50))
plt.colorbar()

#%%

from Green import Sampson_surface, grad_Green_3D
import pdb 

def test_grad_Green_3D():
    center=np.zeros(3)
    h=2
    cube=np.array([[0,0,h/2],
                   [0,0,-h/2],
                   [0,h/2,0],
                   [0,-h/2,0],
                   [h/2,0,0],
                   [-h/2,0,0],
                   ])
    D=3
    total=0
    total_2=0
    for i in cube:
        normal=i*2/h
        total+=np.dot(Sampson_surface((center,), grad_Green_3D, i, h, normal, D), normal)*D*h**2
        total_2+=np.dot(grad_Green_3D((i,center),D), normal)*D*h**2
    return total,total_2




print(test_grad_Green_3D())

#%%
from Green import get_source_potential


get_source_potential(tup_args, x,D)





#%%




