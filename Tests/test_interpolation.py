#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 15:16:31 2023

@author: pdavid
"""


from mesh_1D import mesh_1D
from mesh import cart_mesh_3D
import pdb
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
from neighbourhood import get_neighbourhood
L=5
D=1
mesh=cart_mesh_3D(np.array([L,L,3*L/5]), 5)
mesh.assemble_boundary_vectors()


#%%
def get_8_closest(h,step_x, step_y,x):
    """This function returns the (8) closest Cartesian grid centers
    - x is the position (array of length=3)
    - h is the discretization size (float)
    """
    
    arr=np.array([])
    for i in x: #Loop through each of the axis 
        b=int(int(i//(h/2))%2)*2-1
        arr=np.append(arr, b)
    ID=mesh.get_id(x) #ID of the containing block
    
    blocks=np.array([ID, ID+arr[2]], dtype=int)
    blocks=np.append(blocks, blocks+arr[1]*step_y)
    blocks=np.append(blocks, blocks+arr[0]*step_x)
    
    return(np.sort(blocks))

#%% - Check get_neighbourhood
x=np.array([0.02632653, 0.1   ,0.7       ])
p=mesh.get_id(x)
blocks=get_neighbourhood(1, mesh.cells_x, mesh.cells_y, mesh.cells_z, p)
coords=np.zeros((0,3))
for i in blocks:
    coords=np.vstack((coords,mesh.get_coords(i)))
    
plt.scatter(x[1], x[2], s=100)
plt.scatter(coords[:,1], coords[:,2])
plt.xlabel('y')
plt.ylabel('z')




#%%

startVertex=np.array([0])
endVertex=np.array([1])
pos_vertex=np.array([[L/2, L/2,0],[L/2,L/2,3*L/5]])
vertex_to_edge=[[0],[0,1,2],[1,],[2]]
diameters=np.array([0.15])
h=np.array([0.05])


a=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h)
a.pos_arrays(mesh)
#%%
from hybrid_set_up_noboundary import hybrid_set_up

BC_type=np.array(["Dirichlet", "Dirichlet","Dirichlet", "Dirichlet","Dirichlet","Dirichlet"])
BC_value=np.zeros(6)

b=hybrid_set_up(mesh, a, BC_type, BC_value, 1, 1,np.ones(len(a.pos_s)))

#%%



row_s=np.array([], dtype=int)
col=np.array([], dtype=int)
col_s=np.array([], dtype=int)
row_q=np.array([], dtype=int)

kernel=np.array([])
kernel_s=np.array([])
c=0
test_ax=np.array([])
#for j in z:
for x in np.linspace(0.01,4.99,50):
    for y in np.linspace(0.01,4.99,50):
        pos=np.array([x,y,0.7])
        t,tt,d,e,_,_=b.interpolate(pos)
        row_s=np.append(row_s, c+np.zeros(len(t)))
        col_s=np.append(col_s, tt)
        kernel_s=np.append(kernel_s, t)
        
        row_q=np.append(row_q, c+np.zeros(len(d)))
        col=np.append(col, e)
        kernel=np.append(kernel, d)
        test_ax=np.append(test_ax, x**2+y)
        c+=1

#%%

q=sps.csc_matrix((kernel, (row_q, col)), shape=(c, 1+int(np.max(col))))
value=q.dot(np.ones(1+int(np.max(col))))

s=sps.csc_matrix((kernel_s, (row_s, col_s)), shape=(c, c))
value_s=s.dot(np.arange(c)/(c)*10)

plt.imshow((value+value_s).reshape(50,50),extent=(0,5,0,5) ,origin='lower') 
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()

for i in mesh.x:
    plt.axvline(i+mesh.h/2)
    plt.axhline(i+mesh.h/2)


#%%




#%%
row_s=np.array([], dtype=int)
col=np.array([], dtype=int)
kernel=np.array([])
c=0
test_ax=np.array([])
#for j in z:
for z in np.linspace(0.01,3*L/5-0.0001,30):
    for y in np.linspace(0.01,4.99,50):
        pos=np.array([2.4,y,z])
        t,tt,d,e,_,_=b.interpolate(pos)
        row_s=np.append(row_s, c+np.zeros(len(d)))
        col=np.append(col, e)
        kernel=np.append(kernel, d)
        test_ax=np.append(test_ax, x**2+y)
        c+=1

#%%

q=sps.csc_matrix((kernel, (row_s, col)), shape=(c, 1+int(np.max(col))))
value=q.dot(np.ones(1+int(np.max(col))))

plt.imshow(value.reshape(30,50),extent=(0,5,0,3) ,origin='lower') 
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()

for i in mesh.x:
    plt.axvline(i+mesh.h/2)
    plt.axhline(i+mesh.h/2)

