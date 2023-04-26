#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 17:12:55 2023

@author: pdavid

In this file Im gonna try to verify that the rapid term does what it is supposed to 
and try to perform the first simulations
"""


def get_three_slices(mesh_object, L):
    """Function that returns already reshaped, a field going through
    three equidistant slices perpendicular to three different axis. Therefore, 
    it returns 6 slices"""
    
    a=mesh_object.get_x_slice(L[0]/4),mesh_object.get_x_slice(L[0]/2),mesh_object.get_x_slice(3*L[0]/4)
    
    b=mesh_object.get_y_slice(L[0]/4),mesh_object.get_y_slice(L[0]/2),mesh_object.get_y_slice(3*L[0]/4)
    
    c=mesh_object.get_z_slice(L[0]/4),mesh_object.get_z_slice(L[0]/2),mesh_object.get_z_slice(3*L[0]/4)
    
    
    return a,b,c




#%% - 
import os 
os.chdir(os.path.dirname(__file__))
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

from hybrid_set_up_noboundary import hybrid_set_up

from neighbourhood import get_neighbourhood, get_uncommon
#%



#BC_type=np.array(["Neumann", "Neumann", "Neumann","Neumann","Neumann","Neumann"])
BC_type=np.array(["Dirichlet", "Dirichlet", "Dirichlet","Dirichlet","Dirichlet","Dirichlet"])
BC_value=np.array([0,0,0,0,0,0])

cells=5
n=1
L=np.array([9,9,9])
mesh=cart_mesh_3D(L,cells)

mesh.assemble_boundary_vectors()

#%

startVertex=np.array([0,2])
endVertex=np.array([1,3])
pos_vertex=np.array([[L[0]/2, L[0]/2, 0.1],[L[0]/2,L[0]/2, L[0]-0.1],[L[0]/3, L[0]/3, 0.1],[L[0]/3,L[0]/3, L[0]-0.1]])
vertex_to_edge=[[0],[0],[1],[1]]
diameters=np.array([0.1, 0.2])
h=np.array([0.1])

a=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h,1)
a.pos_arrays(mesh)

net=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h,1)
net.pos_arrays(mesh)
#%

prob=hybrid_set_up(mesh, a, BC_type, BC_value, n, 1, np.zeros(len(diameters))+math.inf)
mesh.get_ordered_connect_matrix()
prob.Assembly_problem()


#%
q=np.arange(len(a.pos_s))[::-1]/len(a.pos_s)
A=prob.A_matrix.toarray()
sol=np.linalg.solve(A, -prob.I_ind_array-prob.B_matrix.dot(q))



#%%
res=100
prob.s=sol
prob.q=q
one=8
zero=1
mid=4.5

cor=np.array([[zero,mid,zero],[zero,mid,one],[one,mid,zero],[one,mid,one]])

import time
start = time.time()
a,b=prob.get_coord_reconst_chat(cor, res, num_processes=12)
end = time.time()
print(end - start)

k,l,m=get_three_slices(mesh, L)
plt.imshow(sol[k[1]].reshape(cells, cells), origin="lower") 
plt.title("slow")
plt.colorbar()
plt.show()

b=b.reshape(res,res)
plt.imshow(b)
plt.title("Fine reconstruction Y plane")
plt.colorbar()
plt.show()


start = time.time()
a,b=prob.get_coord_reconst( cor, res)
end = time.time()
print(end - start)

#%%

res=50
prob.s=sol
prob.q=q
one=8.5
zero=0.5
mid=4.5
cor=np.array([[zero,mid,zero],[zero,mid,one],[one,mid,zero],[one,mid,one]])
c,d=prob.get_coord_reconst_chat(cor, res, num_processes=12)
#%
d=d.reshape(res,res)
plt.imshow(d)
plt.title("Fine reconstruction Z plane")
plt.colorbar()

#%%
full_reconst=True
rec_3D=np.empty(res**3)
g=0

x=np.linspace(L[0]/2/res, L[0]*(1-1/2/res), res)
y=np.linspace(L[1]/2/res, L[1]*(1-1/2/res), res)
z=np.linspace(L[2]/2/res, L[2]*(1-1/2/res), res)

if full_reconst:
    for i in x:
        for j in y:
            for k in z:
                if not g%100: print(g)
                a,b,c,d,e,f=prob.interpolate(np.array([i,j,k]))
                rec_3D[g]=a.dot(sol[b])+c.dot(q[d])
                g+=1             

#%%
from pyevtk.hl import gridToVTK
gridToVTK("./visualization/test", x,y,z, cellData = {'julia': rec_3D})


#%% - Im gonna interpolate in a finer grid, 1D reconstruction


y=np.linspace(0+L[0]/res/2,L[0]*(2*res-1)/2/res,res)
temp=np.zeros(res)+L[0]/2

array=np.array([])
for i in range(res):
    crds=np.array([y[i],4.5,4.5])
    a,b,c,d,e,f=prob.interpolate(crds)
    
    array=np.append(array,a.dot(sol[b])+c.dot(q[d]))
    #array=np.append(array,c.dot(q[d]))
    #array=np.append(array,a.dot(sol[b]))
plt.plot(array)

#%%
#%% - Test 1

A=prob.A_matrix.toarray()

# =============================================================================
# for i in mesh.full_boundary:
#     A[i,:]=0
#     A[i,i]=1
# =============================================================================

B=np.zeros(cells**3)
for i in a.s_blocks:
    B[i]+=1
    
sol=np.linalg.solve(A, -B)



#%% - First let's try the Diffusion matrix


k,l,m=get_three_slices(mesh, L)

plt.imshow(sol[m[1]].reshape(cells, cells), origin="lower"); plt.colorbar()


#%%
cc=0
for i in get_neighbourhood(n, 10,10,10, a.s_blocks[0]):
    print()
    print(prob.B_matrix[i])
    cc+=np.sum(prob.B_matrix[i].toarray())

#%%

for i in get_uncommon(get_neighbourhood(n+1, 10,10,10, a.s_blocks[0]),get_neighbourhood(1, 10,10,10, a.s_blocks[0])):
    print()
    print(prob.B_matrix[i])
    cc+=np.sum(prob.B_matrix[i].toarray())

#%%
from hybrid_set_up_noboundary import hybrid_set_up
prob_2=hybrid_set_up(mesh, net, BC_type, BC_value, n, 1, np.array([math.inf]))
mesh.get_ordered_connect_matrix()
a,b=prob_2.rec_along_mesh("y", L[0]/2, sol, q, np.zeros(2))
plt.imshow(a)
plt.colorbar()



#%%
for i in get_neighbourhood(1, 10,10,10, a.s_blocks[0]):
    print()
# =============================================================================
#     print(np.nonzero(A[i]))
#     print(np.dot(A[i], sol))
#     print(prob.B_matrix.dot(q)[i])
#     print(np.dot(Up[i], sol))
# =============================================================================
    #print(A[i,np.nonzero(A[i])])


#%% - Reconstruction
rec=np.array([])
for k in range(mesh.size_mesh):
    kernel=prob.interpolate(mesh.get_coords(k))
    
    m1=csc_matrix((kernel[2],(np.zeros(len(kernel[3])), kernel[3])), shape=(1, len(q)) )
    ms=csc_matrix((kernel[0],(np.zeros(len(kernel[1])), kernel[1])), shape=(1, mesh.size_mesh ))
    rec=np.append(rec, m1.dot(q)+ms.dot(sol))
    
#%%
from neighbourhood import get_neighbourhood, get_uncommon
s_neigh=get_neighbourhood(1, mesh.cells_x, mesh.cells_y, mesh.cells_z, a.uni_s_blocks[0])
of_neigh=np.array([])
for k in s_neigh:
    
    of_neigh=np.concatenate((of_neigh,get_uncommon(get_neighbourhood(1, mesh.cells_x, mesh.cells_y, mesh.cells_z, k), s_neigh)))
of_neigh=np.unique(of_neigh).astype(int)

#%%

plt.imshow(rec[m[1]].reshape(sq_cells, sq_cells), origin="lower"); plt.colorbar()

#%%
plt.imshow(sol[l[1]].reshape(sq_cells, sq_cells), origin="lower"); plt.colorbar()

#%%
y=np.linspace(0,L[0],1000)
z=np.zeros(1000)+2*L[0]/5+0.1
x=np.zeros(1000)+L[0]/2

array=np.zeros(1000)

for i in np.arange(1000):
    _,temp,_=a.kernel_point(np.array([x[i],y[i], z[i]]), a.uni_s_blocks, get_source_potential, np.array([math.inf]), 1)
    array[i]=np.sum(temp)





#%%

prob=hybrid_set_up(mesh, a, BC_type, BC_value, 1, 1, np.array([math.inf]))
mesh.get_ordered_connect_matrix()
B_matrix=prob.Assembly_B()


#%%



