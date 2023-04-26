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
n=6
L=np.array([9,9,9])
mesh=cart_mesh_3D(L,cells)
D_intra=2

mesh.assemble_boundary_vectors()

#%

startVertex=np.array([0])
endVertex=np.array([1])
pos_vertex=np.array([[L[0]/2, L[0]/2, 0.1],[L[0]/2,L[0]/2, L[0]-0.1]])
vertex_to_edge=[[0],[0]]
diameters=np.array([0.1])
h=np.array([0.2])

a=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h,D_intra)
a.pos_arrays(mesh)

net=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h,D_intra)
net.pos_arrays(mesh)
#%

prob=hybrid_set_up(mesh, a, BC_type, BC_value, n, 1, np.zeros(len(diameters))+math.inf)
prob.Assembly_problem()

#%
q=np.arange(len(a.pos_s))/len(a.pos_s)
q=np.ones(len(a.pos_s))
A=prob.A_matrix.toarray()
sol=np.linalg.solve(A, -prob.I_ind_array-prob.B_matrix.dot(q))
prob.s=sol
prob.q=q

#%%

sol=dir_solve(prob.Full_linear_matrix,prob.Full_ind_array)
prob.s=sol[:prob.F]
prob.q=sol[prob.F:prob.F+prob.S]
prob.Cv=sol[-prob.S:]

#%%
array=np.array(())

for i in range(len(prob.mesh_1D.pos_s)):
    value=prob.D_matrix[i].dot(prob.s)+prob.E_matrix[i].dot(prob.q)
    array=np.append(array,value)
    
plt.plot(array)

#%%
array=np.array(())

for i in range(len(prob.mesh_1D.pos_s)):
    value=prob.F_matrix[i].dot(np.ones(len(prob.mesh_1D.pos_s)))
    array=np.append(array,value)
    
plt.plot(array)


#%%
from hybrid_set_up_noboundary import interpolate_helper
array=np.array(())

for i in prob.mesh_1D.pos_s:
    q,C,sources=a.kernel_point(i, np.arange(mesh.size_mesh), get_source_potential, prob.K,1)
    array=np.append(array,np.sum(q))

plt.plot(array)
    
#%%

res=50

one=8.5
zero=0.5
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


#%%

res=25
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
full_reconst=False
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


#%%
from assembly import assemble_transport_1D


U = 1
D = 1


x = np.linspace(L/2/len(a.pos_s), L-L/2/len(a.pos_s), len(a.pos_s))

Pe = U*L/D

analytical = (np.exp(Pe*x/L)-np.exp(Pe))/(1-np.exp(Pe))

plt.plot(x, analytical)
# %%

aa, bb = assemble_transport_1D(U, D, L/len(a.pos_s), len(a.pos_s))

I = csc_matrix((aa[0], (aa[1], aa[2])), shape=(len(a.pos_s), len(a.pos_s)))

B = np.zeros(len(a.pos_s))
B[0] = -bb[0]

sol = np.linalg.solve(I.toarray(), B)

#sol = np.hstack((np.array((1)), sol, np.array((0))))

plt.plot(x,sol, label='numerical')
plt.plot(x, analytical, label='analytical')
plt.legend()

# plt.plot(x,sol)

#%%

H=-np.identity(len(a.pos_s))/(np.pi*a.R[0]**2)

sol = np.linalg.solve(I.toarray(), B+np.dot(H, np.zeros(len(a.pos_s))+0.0001))

#sol = np.hstack((np.array((1)), sol, np.array((0))))

plt.plot(x,sol, label='numerical')
plt.plot(x, analytical, label='analytical')
plt.legend()

#%%

















