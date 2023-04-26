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



def get_three_slices(mesh_object, L):
    """Function that returns already reshaped, a field going through
    three equidistant slices perpendicular to three different axis. Therefore, 
    it returns 6 slices"""
    
    a=mesh_object.get_x_slice(L[0]/4),mesh_object.get_x_slice(L[0]/2),mesh_object.get_x_slice(3*L[0]/4)
    
    b=mesh_object.get_y_slice(L[0]/4),mesh_object.get_y_slice(L[0]/2),mesh_object.get_y_slice(3*L[0]/4)
    
    c=mesh_object.get_z_slice(L[0]/4),mesh_object.get_z_slice(L[0]/2),mesh_object.get_z_slice(3*L[0]/4)
    
    
    return a,b,c

#%% - This is the validation of the 1D transport eq without reaction
U = 0.1
D = 1
K=0.2
L = 10
R=L/50
cells_1D = 25

s = np.linspace(L/2/cells_1D, L-L/2/cells_1D, cells_1D)

Pe = U*L/D

analytical = (np.exp(Pe*s/L)-np.exp(Pe))/(1-np.exp(Pe))

plt.plot(s, analytical)


a, b = assemble_transport_1D(U, D, L/cells_1D, cells_1D)

A = sp.sparse.csc_matrix((a[0], (a[1], a[2])), shape=(cells_1D, cells_1D))

B = np.zeros(cells_1D)
B[0] = b[0]

sol = sp.sparse.linalg.spsolve(A, -B)

#sol = np.hstack((np.array((1)), sol, np.array((0))))



k=K/np.pi/R**2
A[np.arange(cells_1D), np.arange(cells_1D)]+=k*L/cells_1D #out flux
sol_reac = sp.sparse.linalg.spsolve(A, -B)

#sol = np.hstack((np.array((1)), sol, np.array((0))))

n=np.exp(np.sqrt(U**2+4*k*D)*L/2/D)
A=1/(1-n**2)
B=1/(1-1/n**2)

analytical_reac=np.exp(U*s/2/D)*(A*np.exp(np.sqrt(U**2+4*k*D)*s/2/D)+B*np.exp(-np.sqrt(U**2+4*k*D)*s/2/D))


# Create the figure and axes objects
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

# Plot the first subplot
axes[0].plot(s,sol, label='numerical')
axes[0].plot(s, analytical, label='analytical')
box_string='Pe={}'.format(U*L/D)
axes[0].text(2,0.50, box_string, fontsize = 40, 
 bbox = dict(facecolor = 'white', alpha = 0.5))
axes[0].set_title('Advection - diffusion')

# Plot the second subplot
# Plot the first subplot
axes[1].plot(s,sol_reac, label='numerical')
axes[1].plot(s, analytical_reac, label='analytical')
axes[1].legend()
box_string='Da={:.2f}'.format(k*L/D)
axes[1].text(2,0.50, box_string, fontsize = 40, 
 bbox = dict(facecolor = 'white', alpha = 0.5))
axes[1].set_title('Advection - diffusion - reaction')

# Set the overall title for the figure
fig.suptitle('Reference solution for weak couplings \n\n')

# Show the plot
plt.show()


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

from hybrid_set_up_noboundary import hybrid_set_up

from neighbourhood import get_neighbourhood, get_uncommon
#%



#BC_type=np.array(["Neumann", "Neumann", "Neumann","Neumann","Neumann","Neumann"])
BC_type=np.array(["Dirichlet", "Dirichlet", "Neumann","Neumann","Dirichlet","Dirichlet"])
BC_value=np.array([0,0,0,0,0,0])

cells=9
n=10
L=np.array([10,10,10])
mesh=cart_mesh_3D(L,cells)

mesh.assemble_boundary_vectors()

#%%

startVertex=np.array([0])
endVertex=np.array([1])
pos_vertex=np.array([[L[0]/2, 0.01, L[0]/2],[L[0]/2, L[0]-0.01,L[0]/2]])
vertex_to_edge=[[0],[1]]
diameters=np.array([2*R])
h=np.array([L[0]])/cells_1D

net=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h,1)
net.U=U
net.D=D
net.pos_arrays(mesh)

prob=hybrid_set_up(mesh, net, BC_type, BC_value, n, 1, np.zeros(len(diameters))+K)
mesh.get_ordered_connect_matrix()
prob.Assembly_problem()


#%% - Validation of the H, I, F matrices:
# First: Advection diffusion, no reaction
#Also, try to make sure all the h coefficients that have to be added are properly done and written in the notebook!
#In the notebook write what each assembly function does and where do you multiply by h and why!
I=prob.I_matrix
sol = sp.sparse.linalg.spsolve(I, -prob.III_ind_array)

plt.plot(net.pos_s[:,1],sol, label='hybrid')
plt.plot(s, analytical, label='analytical')
plt.legend()
plt.show()

#%% Advection - diffusion - reaction, weak couplings

new_E=sp.sparse.identity(len(net.pos_s))/(K)
H=prob.H_matrix

F=prob.F_matrix
ind=np.concatenate((np.zeros(len(net.pos_s)), prob.III_ind_array))

L1=sp.sparse.hstack((new_E,F))
L2=sp.sparse.hstack((H,I))

Li=sp.sparse.vstack((L1,L2))

sol=dir_solve(Li, -ind)


plt.plot(net.pos_s[:,1],sol[cells_1D:], label='hybrid reaction')
plt.plot(s, analytical_reac, label='analytical')
plt.legend()
plt.show()

#End of the validation of the 1D transport model, that is G, H, I matrices
