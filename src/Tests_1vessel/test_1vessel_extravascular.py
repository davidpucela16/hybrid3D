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


#%%
def visualization_3D(lim, res, prob, num_proc, vmax):
    
    
    
    a=(lim[1]-lim[0])*np.array([1,2,3])/4
    LIM_1=[lim[0], lim[0], lim[1], lim[1]]
    LIM_2=[lim[0], lim[1], lim[0], lim[1]]
    
    perp_x=np.zeros([3,4,3])
    for i in range(3):
        perp_x[i]=np.array([a[i]+np.zeros(4),LIM_1 , LIM_2]).T
        
    perp_y, perp_z=perp_x.copy(),perp_x.copy()
    
    perp_y[:,:,0]=perp_x[:,:,1]
    perp_y[:,:,1]=perp_x[:,:,0]
    
    perp_z[:,:,0]=perp_x[:,:,2]
    perp_z[:,:,2]=perp_x[:,:,0]
    
    # Create a figure with 3 rows and 3 columns
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18,18))
    
    # Set the titles for each row of subplots
    row_titles = ['X', 'Y', 'Z']
    
    # Set the titles for each individual subplot
    subplot_titles = ['x={:.2f}'.format(a[0]), 'x={:.2f}'.format(a[1]), 'x={:.2f}'.format(a[2]),
                      'y={:.2f}'.format(a[0]), 'y={:.2f}'.format(a[1]), 'y={:.2f}'.format(a[2]),
                      'z={:.2f}'.format(a[0]), 'z={:.2f}'.format(a[1]), 'z={:.2f}'.format(a[2]),]
    
    # Loop over each row of subplots
    for i, ax_row in enumerate(axs):
        # Set the title for this row of subplots
        ax_row[0].set_title(row_titles[i], fontsize=16)
        
        # Loop over each subplot in this row
        for j, ax in enumerate(ax_row):
            # Plot some data in this subplot
            x = [1, 2, 3]
            y = [1, 4, 9]
            
            if i==0: cor=perp_x
            if i==1: cor=perp_y
            if i==2: cor=perp_z
            
            a,b=prob.get_coord_reconst_chat(cor[j], res, num_processes=num_proc)
            im=ax.imshow(b.reshape(res,res), origin='lower', vmax=vmax, vmin=0, extent=[lim[0], lim[1], lim[0], lim[1]])
            # Set the title and y-axis label for this subplot
            ax.set_title(subplot_titles[i*3 + j], fontsize=14)
            if i==0: ax.set_ylabel('z'); ax.set_xlabel('y')
            if i==1: ax.set_ylabel('z'); ax.set_xlabel('x')
            if i==2: ax.set_ylabel('x'); ax.set_xlabel('y')
            
    # Set the x-axis label for the bottom row of subplots
    #axs[-1, 0].set_xlabel('X-axis', fontsize=12)
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    # Show the plot
    plt.show()
    return(perp_x, perp_y, perp_z)


#%% - This is the validation of the 1D transport eq without reaction
U = 0.1
D = 1
K=0.2
L = 10

alpha=50
R=L/alpha
cells_1D = 25

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



BC_type=np.array(["Neumann", "Neumann", "Neumann","Neumann","Neumann","Neumann"])
BC_type=np.array(["Dirichlet", "Dirichlet","Neumann","Neumann", "Dirichlet","Dirichlet"])
BC_value=np.array([0,0,0,0,0,0])

cells=10
n=2
L=np.array([240,960,240])
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

#%%
prob=hybrid_set_up(mesh, net, BC_type, BC_value, n, 1, np.zeros(len(diameters))+K)

mesh.get_ordered_connect_matrix()
prob.Assembly_problem()

#%% First let's try no source

sol=dir_solve(prob.A_matrix, -prob.I_ind_array)

sol=sol.reshape(mesh.cells)


#%% Second let's test the mass balance part of the matrix i.e. A and B. 

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

a,b,c=visualization_3D([0, L[0]], 50, prob, 12, 1)

#%%
res=100
_,middle=prob.get_coord_reconst_chat(b[1], res, num_processes=12)

plt.plot(middle.reshape(res,res)[50,:])
plt.plot(middle.reshape(res,res)[:,50])


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

#%%

sol=dir_solve(prob.Full_linear_matrix, -prob.Full_ind_array)

prob.s=sol[:prob.F]
prob.q=sol[prob.F:-prob.S]
prob.Cv=sol[-prob.S:]

plt.plot(prob.q)
plt.show()

#%%
a,b,c=visualization_3D([0, L[0]], 50, prob, 12, 0.05)



#%%  - Validation with 2D code

Lin_matrix=sp.sparse.hstack((prob.A_matrix, prob.B_matrix))
Lin_matrix=sp.sparse.vstack((Lin_matrix, sp.sparse.hstack((prob.D_matrix, prob.E_matrix))))

prob.C_v_array=np.ones(len(net.pos_s))



sol=dir_solve(Lin_matrix, -np.concatenate((prob.I_ind_array, prob.F_matrix.dot(prob.C_v_array))))

prob.s=sol[:prob.F]
prob.q=sol[prob.F:]
plt.plot(prob.q)

#%%
a,b,c=visualization_3D([0, L[0]], 50, prob, 12, 0.5)

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



