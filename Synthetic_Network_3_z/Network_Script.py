#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:04:46 2023

@author: pdavid
"""

import os 
path_current_file=os.path.dirname(__file__)
path_network="/home/pdavid/Bureau/PhD/Network_Flo/Synthetic_ROIs_300x300x300"

import pandas as pd
import numpy as np 

import pdb 
from numba import njit

import scipy as sp
#%

import matplotlib.pyplot as plt
#%
import os
os.chdir('/home/pdavid/Bureau/Code/hybrid3d/src_final')
from mesh_1D import mesh_1D
from hybridFast import hybrid_set_up
from mesh import cart_mesh_3D
from post_processing import Visualization3D
from assembly_1D import AssembleVertexToEdge, PreProcessingNetwork, CheckLocalConservativenessFlowRate, CheckLocalConservativenessVelocity
from PrePostTemp import SplitFile, SetArtificialBCs, ClassifyVertices, get_phi_bar, Get9Lines, VisualizationTool


mat_path="/home/pdavid/Bureau/Code/hybrid3d/Synthetic_Network_3_z/F20"
output_dir_network="/home/pdavid/Bureau/Code/hybrid3d/Synthetic_Network_3_z/divided_files"
filename=os.path.join(path_network,"Rea1_synthetic_z.Smt.SptGraph.am")



#output_dir_network = '/home/pdavid/Bureau/PhD/Network_Flo/All_files/Split/'  # Specify the output directory here
os.makedirs(output_dir_network, exist_ok=True)  # Create the output directory if it doesn't exist


output_files = SplitFile(filename, output_dir_network)

print("Split files:")
for file in output_files:
    print(file)

#%
df = pd.read_csv(output_dir_network + '/output_0.txt', skiprows=1, sep="\s+", names=["x", "y", "z"])
with open(output_dir_network + '/output_0.txt', 'r') as file:
    # Read the first line
    output_zero = file.readline()
pos_vertex=df.values+2.5

df = pd.read_csv(output_dir_network + '/output_1.txt', skiprows=1, sep="\s+", names=["init", "end"])
with open(output_dir_network + '/output_1.txt', 'r') as file:
    # Read the first line
    output_one = file.readline()
edges=df.values

df = pd.read_csv(output_dir_network + '/output_2.txt', skiprows=1, sep="\s+", names=["cells_per_segment"])
with open(output_dir_network + '/output_2.txt', 'r') as file:
    # Read the first line
    output_two = file.readline()
cells_per_segment=np.ndarray.flatten(df.values)

df = pd.read_csv(output_dir_network + '/output_3.txt', skiprows=1, sep="\s+", names=["x", "y", "z"])
with open(output_dir_network + '/output_3.txt', 'r') as file:
    # Read the first line
    output_three= file.readline()
points=np.ndarray.flatten(df.values)

df = pd.read_csv(output_dir_network + '/output_4.txt', skiprows=1, sep="\s+", names=["length"])
with open(output_dir_network + '/output_4.txt', 'r') as file:
    # Read the first line
    output_four= file.readline()
diameters=np.ndarray.flatten(df.values)

diameters=diameters[np.arange(len(edges))*2]

df = pd.read_csv(output_dir_network + '/output_5.txt', skiprows=1, sep="\s+", names=["flow_rate"])
with open(output_dir_network + '/output_5.txt', 'r') as file:
    # Read the first line
    output_five= file.readline()
Pressure=np.ndarray.flatten(df.values)

df = pd.read_csv(output_dir_network + '/output_6.txt', skiprows=1, sep="\s+", names=["flow_rate"])
with open(output_dir_network + '/output_6.txt', 'r') as file:
    # Read the first line
    output_six= file.readline()
Flow_rate=np.ndarray.flatten(df.values)*1e3



#%%
cells_3D=20
K=np.average(diameters)/np.ndarray.flatten(diameters)
#The flow rate is given in nl/s
U = 4*Flow_rate/np.pi/diameters**2*1e9 #To convert to speed in micrometer/second

startVertex=edges[:,0].copy()
endVertex=edges[:,1].copy()
vertex_to_edge=AssembleVertexToEdge(pos_vertex, edges)

#%% - Pre processing flow_rate

########################################################################
#   THIS IS A CRUCIAL OPERATION
########################################################################

for i in range(len(edges)):
    gradient=Pressure[2*i+1] - Pressure[2*i]
    if gradient<0:
        edges[i,0]=endVertex[i]
        edges[i,1]=startVertex[i]    
    
startVertex=edges[:,0]
endVertex=edges[:,1]


CheckLocalConservativenessFlowRate(startVertex,endVertex, vertex_to_edge, Flow_rate)
# =============================================================================
# flow_rate=Flow_rate*1e6
# vertex=0
# for i in vertex_to_edge:
#     
#     if len(i)>2:
#         a=np.zeros(len(i)) #to store whether the edges are entering or exiting
#         c=0
#         for j in i: #Goes through each edge of the bifurcation
#             a[c]=1 if vertex==startVertex[j] else -1  #Vessel exiting
#             c+=1
#             
#         print(np.dot(flow_rate[i], a))
#         if np.dot(flow_rate[i], a)>1: pdb.set_trace()
#     vertex+=1
# =============================================================================

#%%
L_3D=np.array([305,305,305])

#Set artificial BCs for the network 
BCs_1D=SetArtificialBCs(vertex_to_edge, 1,0, startVertex, endVertex)

#BC_type=np.array(["Dirichlet", "Neumann","Neumann","Neumann","Neumann","Neumann"])
#BC_type=np.array(["Dirichlet","Dirichlet","Dirichlet","Dirichlet","Dirichlet","Dirichlet"])
BC_type=np.array(["Neumann", "Neumann", "Neumann","Neumann","Neumann","Neumann"])
BC_value=np.array([0,0,0,0,0,0])

net=mesh_1D( startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, np.average(diameters)/2,np.average(U))
net.U=np.ndarray.flatten(U)

mesh=cart_mesh_3D(L_3D,cells_3D)
net.PositionalArraysFast(mesh)

cumulative_flow=np.zeros(3)
for i in range(len(Flow_rate)):
    cumulative_flow+=Flow_rate[i]*net.tau[i]
    
    


#%%

n=2
prob=hybrid_set_up(mesh, net, BC_type, BC_value,n,1, K, BCs_1D)
prob.phi_bar_bool=False
prob.B_assembly_bool=False
prob.I_assembly_bool=True
#%%

prob.AssemblyProblem(mat_path)
#M_D=0.001
M_D=0
prob.Full_ind_array[:cells_3D**2]-=M_D*mesh.h**3
print("If all BCs are newton the sum of all coefficients divided by the length of the network should be close to 1", np.sum(prob.B_matrix.toarray())/np.sum(net.L))



#%% - Constant concentration
constant_concentration=1
if constant_concentration:
    from scipy.sparse.linalg import spsolve as dir_solve
    Up=sp.sparse.hstack((prob.A_matrix, prob.B_matrix))
    Down=sp.sparse.hstack((prob.D_matrix, prob.Gij + prob.q_portion))
    
    prob.Cv=np.ones(prob.S)
    ind_array=prob.Full_ind_array.copy()
    ind_array=ind_array[:mesh.size_mesh+prob.S]
    ind_array[-prob.S:]+=prob.F_matrix.dot(prob.Cv)
    
    L=sp.sparse.vstack((Up, Down))
    
    sol=dir_solve(L,-ind_array)
    
    prob.q=sol[-prob.S:]
    prob.s=sol[:-prob.S]

#%%
# =============================================================================
# if not prob.B_assembly_bool and os.path.exists(mat_path + "/q.npy"): 
#     np.load( mat_path + "/q.npy",prob.q)
#     np.load( mat_path + "/s.npy",prob.s)
#     np.load( mat_path + "/Cv.npy",prob.Cv)
# else:
#     prob.SolveProblem()
#     np.save( mat_path + "/q",prob.q)
#     np.save( mat_path + "/s",prob.s)
#     np.save( mat_path + "/Cv",prob.Cv)
# =============================================================================

#prob.SolveProblem()
    
#%%


#post=Visualization3D((mesh.h/2,L_3D[0]/2-mesh.h/2), 50, prob, 12, 0.5)

#%%



phi_bar=get_phi_bar(mat_path, prob.s, prob.q)

#%%
from post_processing import *




phi,crds, others, points_a, points_b=Get9Lines(0, 200, L_3D, prob)
for k in range(9):
    plt.plot(phi[k], label=str(np.array(["x","y","z"])[others]) + "={:.1f}, {:.1f}".format(points_a[k//3], points_b[k%3]))
plt.xlabel(np.array(["x","y","z"])[i])
plt.legend()
plt.show()
pdb.set_trace()
#%%

for i in range(3):
    phi,crds, others, points_a, points_b=Get9Lines(i, 200, L_3D, prob)
    for k in range(9):
        plt.plot(phi[k], label=str(np.array(["x","y","z"])[others]) + "={:.1f}, {:.1f}".format(points_a[k//3], points_b[k%3]))
    plt.xlabel(np.array(["x","y","z"])[i])
    plt.legend()
    plt.show()

#%% - Testing time!
p=np.argsort(net.pos_s[:,0])
#prob.q[p+10]=-1
Up=sp.sparse.hstack((prob.A_matrix, prob.B_matrix))
Down=sp.sparse.hstack((prob.D_matrix, prob.Gij + prob.q_portion))

prob.Cv=np.zeros(prob.S)
prob.Cv[p[-1000:]]=1
ind_array=prob.Full_ind_array.copy()
ind_array=ind_array[:mesh.size_mesh+prob.S]
ind_array[-prob.S:]+=prob.F_matrix.dot(prob.Cv)

L=sp.sparse.vstack((Up, Down))

sol=dir_solve(L,-ind_array)
prob.q=sol[-prob.S:]
prob.s=sol[:-prob.S]
for i in range(3):
    phi,crds, others, points_a, points_b=Get9Lines(i, 200, L_3D, prob)
    for k in range(9):
        plt.plot(phi[k], label=str(np.array(["x","y","z"])[others]) + "={:.1f}, {:.1f}".format(points_a[k//3], points_b[k%3]))
    plt.xlabel(np.array(["x","y","z"])[i])
    plt.legend()
    plt.show()
    
#%%
p=np.argsort(net.pos_s[:,0])
#prob.q[p+10]=-1


sol=dir_solve(prob.Full_linear_matrix,-prob.Full_ind_array)
prob.q=sol[-2*prob.S:-prob.S]
prob.s=sol[:-prob.S]
prob.Cv=sol[-prob.S:]

#%%
for i in range(3):
    phi,crds, others, points_a, points_b=Get9Lines(i, 200, L_3D, prob)
    for k in range(9):
        plt.plot(phi[k], label=str(np.array(["x","y","z"])[others]) + "={:.1f}, {:.1f}".format(points_a[k//3], points_b[k%3]))
    plt.xlabel(np.array(["x","y","z"])[i])
    plt.legend()
    plt.show()
#%%
pos_array=np.array([0.2,0.4,0.6,0.8])
resolution=100
phi=[] #Will store the plane reconstruction variables 
phi_extra=[]
phi_1D_full=[]
coordinates=[]
for x in pos_array*L_3D[0]:
    corners=np.array([[x,5,5],[x,5,300],[x,300,5],[x,300,300]])
    phi_final,_,_, phi_2, crds=GetPlaneReconstructionFast(x, 0, 1,2,corners , resolution, prob, prob.Cv)
    crds_1D=crds.reshape(resolution, resolution,3)[np.array(pos_array*resolution,dtype=np.int32)]
    phi_1D=[]
    for i in range(len(pos_array)):
        phi_1D.append(ReconstructionCoordinatesFast(crds_1D[i], prob.n, prob.mesh_3D.cells_x, prob.mesh_3D.cells_y,prob.mesh_3D.cells_z, 
                                                    prob.mesh_3D.h,prob.mesh_3D.pos_cells,prob.mesh_1D.s_blocks, 
                                                    prob.mesh_1D.source_edge,prob.mesh_1D.tau, prob.mesh_1D.pos_s, prob.mesh_1D.h, 
                                                    prob.R, 1, prob.s, prob.q))
    phi_1D_full.append(np.array(phi_1D))
    phi_extra.append(phi_2)
    phi.append(phi_final)
    coordinates.append(crds_1D)
    
#%%

# Generate example matrices
# Define the minimum and maximum values for the color scale
vmin = np.min([phi_extra[0],phi_extra[1],phi_extra[2],phi_extra[3]])
vmax = np.max([phi_extra[0],phi_extra[1],phi_extra[2],phi_extra[3]])

# Plot the matrices using imshow
fig, axs = plt.subplots(2, 4, figsize=(30,16))
im1 = axs[0, 0].imshow(phi[0], cmap='bwr', vmin=vmin, vmax=vmax)
axs[0, 0].set_xlabel("y")
axs[0, 0].set_ylabel("z")
axs[0, 1].plot(coordinates[0][0,:,1],phi_1D_full[0].T)
axs[0, 1].set_xlabel("y")

im2 = axs[0, 2].imshow(phi[1], cmap='bwr', vmin=vmin, vmax=vmax)
axs[0,2].set_xlabel("y")
axs[0, 2].set_ylabel("z")
axs[0, 3].plot(coordinates[1][0,:,1],phi_1D_full[1].T)
axs[0, 3].set_xlabel("y")

im3 = axs[1, 0].imshow(phi[2], cmap='bwr', vmin=vmin, vmax=vmax)
axs[1, 0].set_xlabel("y")
axs[1, 0].set_ylabel("z")
axs[1, 1].plot(coordinates[0][0,:,1],phi_1D_full[2].T)
axs[1, 1].set_xlabel("y")


im4 = axs[1, 2].imshow(phi[3], cmap='bwr', vmin=vmin, vmax=vmax)
axs[1, 2].set_xlabel("y")
axs[1, 2].set_ylabel("z")
axs[1, 3].plot(coordinates[0][0,:,1],phi_1D_full[3].T)
axs[1,3].set_xlabel("y")

# Set titles for the subplots
axs[0, 0].set_title('x={:.1f}'.format(pos_array[0]*L_3D[0]))
axs[0, 1].set_title('x={:.1f}'.format(pos_array[0]*L_3D[0]))
axs[0, 2].set_title('x={:.1f}'.format(pos_array[1]*L_3D[0]))
axs[0, 3].set_title('x={:.1f}'.format(pos_array[1]*L_3D[0]))
axs[1, 0].set_title('x={:.1f}'.format(pos_array[2]*L_3D[0]))
axs[1, 1].set_title('x={:.1f}'.format(pos_array[2]*L_3D[0]))
axs[1, 2].set_title('x={:.1f}'.format(pos_array[3]*L_3D[0]))
axs[1, 3].set_title('x={:.1f}'.format(pos_array[3]*L_3D[0]))

# Adjust spacing between subplots
fig.tight_layout()

# Move the colorbar to the right of the subplots
cbar = fig.colorbar(im1, ax=axs, orientation='vertical', shrink=0.8)
cbar_ax = cbar.ax
cbar_ax.set_position([0.83, 0.15, 0.03, 0.7])  # Adjust the position as needed

# Show the plot
plt.show()


res=100
aax=VisualizationTool(prob, 0,1,2, np.array([[10,10],[10,295],[295,10],[295,295]]), res)
aax.GetPlaneData()

aay=VisualizationTool(prob, 1,0,2, np.array([[10,10],[10,295],[295,10],[295,295]]), res)
aay.GetPlaneData()

aaz=VisualizationTool(prob, 2,0,1, np.array([[10,10],[10,295],[295,10],[295,295]]), res)
aaz.GetPlaneData()
#%%
aax2=VisualizationTool(prob, 0,2,1, np.array([[10,10],[10,295],[295,10],[295,295]]), res)
aax2.GetPlaneData()

#%%
aax.PlotData()
aay.PlotData()
aaz.PlotData()
#%%
aax2.PlotData()
#%%

full=[]
for i in range(4):
    plt.imshow(phi3[i][3], origin="lower", vmax=0.4, vmin=0.0)
    plt.xlabel("y")
    plt.ylabel("z")
    plt.title("x={:.1f}".format(pos_array[i]*L_3D[0]))
    plt.colorbar()
    plt.show()
    full.append(phi3[i][3])
full=np.array(full)



#%%
#p=np.argmin(np.linalg.norm(net.pos_s-L_3D/2, axis=1))
p=np.argsort(net.pos_s[:,0])
prob.q[:]=0
prob.q[p[-10:]]=1
prob.q[p[:-10]]=-0.002
#prob.q[p+10]=-1

new_L=prob.A_matrix.copy()
#new_ind=prob.I_ind_array-M_D*mesh.h**3 + prob.B_matrix.dot(prob.q)
new_ind=prob.I_ind_array + prob.B_matrix.dot(prob.q)
s=dir_solve(new_L, -new_ind)
prob.s=s

for i in range(3):
    phi,crds, others, points_a, points_b=Get9Lines(i, 200, L_3D, prob)
    for k in range(9):
        plt.plot(phi[k], label=str(np.array(["x","y","z"])[others]) + "={:.1f}, {:.1f}".format(points_a[k//3], points_b[k%3]))
    plt.xlabel(np.array(["x","y","z"])[i])
    plt.legend()
    plt.show()


#%%
from post_processing import GetPlaneReconstructionFast
begin2=time.time()
P_per_edge=Pressure[np.arange(len(Pressure)/2, dtype=int)*2]
P_per_source=np.repeat(P_per_edge, net.cells)    

phi=[]
for z in np.array([0.2,0.4,0.6,0.8])*L_3D[0]:
    corners=np.array([[5,5,z],[5,300,z],[300,5,z],[300,300,z]])
    begin=time.time()
    phi.append(GetPlaneReconstructionFast(z, 2, 0,1,corners , 100, prob, prob.Cv))
    end=time.time()
    a=end-begin
    print(a)
end2=time.time()


#%%
for i in range(4):
    plt.imshow(phi[i][0], origin="lower", vmax=0.3, vmin=0)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("z={:.1f}".format(np.array([0.2,0.4,0.6,0.8])[i]*L_3D[2]))
    plt.colorbar()
    plt.show()
# =============================================================================
#     plt.imshow(phi[i][1], origin="lower")
#     plt.colorbar()
#     plt.show()
#     plt.imshow(phi[i][2], origin="lower")
#     plt.colorbar()
#     plt.show()
# =============================================================================

#%%

phi2=[]
for y in np.array([0.2,0.4,0.6,0.8])*L_3D[0]:
    corners=np.array([[5,y,5],[5,y,300],[300,y,5],[300,y,300]])
    begin=time.time()
    phi2.append(GetPlaneReconstructionFast(y, 1, 0,2,corners , 100, prob, prob.Cv))
    end=time.time()
    a=end-begin
    print(a)


#%%
for i in range(4):
    plt.imshow(phi2[i][0], origin="lower", vmax=0.8, vmin=0.3)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.colorbar()
    plt.show()





