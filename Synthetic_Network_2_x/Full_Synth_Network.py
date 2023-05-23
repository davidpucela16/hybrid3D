#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 10:09:55 2023

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

mat_path="/home/pdavid/Bureau/Code/hybrid3d/Synthetic_Network_2_x/F20"
phi_bar_path=os.path.join(mat_path, "matrix_phi_bar")
output_dir_network="/home/pdavid/Bureau/Code/hybrid3d/Synthetic_Network_2_x/divided_files"
filename=os.path.join(path_network,"Rea1_synthetic_x.Smt.SptGraph.am")



def SplitFile(filename, output_dir_network):
    with open(filename, 'r') as file:
        output_files = []
        current_output = None
        line_counter = 0

        for line in file:
            line_counter += 1

# =============================================================================
#             if line_counter < 25:
#                 continue
# =============================================================================

            if line.startswith('@'):
                if current_output:
                    current_output.close()
                output_filename = f"output_{len(output_files)}.txt"
                output_path = os.path.join(output_dir_network, output_filename)
                current_output = open(output_path, 'w')
                output_files.append(output_path)

            if current_output:
                current_output.write(line)

        if current_output:
            current_output.close()

        return output_files



def SetArtificialBCs(vertex_to_edge, entry_concentration, exit_concentration, init, end):
    """Assembles the BCs_1D array with concentration=entry_concentration for the init vertices and 
    concentration=exit_concentration for exiting vertices
    
    Remember to have preprocessed the init and end arrays for the velocity to always be positive"""
    BCs_1D=np.zeros(2, dtype=np.int64)
    c=0
    for i in vertex_to_edge: #Loop over all the vertices
    #i contains the edges the vertex c is in contact with
        if len(i)==1:
            vertex=i[0]
            if np.in1d(i, init):
                BCs_1D=np.vstack((BCs_1D, np.array([c, entry_concentration])))
            else:
                BCs_1D=np.vstack((BCs_1D, np.array([c, exit_concentration])))
        c+=1
    return(BCs_1D)

#%%

cells_3D=20

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
K=np.average(diameters)/np.ndarray.flatten(diameters)/4
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
    
    
def ClassifyVertices(vertex_to_edge, init):
    """Classifies each vertex as entering, exiting or bifurcation
    The flow must have already been pre processed so it is always positive, and the direction is given 
    by the edges array"""
    #BCs is a two dimensional array where the first entry is the vertex and the second the value of the Dirichlet BC
    entering=[]
    exiting=[]
    bifurcation=[]
    vertex=0 #counter that indicates which vertex we are on
    for i in vertex_to_edge:
        if len(i)==1: #Boundary condition
            #Mount the boundary conditions here
            if init[i]!=vertex: #Then it must be the end Vertex of the edge 
                exiting.append(vertex)
            else: 
                entering.append(vertex)
        else: #Bifurcation between two or three vessels (or apparently more)
            bifurcation.append(vertex)
        vertex+=1
    return entering, exiting, bifurcation

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

def get_phi_bar(phi_bar_path, s, q):
    phi_bar_s=sp.sparse.load_npz( phi_bar_path + "/phi_bar_s.npz")
    phi_bar_q=sp.sparse.load_npz( phi_bar_path + "/phi_bar_q.npz")
    
    phi_bar=phi_bar_s.dot(s)+phi_bar_q.dot(q)
    return phi_bar

phi_bar=get_phi_bar(mat_path, prob.s, prob.q)

#%%
from post_processing import *


def Get9Lines(ind_axis, resolution, L_3D, prob):
    
    #The other two axis are:
    others=np.delete(np.array([0,1,2]), ind_axis)
    
    points_a=np.array([1/6,3/6,5/6])*L_3D[others[0]]
    points_b=np.array([1/6,3/6,5/6])*L_3D[others[1]]
    
    crds=np.zeros([9,3,resolution])
    
    indep=np.linspace(0,L_3D[ind_axis], resolution)*0.98+L_3D[ind_axis]*0.01
    
    
    for i in range(3):
        for j in range(3):
            a=np.zeros(resolution)+points_a[i]
            b=np.zeros(resolution)+points_b[j]
            
            crds[j+3*i, ind_axis,:]=indep
            crds[j+3*i, others[0],:]=a
            crds[j+3*i, others[1],:]=b
        
    
    phi = []
    for i in range(9):
        phi.append(ReconstructionCoordinatesFast(crds[i, :, :].T, n, prob.mesh_3D.cells_x, 
                                                          prob.mesh_3D.cells_y,prob.mesh_3D.cells_z, prob.mesh_3D.h,
                                                          prob.mesh_3D.pos_cells,prob.mesh_1D.s_blocks, 
                                                          prob.mesh_1D.source_edge,prob.mesh_1D.tau, 
                                                          prob.mesh_1D.pos_s, prob.mesh_1D.h, prob.mesh_1D.R, 
                                                          1, prob.s, prob.q))
    
    return(phi,crds, others, points_a, points_b)

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
for x in pos_array*L_3D[0]:
    corners=np.array([[x,5,5],[x,5,300],[x,300,5],[x,300,300]])
    phi_final,_,_, phi_2, crds=GetPlaneReconstructionFast(x, 0, 1,2,corners , resolution, prob, prob.Cv)
    cords_1D=crds.reshape(resolution, resolution)[np.array(pos_array*resolution,dtype=np.int32)]
    phi_1D=[]
    for i in range(len(pos_arrays)):
        phi_1D.append(ReconstructionCoordinatesFast(crds_1D[i], prob.n, prob.mesh_3D.cells_x, prob.mesh_3D.cells_y,prob.mesh_3D.cells_z, 
                                                    prob.mesh_3D.h,prob.mesh_3D.pos_cells,prob.mesh_1D.s_blocks, 
                                                    prob.mesh_1D.source_edge,prob.mesh_1D.tau, prob.mesh_1D.pos_s, prob.mesh_1D.h, 
                                                    prob.R, 1, prob.s, prob.q))
    phi_1D_full.append(phi_1D)
    phi_extra.append(phi_2)
    phi.append(phi_final)
    
#%%

# Generate example matrices
# Define the minimum and maximum values for the color scale
vmin = np.min([phi3[0][3],phi3[1][3],phi3[2][3],phi3[3][3]])
vmax = np.max([phi3[0][3],phi3[1][3],phi3[2][3],phi3[3][3]])

# Plot the matrices using imshow
fig, axs = plt.subplots(2, 4, figsize=(15, 12))
im1 = axs[0, 0].imshow(phi3[0][0], cmap='bwr', vmin=vmin, vmax=vmax)
im2 = axs[0, 2].imshow(phi3[1][0], cmap='bwr', vmin=vmin, vmax=vmax)
im3 = axs[1, 0].imshow(phi3[2][0], cmap='bwr', vmin=vmin, vmax=vmax)
im4 = axs[1, 2].imshow(phi3[3][0], cmap='bwr', vmin=vmin, vmax=vmax)

# Set titles for the subplots
axs[0, 0].set_title('Matrix 1')
axs[0, 1].set_title('Matrix 2')
axs[1, 0].set_title('Matrix 3')
axs[1, 1].set_title('Matrix 4')

# Adjust spacing between subplots
fig.tight_layout()

# Move the colorbar to the right of the subplots
cbar = fig.colorbar(im1, ax=axs, orientation='vertical', shrink=0.8)
cbar_ax = cbar.ax
cbar_ax.set_position([0.92, 0.15, 0.03, 0.7])  # Adjust the position as needed

# Show the plot
plt.show()

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







#%% - plane
# =============================================================================
# %%time
# phi = []
# for i in range(ll):
#     phi.append(delayed(ReconstructionCoordinatesFast)(np.array([y, np.zeros(ll)+ y[i], np.zeros(ll)+ L_3D[0]/2]).T, prob))
# 
# phi_array = dask.compute(*phi)
# 
# 
# #%%
# 
# %%time
# from small_functions import GetBoundaryStatus
# phi = np.zeros([ll,ll])
# for i in range(ll):
#     crds=np.array([y, np.zeros(ll)+ y[i], np.zeros(ll)+ L_3D[0]/4*3]).T
#     
#     phi[i]=ReconstructionCoordinatesFast(crds, n, prob.mesh_3D.cells_x, prob.mesh_3D.cells_y,prob.mesh_3D.cells_z,
#                                          prob.mesh_3D.h,prob.mesh_3D.pos_cells,
#                                          prob.mesh_1D.s_blocks, prob.mesh_1D.source_edge,prob.mesh_1D.tau, prob.mesh_1D.pos_s, prob.mesh_1D.h, 
#                                          prob.R, 1, prob.s, prob.q)
# 
# plt.imshow(phi, origin="lower")
# plt.xlabel("x")
# plt.ylabel("y")
# 
# =============================================================================


# =============================================================================
# #%%
# from post_processing import *
# GetPlaneReconstructionFast
# def GetPlaneReconstructionFast(plane_coord,plane_axis, i_axis, j_axis,corners, resolution, prob, property_array):
#     crds=GetCoordsPlane(corners, resolution)
#     mask=GetPlaneIntravascularComponent(plane_coord, prob.mesh_1D.pos_s, prob.mesh_1D.source_edge, 
#                                         plane_axis, i_axis, j_axis, corners, prob.mesh_1D.tau, 
#                                         resolution, prob.mesh_1D.R, prob.mesh_1D.h, prob.mesh_1D.cells)
#     intra=property_array[mask-1]
#     result = np.where(mask == 0, np.nan, intra)
#     new_mask=mask > 0
#     
#     phi=ReconstructionCoordinatesFast(crds, n, prob.mesh_3D.cells_x, prob.mesh_3D.cells_y,prob.mesh_3D.cells_z,
#                                          prob.mesh_3D.h,prob.mesh_3D.pos_cells,
#                                          prob.mesh_1D.s_blocks, prob.mesh_1D.source_edge,prob.mesh_1D.tau, prob.mesh_1D.pos_s, prob.mesh_1D.h, 
#                                          prob.R, 1, prob.s, prob.q)
#     plt.imshow(mask)
#     plt.show()
#     
#     plt.imshow(phi.reshape(resolution,resolution))
#     plt.show()
#     
#     plt.imshow(result)
#     plt.show()
#     
#     phi_final=phi.reshape(resolution,resolution)
#     phi_final[new_mask]=result[new_mask]
#     
#     return  phi_final,result,mask
# 
# def GetPlaneReconstructionParallel(plane_coord,plane_axis, i_axis, j_axis,corners, resolution, prob, property_array):
#     crds=GetCoordsPlane(corners, resolution)
#     mask=GetPlaneIntravascularComponent(plane_coord, prob.mesh_1D.pos_s, prob.mesh_1D.source_edge, 
#                                         plane_axis, i_axis, j_axis, corners, prob.mesh_1D.tau, 
#                                         resolution, prob.mesh_1D.R, prob.mesh_1D.h, prob.mesh_1D.cells)
#     intra=property_array[mask-1]
#     result = np.where(mask == 0, np.nan, intra)
#     new_mask=mask > 0
#     
#     phi=ReconstructionCoordinatesParallel(crds, n, prob.mesh_3D.cells_x, prob.mesh_3D.cells_y,prob.mesh_3D.cells_z,
#                                          prob.mesh_3D.h,prob.mesh_3D.pos_cells,
#                                          prob.mesh_1D.s_blocks, prob.mesh_1D.source_edge,prob.mesh_1D.tau, prob.mesh_1D.pos_s, prob.mesh_1D.h, 
#                                          prob.R, 1, prob.s, prob.q)
#     phi2=phi.copy()
#     
#     plt.imshow(phi.reshape(resolution,resolution), origin="lower")
#     plt.show()
#     
#     plt.imshow(result, origin="lower")
#     plt.show()
#     
#     phi_final=phi.reshape(resolution,resolution)
#     phi_final[new_mask]=result[new_mask]
#     
#     return phi_final,result,phi2.reshape(resolution, resolution)
#         
#         
# def GetPlaneIntravascularComponent(plane_coord, pos_s, source_edge,
#                                    plane_axis, i_axis, j_axis, corners, 
#                                    tau_array, resolution,R, h_1D, cells_per_segment):
#     """This function aims to provide the voxels of the plane defined by plane_coord whose center fall
#     within a source cylinder. We work in 2D so it is not as confusing
#     1st - We figure out which sources intersect the plane 
#     2nd - Out of those possible sources, we loop through each of them to find which voxels fall within the cylinder
#     
#     plane_coord -> The coordinates of the plane on the axis perpendicular to itself
#     pos_s -> position of the sources
#     plane_axis -> 1, 2 or 3 for x, y or z
#     corners_plane -> self explanatory
#     tau_array -> the axial unitary vector for each source
#     i_axis -> axis that we want to be in the horizontal direction of the matrix
#     j_axis -> axis that we want to be in the vertical direction of the matrix
#     
#     Corners are given in the following order:
#         (0,0), (0,1), (1,0), (1,1)
#     where the first entry is the horizontal direction and the second is the vertical direction
#     according to the i_axis and j_axis"""
#     
#     #crds=GetCoordsPlane(corners, resolution)
#     #crds_plane=np.delete(crds, plane_axis, axis=1) #We eliminate the component perpendicular to the plane
#     
#     #First - Figure out the sources that intersect the plane
#     pos_s_plane=pos_s[:,plane_axis]
#     sources=np.where(np.abs(pos_s_plane - plane_coord) < np.repeat(h_1D, cells_per_segment)/2)[0]
#     
#     mask=np.zeros((resolution, resolution), dtype=np.int64)
#     
#     tau=np.linalg.norm((corners[2]-corners[0])/resolution)
#     h=np.linalg.norm((corners[1]-corners[0])/resolution)
#     
#     x=np.linspace(corners[0,i_axis]+tau/2, corners[2,i_axis]-tau/2, resolution)
#     y=np.linspace(corners[0,j_axis]+h/2, corners[1,j_axis]-h/2, resolution)
#     X,Y=np.meshgrid(x,y)
#     
#     for s in sources:
# # =============================================================================
# #         x_dist=X-pos_s[s,i_axis]
# #         y_dist=Y-pos_s[s,j_axis]
# #         
# #         a=(x_dist*tau_array[s,i_axis]**2+y_dist*tau_array[s,j_axis]**2 < h_1D[s]**2)
# #         b=(x_dist*tau_array[s,j_axis]**2+y_dist*tau_array[s,i_axis]**2 < R[s]**2)
# #         d=np.where(a & b)
# # =============================================================================
#         P = np.stack((X - pos_s[s,0], Y - pos_s[s,1], np.zeros_like(X)), axis=-1)
#         
#         # Project the position vectors onto the direction vector to get the scalar value t
#         v=tau_array[source_edge[s]]
#         t = np.sum(P *v, axis=-1)
#         
#         # Calculate the perpendicular distance of each point from the cylinder axis
#         d = np.linalg.norm(P - (t[:, :, np.newaxis] * v), axis=-1)
#         
#         # Find the indices of points that are within the cylinder
#         indices = np.where((d <= R[source_edge[s]]) & (np.abs(t) <= h_1D[source_edge[s]]/2))
#         mask[indices]=s+1
#     return mask
# 
# 
# def GetCoordsPlane(corners, resolution):
#     """We imagine the plane with a horizontal (x) and vertical direction (y).
#     Corners are given in the following order:
#         (0,0), (0,1), (1,0), (1,1)
#         
#         - tau indicates the discretization size in horizontal direction
#         - h indicates the discretization size in vertical direcion"""
#     crds=np.zeros((0,3))
#     tau=(corners[2]-corners[0])/resolution
#     
#     h=(corners[1]-corners[0])/resolution
#     L_h=np.linalg.norm((corners[1]-corners[0]))
#     
#     local_array= np.linspace(corners[0]+tau/2, corners[2]-tau/2 , resolution ) #along the horizontal direction
#     for j in range(resolution):
#         arr=local_array.copy()
#         arr[:,0]+=h[0]*(j+1/2)
#         arr[:,1]+=h[1]*(j+1/2)
#         arr[:,2]+=h[2]*(j+1/2)
#         
#         crds=np.vstack((crds, arr))
#     return(crds)
# =============================================================================

#corners=np.array([[5,5,150],[5,300,150],[300,5,150],[300,300,150]])
# =============================================================================
# hi=GetPlaneReconstructionParallel(150, 2, 0,1,corners , 150, prob)
# #%%
# plt.imshow(hi, origin="lower", vmax=0.8)
# plt.colorbar()
# plt.show()
# 
# =============================================================================
#%%
# =============================================================================
# begin=time.time()
# hi=GetPlaneReconstructionParallel(150, 2, 0,1,corners , 500, prob)
# end=time.time()
# a=end-begin
# print(a)
# 
# begin=time.time()
# hi=GetPlaneReconstructionFast(150, 2, 0,1,corners , 500, prob)
# end=time.time()
# b=end-begin
# =============================================================================