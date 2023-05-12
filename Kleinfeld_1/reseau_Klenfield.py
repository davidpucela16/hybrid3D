#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 17:29:09 2023

@author: pdavid
"""


import os 
path=os.path.dirname(__file__)
os.chdir(path)

import pandas as pd
import numpy as np 

import pdb 
from numba import njit

import scipy as sp
#%

filename='/home/pdavid/Bureau/PhD/Network_Flo/All_files/Network1_Figure_Data.txt'
#%
import os

def split_file(filename, output_dir):
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
                output_path = os.path.join(output_dir, output_filename)
                current_output = open(output_path, 'w')
                output_files.append(output_path)

            if current_output:
                current_output.write(line)

        if current_output:
            current_output.close()

        return output_files

# Usage:
output_dir = '/home/pdavid/Bureau/PhD/Network_Flo/All_files/Split/'  # Specify the output directory here
os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
output_files = split_file(filename, output_dir)

print("Split files:")
for file in output_files:
    print(file)



#%
df = pd.read_csv(output_dir + '/output_0.txt', skiprows=1, sep="\s+", names=["x", "y", "z"])
pos_vertex=df.values

df = pd.read_csv(output_dir + '/output_1.txt', skiprows=1, sep="\s+", names=["init", "end"])
edges=df.values

df = pd.read_csv(output_dir + '/output_2.txt', skiprows=1, sep="\s+", names=["cells_per_segment"])
cells_per_segment=np.ndarray.flatten(df.values)

df = pd.read_csv(output_dir + '/output_8.txt', skiprows=1, sep="\s+", names=["diameters"])
diameters=np.ndarray.flatten(df.values)

df = pd.read_csv(output_dir + '/output_9.txt', skiprows=1, sep="\s+", names=["length"])
length=np.ndarray.flatten(df.values)

df = pd.read_csv(output_dir + '/output_10.txt', skiprows=1, sep="\s+", names=["flow_rate"])
flow_rate=np.ndarray.flatten(df.values)
#%
K=np.average(diameters)/np.ndarray.flatten(diameters)

U = 4*flow_rate/np.pi/diameters**2

#%%
startVertex=edges[:,0]
endVertex=edges[:,1]

vertex_to_edge=[[0],[0,1,2], [1], [2]]

def Assemble_vertex_to_edge(vertices, edges):
    """I will use np where because I don't know how to optimize it other wise"""
    vertex_to_edge=[]
    for i in range(len(vertices)):
        a=np.where(edges[:,0]==i)[0]
        b=np.where(edges[:,1]==i)[0]
        temp=list(np.concatenate((a,b)))
        vertex_to_edge.append(temp)
    return vertex_to_edge

@njit
def pre_processing_network(init, end, flow_rate):
    """Pre processes the edges so blood flows always from init to end"""
    for i in np.where(flow_rate<0)[0]:
        temp=init[i]
        init[i]=end[i]
        end[i]=temp
    return init, end

def set_artificial_BCs(vertex_to_edge, entry_concentration, exit_concentration, init, end):
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

def check_local_conservativeness_flow_rate(init, end, vertex_to_edge, flow_rate):
    """Checks if mass is conserved at the bifurcations"""
    vertex=0
    for i in vertex_to_edge:
        
        if len(i)>2:
            a=np.zeros(len(i)) #to store whether the edges are entering or exiting
            c=0
            for j in i: #Goes through each edge of the bifurcation
                a[c]=1 if vertex==init[j] else -1  #Vessel exiting
                c+=1
                
            print(np.dot(flow_rate[i], a))
        vertex+=1
    return
        

def check_local_conservativeness_velocity(init, end, vertex_to_edge, flow_rate, R):
    """Checks if mass is conserved at the bifurcations"""
    vertex=0
    for i in vertex_to_edge:
        
        if len(i)>2:
            a=np.zeros(len(i)) #to store whether the edges are entering or exiting
            c=0
            for j in i: #Goes through each edge of the bifurcation
                a[c]=1 if vertex==init[j] else -1  #Vessel exiting
                c+=1
                
            print(np.dot(flow_rate[i], a))
        vertex+=1
    return

#%
os.chdir('../src_full_opt/')

from mesh_1D import mesh_1D
from mesh import cart_mesh_3D
from hybrid_opt import hybrid_set_up

init, end=pre_processing_network(edges[:-1,0], edges[:-1,1], flow_rate)

vertex_to_edge=Assemble_vertex_to_edge(pos_vertex, edges[:-1])


#%
net=mesh_1D( init, end, vertex_to_edge ,pos_vertex, diameters, np.average(diameters)/2,np.average(U))
net.U=np.ndarray.flatten(U)
cells_3D=10
n=2
L_3D=np.array([1100,1100,1100])
mesh=cart_mesh_3D(L_3D,cells_3D)

net.pos_arrays_fast(mesh)

#Set artificial BCs for the network 
BCs_1D=set_artificial_BCs(vertex_to_edge, 1,0, init, end)

BC_type=np.array(["Neumann", "Neumann", "Neumann","Neumann","Neumann","Neumann"])
BC_value=np.array([0,0,0,0,0,0])

#%%
from hybrid_opt import hybrid_set_up
prob=hybrid_set_up(mesh, net,  BC_type, BC_value,n, 1, K, BCs_1D)


# =============================================================================
# #%%
# from assembly_1D import full_adv_diff_1D, assemble_transport_1D,assemble_vertices,assemble_transport_1D_fast
# data_2, row_2, col_2=assemble_transport_1D_fast(np.ndarray.flatten(U), 1e-15, net.h, net.cells)
# 
# 
# two=sp.sparse.csc_matrix((data_2, (row_2, col_2)), shape=(np.sum(net.cells), np.sum(net.cells)))
# 
# from assembly_1D import assemble_vertices
# assemble_vertices(np.abs(U), 1e-15, net.h, net.cells, np.array([[],[],[]]), vertex_to_edge, diameters/2, init, BCs_1D)
# 
# #%%
# from assembly_1D import full_adv_diff_1D
# import time
# begin=time.time()
# full_adv_diff_1D(np.ndarray.flatten(U), 1, net.h, net.cells, init, vertex_to_edge, diameters/2, BCs_1D)
# end=time.time()
# print("time: ", end-begin)        
# =============================================================================
#%% - Assembly of 3D data

from mesh import cart_mesh_3D

cells_3D=10
n=2
L_3D=np.array([1100,1100,1100])
mesh=cart_mesh_3D(L_3D,cells_3D)

net.pos_arrays(mesh)

BC_type=np.array(["Neumann", "Neumann", "Neumann","Neumann","Neumann","Neumann"])
BC_type=np.array(["Dirichlet", "Dirichlet","Neumann","Neumann", "Dirichlet","Dirichlet"])
BC_value=np.array([0,0,0,0,0,0])

prob=hybrid_set_up(mesh, net, BC_type, BC_value,n,1, K, BCs_1D)

#%%
import time
begin=time.time()
prob.Assembly_I()
end=time.time()
print("time: ", end-begin)  

#%%
pdb.set_trace()
prob.Assembly_D_E_F()
#%%
import pstats
import cProfile
if __name__ == '__main__':
    # run the profiler on the code and save results to a file
    cProfile.run('prob.Assembly_problem()', filename=path + '/profile_results.prof')



stats = pstats.Stats(path + '/profile_results.prof')

stats.strip_dirs().sort_stats('cumulative').print_stats(30)

