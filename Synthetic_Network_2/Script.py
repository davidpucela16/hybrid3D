#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 10:09:55 2023

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
os.chdir('../src_final/')
from mesh_1D import mesh_1D
from hybridFast import hybrid_set_up
from mesh import cart_mesh_3D

from assembly_1D import AssembleVertexToEdge, PreProcessingNetwork, CheckLocalConservativenessFlowRate, CheckLocalConservativenessVelocity

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


# =============================================================================
# def AssembleVertexToEdge(vertices, edges):
#     """I will use np where because I don't know how to optimize it other wise"""
#     vertex_to_edge=[]
#     for i in range(len(vertices)):
#         a=np.where(edges[:,0]==i)[0]
#         b=np.where(edges[:,1]==i)[0]
#         temp=list(np.concatenate((a,b)))
#         vertex_to_edge.append(temp)
#     return vertex_to_edge
# 
# @njit
# def PreProcessingNetwork(init, end, flow_rate):
#     """Pre processes the edges so blood flows always from init to end"""
#     for i in np.where(flow_rate<0)[0]:
#         temp=init[i]
#         init[i]=end[i]
#         end[i]=temp
#     return init, end
# 
# 
# 
# def CheckLocalConservativenessFlowRate(init, end, vertex_to_edge, flow_rate):
#     """Checks if mass is conserved at the bifurcations"""
#     vertex=0
#     for i in vertex_to_edge:
#         
#         if len(i)>2:
#             a=np.zeros(len(i)) #to store whether the edges are entering or exiting
#             c=0
#             for j in i: #Goes through each edge of the bifurcation
#                 a[c]=1 if vertex==init[j] else -1  #Vessel exiting
#                 c+=1
#                 
#             print(np.dot(flow_rate[i], a))
#         vertex+=1
#     return
#         
# 
# def CheckLocalConservativenessVelocity(init, end, vertex_to_edge, flow_rate, R):
#     """Checks if mass is conserved at the bifurcations"""
#     vertex=0
#     for i in vertex_to_edge:
#         
#         if len(i)>2:
#             a=np.zeros(len(i)) #to store whether the edges are entering or exiting
#             c=0
#             for j in i: #Goes through each edge of the bifurcation
#                 a[c]=1 if vertex==init[j] else -1  #Vessel exiting
#                 c+=1
#                 
#             print(np.dot(flow_rate[i], a))
#         vertex+=1
#     return
# =============================================================================



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


def get_problem(cells_3D, n):
    
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

    startVertex=edges[:,0]
    endVertex=edges[:,1]
    init, end=PreProcessingNetwork(startVertex, endVertex, flow_rate)
    
    vertex_to_edge=AssembleVertexToEdge(pos_vertex, edges)
    
    L_3D=np.array([1100,1100,1100])
    
    #Set artificial BCs for the network 
    BCs_1D=SetArtificialBCs(vertex_to_edge, 1,0, init, end)
    
    BC_type=np.array(["Neumann", "Neumann", "Neumann","Neumann","Neumann","Neumann"])
    BC_value=np.array([0,0,0,0,0,0])
    
    net=mesh_1D( init, end, vertex_to_edge ,pos_vertex, diameters, np.average(diameters)/2,np.average(U))
    net.U=np.ndarray.flatten(U)
    
    mesh=cart_mesh_3D(L_3D,cells_3D)
    net.PositionalArraysFast(mesh)
    
    prob=hybrid_set_up(mesh, net, BC_type, BC_value,n,1, K, BCs_1D)
    return (prob)