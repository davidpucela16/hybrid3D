#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 17:56:05 2023

@author: pdavid
"""
from Green import grad_point, log_line, Simpson_surface, get_source_potential,get_self_influence
from neighbourhood import get_neighbourhood, get_multiple_neigh
import numpy as np
import pdb


class mesh_1D():
    def __init__(self, startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h):
        """Generates the 1D mesh of cylinders with their centers stored within pos_s       
        
            - startVertex contains the ID of the starting vertex for each edge
           - endVertex: same thing but with the end vertex
           - vertex_to_edge contains the edges each vertex is connected to
           - pos_vertex is the position in the three dimensions of each vertex
           - h is the discretization size of the 1D mesh
           - R is the array of radii for each edge
           - K is the effective diffusivity of each vessel wall"""
        
        L=np.sum((pos_vertex[endVertex] - pos_vertex[startVertex])**2, axis=1)**0.5
        self.L=L
        h=L/np.array(L/h, dtype=int)
        self.h=h
        self.tau=np.divide((pos_vertex[endVertex] - pos_vertex[startVertex]).T,L).T
        self.edges=np.arange(len(startVertex)) #Total number of edges in the network
        pos_s=np.zeros((0,3))
        
        self.source_edge=np.array([], dtype=int) #Array that returns for each cyl source the edge it lies in
        for i in self.edges:
            local=np.linspace(h[i]/2, L[i]-h[i]/2, int(L[i]/h[i]))
            glob=np.multiply.outer(local, self.tau[i])+pos_vertex[startVertex[i]]
            pos_s=np.concatenate([pos_s, glob],axis=0)
            self.source_edge=np.append(self.source_edge, np.zeros(len(local), dtype=int)+i)
            
        self.pos_s=pos_s
        
        
        bif=np.array([], dtype=int)
        bo_ver=np.array([], dtype=int)
        
        c=0
        for i in vertex_to_edge:
            if len(i)==3:
                bif=np.append(bif,c)  
            else:
                bo_ver=np.append(bo_ver,c)
            c+=1
        
        self.bifurcations=bif
        self.boundary_vertex=bo_ver
        
        self.R=diameters/2 #One entry per edge
        return
    
    
    def pos_arrays(self, mesh_3D):
        """This function is the pre processing step. It is meant to create the s_blocks
        and uni_s_blocks arrays which will be used extensively throughout. s_blocks represents
        the block where each source is located, uni_s_blocks contains all the source blocks
        in a given order that will be respected throughout the resolution
        
            - h_cart is the size of the cartesian mesh"""
            
        # pos_s will dictate the ID of the sources by the order they are kept in it!
        s_blocks = np.array([]).astype(int)
        uni_s_blocks = np.array([], dtype=int)
        self.a_array=np.zeros((0,3))
        self.b_array=np.zeros((0,3))
        for i in range(len(self.pos_s)):
            ed=self.source_edge[i] #Current edge (int) the source lies on 
            x_j=self.pos_s[i] #Center of the cylinder
            u=np.array([x_j-self.tau[ed]*self.h[ed]/2 ,x_j+self.tau[ed]*self.h[ed]/2]) #a and b 
            
            self.a_array=np.vstack((self.a_array, u[0]))
            self.b_array=np.vstack((self.b_array, u[1]))
            
            s_blocks=np.append(s_blocks, mesh_3D.get_id(x_j))
# =============================================================================
#             if s_blocks[-1] not in uni_s_blocks:
#                 uni_s_blocks = np.append(uni_s_blocks, s_blocks[-1])
# =============================================================================
        self.s_blocks=s_blocks
        self.uni_s_blocks = np.unique(s_blocks)

        total_sb = len(uni_s_blocks)  # total amount of source blocks
        self.total_sb = total_sb

    def kernel_point(self,x, neighbourhood, function, K, D):
        """Returns the kernels to multiply the vectors of unknowns q and C_v"""
        
        sources=np.arange(len(self.s_blocks))[np.in1d(self.s_blocks, neighbourhood)]
        
        q_array, C_v_array=np.array([]), np.array([])
        for i in sources:
            ed=self.source_edge[i]
            tau=self.tau[ed]
            a,b= self.pos_s[i]-tau*self.h[ed]/2, self.pos_s[i]+tau*self.h[ed]/2
            
            
            if (( np.dot(x-a, tau)>0 ) and ( np.dot(x-a, tau)<self.h[ed] ) and ( np.cross(x-a, tau)<self.R[ed])):
                
                q=get_self_influence(self.R[ed], self.h[ed], self.D)
                C=0
                
            else:
                q,C=function((a,b,x, self.R[ed], K[ed]),D)
                ########################
                #Coefficients due to geometry are already added!
                ########################
            q_array=np.append(q_array, q)
            C_v_array=np.append(C_v_array, C)
        
        #Return q_kernel_data, C_v_kernel_data, the columns for both 
        return q_array, C_v_array, sources
    
    def kernel_integral_surface(self, center,normal, neighbourhood,function, K,D):
        """Returns the kernel that multiplied (scalar, dot) by the array of fluxes (q) returns
        the integral of the rapid term over the surface
        
        Main function used to calculate J
        """
        
        sources=np.arange(len(self.s_blocks))[np.in1d(self.s_blocks, neighbourhood)]
        
        q_array, C_v_array=np.array([]), np.array([])
        for i in sources:
            ed=self.source_edge[i]
            a,b= self.pos_s[i]-self.tau[ed]*self.h[ed]/2, self.pos_s[i]+self.tau[ed]*self.h[ed]/2
            q,C=Simpson_surface((a,b, self.R[ed], K[ed]),function, center,self.h[ed], normal,D)
            ########################
            #Coefficients due to geometry are already added!
            ########################
            q_array=np.append(q_array, q)
            C_v_array=np.append(C_v_array, C)
        
        #Return q_kernel_data, C_v_kernel_dat
        return q_array,  sources


        

    

        
