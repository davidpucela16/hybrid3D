#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 09:22:23 2023

@author: pdavid
"""

import numpy as np 
import pdb 
import matplotlib.pyplot as plt
from neighbourhood import get_neighbourhood, get_uncommon
from small_functions import trilinear_interpolation, auto_trilinear_interpolation
from Green import get_grad_source_potential, get_source_potential
from small_functions import for_boundary_get_normal

class hybrid_set_up():
    def __init__(self, mesh_3D, mesh_1D,  BC_type, BC_value,n, D, K):
        
        self.K=K
        
        self.mesh_3D=mesh_3D
        self.h=mesh_3D.h
        
        self.mesh_1D=mesh_1D
        self.D=D
        self.BC_type=BC_type
        self.BC_value=BC_value
        
        self.n=n
        
        return
    
    def interpolate(self, x):
        """returns the kernels to obtain an interpolation on the point x. 
        In total it will be 6 kernels, 3 for columns and 3 with data for s, q, and
        C_v respectively"""

        bound_status=self.get_bound_status(x)
        if np.sum(bound_status): #boundary interpolation
            nodes=self.construct_dual_cube(x, bound_status)
            nodes=self.get_cube_boundary_slow_values(nodes)
            
            
        else: #no boundary node (standard interpolation)
            blocks=self.mesh_3D.get_8_closest(x)
            nodes=np.array([], dtype=node)
            dual_neigh=np.array([], dtype=int)
            c=0
            for i in blocks:
                #Create node object
                nodes=np.append(nodes, node(self.mesh_3D.get_coords(i), c))
                #Append the neighbourhood to the neigh object variable
                nodes[c].neigh=get_neighbourhood(self.n, self.mesh_3D.cells_x, 
                                                 self.mesh_3D.cells_y, 
                                                 self.mesh_3D.cells_z, 
                                                 i)
                dual_neigh=np.concatenate((dual_neigh, nodes[c].neigh))
                c+=1
        return self.get_interp_kernel(x,nodes, np.unique(dual_neigh))
    
    def get_interp_kernel(self,x, nodes, dual_neigh):
        #From the nodes coordinates, their neighbourhoods and their kernels we do the interpolation
        #Therefore, the kernels represent the slow term, the corrected rapid term must be calculated
        
        #INTERPOLATED PART:
        kernel_q, kernel_C_v, kernel_s, col_q, col_C_v, col_s=self.get_I_1(x, nodes, dual_neigh)
        
        #RAPID (NON INTERPOLATED) PART
        q,C,sources=self.mesh_1D.kernel_point(x, dual_neigh, get_source_potential, self.K, self.D)
        
        kernel_q=np.concatenate((kernel_q, q))
        kernel_C_v=np.concatenate((kernel_C_v, C))
        col_q=np.concatenate((col_q, sources))
        col_C_v=np.concatenate((col_C_v, sources))
        
        return kernel_q, kernel_C_v, kernel_s, col_q, col_C_v, col_s
    
    
    def get_I_1(self, x,nodes, dual_neigh):
        """Returns the kernels of the already interpolated part, il reste juste ajouter
        le term rapide corrigé
            - x is the point where the concentration is interpolated"""
        
        weights=auto_trilinear_interpolation(x, nodes)
        
        kernel_q=np.array([])
        kernel_C_v=np.array([])
        kernel_s=np.array([])
        col_q=np.array([])
        col_C_v=np.array([])
        col_s=np.array([])
        
        for i in range(8): #Loop through each of the nodes
            
            U=get_uncommon(dual_neigh, nodes[i].neigh) 
            #The following variable will contain the data kernel for q, the data kernel
            #for C_v and the col kernel i.e. the sources 
            a=self.mesh_1D.kernel_point(x, U, get_source_potential, self.K, self.D)
            
            nodes[i].kernel_q=np.concatenate((nodes[i].kernel_q, a[0]))
            nodes[i].kernel_C_v=np.concatenate((nodes[i].kernel_q, a[1]))
            
            nodes[i].col_q=np.concatenate((nodes[i].kernel_q, a[2]))
            if np.any(a[1]): 
                nodes[i].col_C_v=np.concatenate((nodes[i].col_C_v, a[2]))
                
            #This operation is a bit redundant
            nodes[i].multiply_by_value(weights[i])
            
            kernel_q=np.concatenate((kernel_q, nodes[i].kernel_q))
            kernel_C_v=np.concatenate((kernel_C_v, nodes[i].kernel_C_v))
            kernel_s=np.concatenate((kernel_s, nodes[i].kernel_s))
            
            col_q=np.concatenate((col_q, nodes[i].col_q))
            col_C_v=np.concatenate((col_C_v, nodes[i].col_C_v))
            col_s=np.concatenate((col_s, nodes[i].col_s))
            
        return kernel_q, kernel_C_v, kernel_s, col_q, col_C_v, col_s
        
    
  
    def get_bound_status(self, coords):
        bound_status=np.array([], dtype=int) #array that contains the boundaries that lie less than h/2 from the point
        if int(coords[0]/(self.h/2))==0: bound_status=np.append(bound_status, 5) #down
        elif int(coords[0]/(self.h/2))==self.mesh_3D.cells_x-1: bound_status=np.append(bound_status, 4) #top
        if int(coords[1]/(self.h/2))==0: bound_status=np.append(bound_status, 3) #west
        elif int(coords[1]/(self.h/2))==self.mesh_3D.cells_y-1: bound_status=np.append(bound_status, 2) #east
        if int(coords[2]/(self.h/2))==0: bound_status=np.append(bound_status, 1) #south
        elif int(coords[2]/(self.h/2))==self.mesh_3D.cells_z-1: bound_status=np.append(bound_status, 0) #north
        
        return bound_status
    
    def construct_dual_cube(self, coords, bound_status):
        """This function constructs the cube by initiliazing the 8 nodes composing 
        the dual cube.
        The function returns the nodes where each of them will contain as internal variables:
            - The position
            - the local coordinates (WHICH I'M NOT SURE ARE USED AFTEWARDS)
            - Wether it is a boundary or not"""
        k=self.mesh_3D.get_id(coords) #Nearest cartesian node
        
        x_k=self.mesh_3D.get_coords(k)
        #The following vector gives the direction from x_k that the cube has to be constructed
        d=np.array((np.sign(coords-x_k)+1)/2).astype(int) 
        #For each direction d is 0 if against the axis and 1 otherwise
        
        direction=np.array([[1,0],
                            [3,2],
                            [5,4]]) [[0,1,2],[d]][0]
        h_array=np.array([[0,0,self.h],
                    [0,0,-self.h],
                    [0,self.h,0],
                    [0,-self.h,0],
                    [self.h,0,0],
                    [-self.h,0,0]])
        
        h_array[bound_status]/=2
        h_plus=h_array[direction] #the signed distance on each direction for dual cube
        
        #We want an array that which one of the three directions (0, 1 or 2) points to a boundary
        #0 for the z direction, 1 for the y direction, 2 for the x direction
        dir_boundary=np.where(np.abs(h_plus[np.arange(3), np.array([2,1,0])])<self.h-1e-7)[0]
        bound_dir=np.zeros(3, dtype=int)-1
        for i in dir_boundary: #loop through each of the directions where we encounter a boundary 
        #len(dir_boundary)= 0 if no boundary, =1 if boundary, =2 if edge, =3 if corner
            bound_dir[i]=d[i]
            
            
        #Now we have an array that for each direction: -1 if no boundary
                                              #0 if boundary in the negative direction of the axis
                                              #1 if boundary in the positive direction
        #So dir boundary indicates which one of the three axis contains a boundary
        #bound_dir contains the complementary information of the direction where the boundary lies
        nodes=np.array([])
        local_coords=np.array([[0,0,0]])
        
        nodes=np.append(nodes, node(x_k, 0))
        
        nodes=np.append(nodes, node(x_k+h_plus[0], 1))
        nodes[1].bound=np.append(nodes[1].bound, [1,0,-1][bound_dir[0]]) #[1,0,-1]=['south', 'north', 'no_bound']
        
        nodes=np.append(nodes, node(x_k+h_plus[1], 2))
        nodes[2].bound=np.append(nodes[2].bound, [3,2,-1][bound_dir[1]])
        local_coords=np.vstack((local_coords, nodes[2].coords))
        
        nodes=np.append(nodes, node(x_k+h_plus[2], 3))
        nodes[3].bound=np.append(nodes[3].bound, [5,4,-1][bound_dir[2]])
        local_coords=np.vstack((local_coords, nodes[3].coords))
        
        #This ones have to have at least one boundary 
        nodes=np.append(nodes, node(x_k+h_plus[0]+h_plus[1], 4))
        nodes[4].bound=np.append(nodes[1].bound, nodes[2].bound)
        local_coords=np.vstack((local_coords, nodes[4].coords))
        
        nodes=np.append(nodes, node(x_k+h_plus[0]+h_plus[2], 5))
        nodes[5].bound=np.append(nodes[1].bound, nodes[3].bound)
        local_coords=np.vstack((local_coords, nodes[5].coords))

        nodes=np.append(nodes, node(x_k+h_plus[1]+h_plus[2], 6))
        nodes[6].bound=np.append(nodes[2].bound, nodes[3].bound)
        local_coords=np.vstack((local_coords, nodes[6].coords))
        
        nodes=np.append(nodes, node(x_k+h_plus[0]+h_plus[1]+h_plus[2], 7))
        nodes[7].bound=np.concatenate((nodes[1].bound,nodes[2].bound, nodes[3].bound))
        local_coords=np.vstack((local_coords, nodes[7].coords))
        
        return nodes
    
    def get_cube_boundary_slow_values(self, nodes):
        """This function records in each node object the three kernels to multiply 
        the unknowns by to obtain the slow term value at that node.
        At the end of the function, each node has also stored its neighbourhood
        as an internal variable, together with the 3 data kernels and the 3 col kernels
        """
        
        
        direct=np.array([[1,2,3],
                         [0,4,5],
                         [4,0,6],
                         [5,6,0],
                         [2,1,7],
                         [3,7,1],
                         [7,3,0],
                         [6,5,4]])
        
        #This following nodes cannot fall within an edge nor a corner
        cc=0
        for i in nodes[1:4]:
            cc+=1
            i.block_3D=self.mesh_3D.get_id(i.coords)
            i.neigh=get_neighbourhood(self.n, self.mesh_3D.cells_x, 
                                             self.mesh_3D.cells_y, 
                                             self.mesh_3D.cells_z, 
                                             i.block_3D)
            
            #To review these coeffs!!
            i.kernel_s=np.append(i.kernel_s, 1)
            i.col_s=np.append(i.col_s, self.mesh_3D.get_id(i.coords))
            
            if np.sum(i.bound > -1)>0: #Only one can be true
                #normal=(i.coords-nodes[0].coords)/np.linalg.norm(i.coords-nodes[0].coords)
                normal=for_boundary_get_normal(int(i.bound[i.bound > -1]))
                 
                a,b=self.get_values_boundary_nodes(normal, i)
                i.kernel_q=np.concatenate((i.kernel_q, a))
                i.col_q=np.concatenate((i.col_q, b))

            
            
        for i in nodes[4:]: #This ones can have multiple boundaries
            cc+=1
            i.block_3D=self.mesh_3D.get_id(i.coords)
            i.neigh=get_neighbourhood(self.n, self.mesh_3D.cells_x, 
                                             self.mesh_3D.cells_y, 
                                             self.mesh_3D.cells_z, 
                                             i.block_3D)
            
            #To review these coeffs!!
            if np.sum(i.bound > -1)==1: #Only one boundary
                #normal=(i.coords-nodes[0].coords)/np.linalg.norm(i.coords-nodes[0].coords)
                normal=for_boundary_get_normal(int(i.bound[i.bound > -1]))
                 
                one, two, three, four=self.get_values_boundary_nodes(normal, i)
                
                i.kernels_append([one, two, three, four, np.array([]), np.array([])])
                
                
            else: #Multiple boundaries
                i.bound//2 #The neighbourhoods it shares
                for m in i.bound//2:
                    i.kernel_q=np.concatenate((i.kernel_q,nodes[direct[i.ID, m]].kernel_q))
                    i.col_q=np.concatenate((i.col_q,nodes[direct[i.ID, m]].col_q))
                    
                    i.kernels_append([nodes[direct[i.ID, m]].kernel_s, nodes[direct[i.ID, m]].col_s, 
                                      nodes[direct[i.ID, m]].kernel_q, nodes[direct[i.ID, m]].col_q, 
                                      np.array([]), np.array([])])
                    
                    
                i.kernel_q/=len(i.bound)
                i.kernel_s/=len(i.bound)
        return nodes
    
            
    def get_values_boundary_nodes(self,normal, node):
        """We can easily reconstruct the values at the FV nodes. We need to do the same
        for the boundary nodes
        
        Basically, this function provides the kernel to calculate the value of the slow term 
        at a boundary node. In the development, this would be considered a dummy variable, but they 
        are quite crucial for the interpolation, and therefore to reduce errors
        
        THIS FUNCTION IS FOR BOUNDARY NODES, SO THERE IS NO EDGE OR CORNER NODE!
        IT ASSUMES IT IS THE CENTER OF THE SURFACE ELEMENT CREATING THE BOUNDARY BETWEEN THE FV CELL AND THE OUTSIDE"""
        
        coords, boundary_number, neigh=node.coords, node.bound[node.bound>-1], node.neigh
        
        s_blocks=self.mesh_1D.s_blocks
        K=self.K
        D=self.D
        h=self.mesh_3D.h
        pos_s=self.mesh_1D.pos_s
        R=self.mesh_1D.R
        
        if self.BC_type[boundary_number]=="Dirichlet":
            #I think here the kernel s must be modified somehow
            kernel_q, kernel_col=self.mesh_1D.kernel_integral_surface(coords,normal, neigh,get_source_potential, K,D)
            node.BC_value=self.BC_value[boundary_number]
            return(np.array([]), np.array([]),-kernel_q, kernel_col )
        
        elif self.BC_type[boundary_number]=="Neumann":
            #To figure out!!!
            kernel_q,  kernel_col=self.mesh_1D.kernel_integral_surface(coords,normal, neigh,get_grad_source_potential, K,D)
            node.BC_value=self.BC_value[boundary_number]*self.h/2
            return(np.array([1]), np.array([node.block_3D]), -kernel_q*self.h/2,kernel_col)
        
   



class node(): 
    def __init__(self, coords, local_ID):
        self.bound=np.array([], dtype=int) #meant to store -1 if it is not a boundary node and
                                #the number of the boundary if it is
        self.coords=coords
        self.ID=local_ID
        
        self.BC_value=0 #Variable of the independent term caused by the BC. Only not 0 when in a boundary
        #The kernels to multiply the unknowns to obtain the value of the slow term at 
        #the node 
        self.kernel_q=np.array([])
        self.kernel_C_v=np.array([])
        self.kernel_s=np.array([])
        #The kernels with the positions
        self.col_s=np.array([])
        self.col_C_v=np.array([])
        self.col_q=np.array([])
        
    def multiply_by_value(self, value):
        """This function is used when we need to multiply the value of the node 
        but when working in kernel form """

        self.kernel_q*=value
        self.kernel_C_v*=value
        self.kernel_s*=value
        return
    
    def kernels_append(self,arrays_to_append):
        """Function that simplifies a lot the append process"""
        
        a,b,c,d,e,f=arrays_to_append
        
        self.kernel_s=np.concatenate((self.kernel_s,a))
        self.col_s=np.concatenate((self.col_s, b))
        self.kernel_q=np.concatenate((self.kernel_q, c))
        self.col_q=np.concatenate((self.col_q, d))
        self.kernel_C_v=np.concatenate((self.kernel_C_v, e))
        self.col_C_v=np.concatenate((self.col_C_v, f))
        return
                                
        
        