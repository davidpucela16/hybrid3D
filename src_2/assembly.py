#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 16:17:50 2023

@author: pdavid
"""
import numpy as np 
import scipy as sp 

from small_functions import append_sparse


def Assembly_diffusion_3D_interior(mesh_object):
    #arrays containing the non-zero entries of the matrix
    row_array=np.array([]) #Contains the row values of each entry
    col_array=np.array([]) #Contains the colume values of each entry
    data_array=np.array([]) #Contains the entry value
    
    mesh_object.assemble_boundary_vectors()
    
    for k in mesh_object.int: #Loop that goes through each non-boundary cell
        rows=np.zeros(7)+k
        cols=mesh_object.get_diff_stencil(k)
        data=np.array([-6,1,1,1,1,1,1])
        
        row_array=np.concatenate((row_array, rows))
        col_array=np.concatenate((col_array, cols))
        data_array=np.concatenate((data_array, data))
        
    c=0
    for k in mesh_object.full_boundary: 
        neighs=np.squeeze(np.array([mesh_object.connect_list[c]]))
        
        rows=np.zeros(len(neighs)+1)+k
        cols=np.concatenate((np.array([k]) , neighs))
        data=np.ones(len(neighs)+1)
        data[0]=-len(neighs)
    
        row_array=np.concatenate((row_array, rows))
        col_array=np.concatenate((col_array, cols))
        data_array=np.concatenate((data_array, data))
        c+=1
        
        # =============================================================================
        #       We only multiply the non boundary part of the matrix by h because in the boundaries assembly we need to include the h due to the difference
        #       between the Neumann and Dirichlet boundary conditions. In short Assembly_diffusion_3D_interior returns data that needs to be multiplied by h 
        #       while Assembly_diffusion_3D_boundaries is already dimensionally consistent
        # =============================================================================

    return(np.array([row_array, col_array, data_array]))

        
def Assembly_diffusion_3D_boundaries(mesh_object, BC_type, BC_value):
    """For now, only a single value for the gradient or Dirichlet BC can be given
    and this function will transform the already assembled matrix to include the 
    boundary ocnditions"""
    row_array=np.array([]) #Contains the row values of each entry
    col_array=np.array([]) #Contains the colume values of each entry
    data_array=np.array([]) #Contains the entry value
    
    BC_array=np.zeros(mesh_object.size_mesh) #The array with the BC values 
    c=0
    for bound in mesh_object.full_full_boundary:
    #This loop goes through each of the boundary cells, and it goes repeatedly 
    #through the edges and corners accordingly
        for k in bound: #Make sure this is the correct boundary variable
            if BC_type[c]=="Dirichlet":
                row_array=np.append(row_array, k)
                col_array=np.append(col_array, k)
                data_array=np.append(data_array, -2*mesh_object.h)
                BC_array[k]=2*BC_value[c]*mesh_object.h
                
            if BC_type[c]=="Neumann":
                BC_array[k]=BC_value[c]*mesh_object.h**2
        c+=1
        
        # =============================================================================
        #       We only multiply the non boundary part of the matrix by h because in the boundaries assembly we need to include the h due to the difference
        #       between the Neumann and Dirichlet boundary conditions. In short Assembly_diffusion_3D_interior returns data that needs to be multiplied by h 
        #       while Assembly_diffusion_3D_boundaries is already dimensionally consistent
        # =============================================================================
    return(row_array, col_array, data_array, BC_array)      

    

def assemble_transport_1D(U, D, h, cells):
    """Assembles the linear matrix for convection-dispersion for a single vessel with Dirichlet
    boundary conditions on both sides"""
    
    sparse_arrs=np.array([]), np.array( []), np.array([])
    
    for i in np.arange(cells-2)+1:
        sparse_arrs =append_sparse(sparse_arrs, np.array([-U-D/h,U+ 2*D/h, -D/h]), np.array([i,i,i]), np.array([i-1,i,i+1]))
    
    #Mount the boundary conditions here
    sparse_arrs=append_sparse(sparse_arrs, np.array([U+3*D/h, -D/h]), np.array([0,0]), np.array([0,1]))
    initial_value_Dirichlet=-U-2*D/h
    
# =============================================================================
#     sparse_arrs=append_sparse(sparse_arrs, np.array([-U/2-D/h,-U/2+3*D/h]), np.array([cells-1,cells-1]), np.array([cells-2,cells-1]))
#     final_value_Dirichlet=U-2*D/h
# =============================================================================
    sparse_arrs=append_sparse(sparse_arrs, np.array([-U-D/h,U+3*D/h]), np.array([cells-1,cells-1]), np.array([cells-2,cells-1]))
    final_value_Dirichlet=-2*D/h
        

    return sparse_arrs, (initial_value_Dirichlet, final_value_Dirichlet)


       
                
#analytical=(np.exp(-Pe*x)-np.exp(-Pe))/(1-np.exp(-Pe))
                
                
            
    
    
    