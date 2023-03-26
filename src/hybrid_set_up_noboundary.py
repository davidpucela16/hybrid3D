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
from small_functions import for_boundary_get_normal, append_sparse

from assembly import Assembly_diffusion_3D_boundaries, Assembly_diffusion_3D_interior

import scipy as sp

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
    
    def Assembly_problem(self):
        size=self.mesh_3D.size_mesh
        
        a=Assembly_diffusion_3D_interior(self.mesh_3D)
        b=Assembly_diffusion_3D_boundaries(self.mesh_3D, self.BC_type, self.BC_value)
        
        A_matrix=sp.sparse.csc_matrix((a[2], (a[0], a[1])), shape=(size,size)) + sp.sparse.csc_matrix((b[2], (b[0], b[1])), shape=(size, size))
        
        
        self.ind_array=b[3]
        
        self.A_matrix=A_matrix
        
        B_matrix=self.Assembly_B()
        self.B_matrix=B_matrix
        
        return
    
    def Assembly_A(self):
        return
    
    def Assembly_B(self):
        """Assembly of the B matrix i.e. computation of the arrays J_k_m and 
        adding them to the B matrix"""
        B=sp.sparse.csc_matrix((0, len(self.mesh_1D.s_blocks)))
        for k in range(self.mesh_3D.size_mesh):
            N_k= self.mesh_3D.ordered_connect_matrix[k] #Set with all the neighbours
            
            J_k=0
            for m in N_k:
                J_k+=self.get_J_k_m(k,m)
            B=sp.sparse.vstack((B, J_k))
        
        return B

            
# =============================================================================
#     def Assembly_B_boundaries(self, BC_value, BC_type):
# 
#         B=self.B.tolil()
#         
#         
#         return(row_array, col_array, data_array, BC_array)      
# =============================================================================
        
    def Assembly_D_E_F(self):
# =============================================================================
#         row_D=np.array([], dtype=int)
#         col_D=np.array([], dtype=int)
#         data_D=np.array([])        
#         
#         row_E=np.array([], dtype=int)
#         col_E=np.array([], dtype=int)
#         data_E=np.array([])      
#         
#         row_F=np.array([], dtype=int)
#         col_F=np.array([], dtype=int)
#         data_F=np.array([])   
# =============================================================================
        
        D=np.zeros([3,0])
        E=np.zeros([3,0])
        F=np.zeros([3,0])
        
   
        
        for j in range(len(self.mesh_1D.s_blocks)):
            kernel_s,col_s,kernel_q, col_q,kernel_C_v,  col_C_v=self.interpolate(self.mesh_1D.pos_s)
            D=append_sparse(D, kernel_s,np.zeros(len(col_s))+j, col_s)
            
            E=append_sparse(E, kernel_q,np.zeros(len(col_q))+j, col_q)
            
            F=append_sparse(F, kernel_C_v,np.zeros(len(col_C_v))+j, col_C_v)
            
            E=append_sparse(E, self.K , np.arange(len(self.mesh_1D.s_blocks)), np.arange(len(self.mesh_1D.s_blocks)))
            
            E=append_sparse(F, np.ones(len(self.mesh_1D.s_blocks)) , np.arange(len(self.mesh_1D.s_blocks)), np.arange(len(self.mesh_1D.s_blocks)))
            
            
        return 
    
    def get_coord_reconst(self,corners, resolution):
        """Corners given in order (0,0),(0,1),(1,0),(1,1)"""
        crds=np.zeros((0,3))
        tau=(corners[2]-corners[0])/resolution
        
        h=(corners[1]-corners[0])/resolution
        L_h=np.linalg.norm((corners[1]-corners[0]))
        
        local_array= np.linspace(corners[0]+h/2, corners[1]-h/2 , resolution )
        for j in range(resolution):
            arr=local_array.copy()
            arr[:,0]+=tau[0]*(j+1/2)
            arr[:,1]+=tau[1]*(j+1/2)
            arr[:,2]+=tau[2]*(j+1/2)
            
            crds=np.vstack((crds, arr))
        
        rec=np.array([])
        for k in crds:
            a,b,c,d,e,f=self.interpolate(k)
            rec=np.append(rec,a.dot(self.s[b])+c.dot(self.q[d]))
            
        return crds, rec
            
        
        
    
    def rec_along_mesh(self,axis, crds_along_axis, s_field, q, C_v_array):
        
        mesh=self.mesh_3D
        net=self.mesh_1D
        
        if axis=="x":
            array=mesh.get_x_slice(crds_along_axis)
            cells=mesh.cells_z, mesh.cells_y
            names="z","y"
        elif axis=="y":
            array=mesh.get_y_slice(crds_along_axis)
            cells=mesh.cells_z, mesh.cells_x
            names="z","x"
        elif axis=="z":
            array=mesh.get_z_slice(crds_along_axis)
            cells=mesh.cells_y, mesh.cells_x
            names="y","x"
            
            
        sol=np.array([])
        for k in array:
            a,b,c,d,e,f=self.interpolate(mesh.get_coords(k))
# =============================================================================
#             if np.sum(d):
#                 sol=np.append(sol,a.dot(s_field[b])+c.dot(q[d]))
#             else:
#                 sol=np.append(sol,a.dot(s_field[b]))
# =============================================================================
            #sol=np.append(sol,a.dot(s_field[b]))
            
            if np.sum(d):
                sol=np.append(sol,c.dot(q[d]))
                #if k not in get_neighbourhood(self.n, mesh.cells_x,mesh.cells_y,mesh.cells_z, net.s_blocks[0]): pdb.set_trace()
            else:
                sol=np.append(sol,0)
        
        return sol.reshape(cells), names
    
    def interpolate(self, x):
        """returns the kernels to obtain an interpolation on the point x. 
        In total it will be 6 kernels, 3 for columns and 3 with data for s, q, and
        C_v respectively"""
        bound_status=self.get_bound_status(x)
        if len(bound_status): #boundary interpolation
            #if x[0]>1 and x[1]<0.5:pdb.set_trace()
            nodes=np.array([node(x, 0)])
            nodes[0].neigh=get_neighbourhood(self.n, self.mesh_3D.cells_x, 
                                             self.mesh_3D.cells_y, 
                                             self.mesh_3D.cells_z, 
                                             self.mesh_3D.get_id(x))
            nodes[0].block_3D=self.mesh_3D.get_id(nodes[0].coords)
            nodes[0].dual_neigh=nodes[0].neigh
            dual_neigh=nodes[0].dual_neigh
            
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
                nodes[c].block_3D=self.mesh_3D.get_id(nodes[c].coords)
                dual_neigh=np.concatenate((dual_neigh, nodes[c].neigh))
                c+=1
        #if np.all(x==np.array([1.5, 4.5, 2.5])): pdb.set_trace()
        return self.get_interp_kernel(x,nodes, np.unique(dual_neigh))
    
    def get_interp_kernel(self,x, nodes, dual_neigh):
        
        #From the nodes coordinates, their neighbourhoods and their kernels we do the interpolation
        #Therefore, the kernels represent the slow term, the corrected rapid term must be calculated
        
        #INTERPOLATED PART:
        #kernel_s,col_s,kernel_q, col_q,kernel_C_v,  col_C_v=self.get_I_1(x, nodes, dual_neigh)
        self.nodes,kernel_s,col_s,kernel_q, col_q,kernel_C_v,  col_C_v=get_I_1(x,nodes, dual_neigh, self.K, self.D,
                                                                    self.mesh_3D.h, self.mesh_1D)
        
        #kernel_s,col_s,kernel_q, col_q,kernel_C_v,  col_C_v=np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
        #RAPID (NON INTERPOLATED) PART
        q,C,sources=self.mesh_1D.kernel_point(x, dual_neigh, get_source_potential, self.K, self.D)
        kernel_q=np.concatenate((kernel_q, q))
        kernel_C_v=np.concatenate((kernel_C_v, C))
        col_q=np.concatenate((col_q, sources))
        col_C_v=np.concatenate((col_C_v, sources))
        
        
        return kernel_s,col_s.astype(int),kernel_q, col_q.astype(int),kernel_C_v,  col_C_v.astype(int)
    
    

    
  
    def get_bound_status(self, coords):
        """Take good care of never touching the boundary!!"""
        bound_status=np.array([], dtype=int) #array that contains the boundaries that lie less than h/2 from the point
        if int(coords[0]/(self.h/2))==0: bound_status=np.append(bound_status, 5) #down
        elif int(coords[0]/(self.h/2))==2*self.mesh_3D.cells_x-1: bound_status=np.append(bound_status, 4) #top
        if int(coords[1]/(self.h/2))==0: bound_status=np.append(bound_status, 3) #west
        elif int(coords[1]/(self.h/2))==2*self.mesh_3D.cells_y-1: bound_status=np.append(bound_status, 2) #east
        if int(coords[2]/(self.h/2))==0: bound_status=np.append(bound_status, 1) #south
        elif int(coords[2]/(self.h/2))==2*self.mesh_3D.cells_z-1: bound_status=np.append(bound_status, 0) #north
        
        return bound_status
    

    
            
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
            kernel_q, kernel_col=self.mesh_1D.kernel_integral_surface(coords,normal, h,neigh,get_source_potential, K,D)
            node.BC_value=self.BC_value[boundary_number]
            return(np.array([]), np.array([]),-kernel_q, kernel_col )
        
        elif self.BC_type[boundary_number]=="Neumann":
            #To figure out!!!
            kernel_q,  kernel_col=self.mesh_1D.kernel_integral_surface(coords,normal, h,neigh,get_grad_source_potential, K,D)
            node.BC_value=self.BC_value[boundary_number]*self.h/2
            return(np.array([1]), np.array([node.block_3D]), -kernel_q*self.h/2,kernel_col)
        
   
    def get_interface_kernels(self,k,m):
        """This function returns the relevant kernels for the integrals needed to
        assemble the vector J_km and therefore the b_matrix
        
        k and m must be direct neighbours"""
        
        net=self.mesh_1D #1D mesh object that contains the functions to calculate the integral kernels
        mesh=self.mesh_3D
        h=mesh.h
        k_neigh=get_neighbourhood(self.n, mesh.cells_x, mesh.cells_y, mesh.cells_z, k)
        m_neigh=get_neighbourhood(self.n, mesh.cells_x, mesh.cells_y, mesh.cells_z, m)
        
        
        pos_k=mesh.get_coords(k)
        pos_m=mesh.get_coords(m)
        
        normal=(pos_m-pos_k)/mesh.h
        
        
        r_k_m=net.kernel_integral_surface(pos_m/2+pos_k/2, normal,  get_uncommon(k_neigh, m_neigh), get_source_potential, self.K,self.D)
        r_m_k=net.kernel_integral_surface(pos_m/2+pos_k/2, normal,  get_uncommon(m_neigh, k_neigh), get_source_potential, self.K,self.D)
        
        #if np.sum(r_m_k[0]) and np.sum(r_k_m[0]): pdb.set_trace()
        
        grad_r_k_m=net.kernel_integral_surface(pos_m/2+pos_k/2, normal, get_uncommon(k_neigh, m_neigh), get_grad_source_potential, self.K,self.D)
        grad_r_m_k=net.kernel_integral_surface(pos_m/2+pos_k/2, normal, get_uncommon(m_neigh, k_neigh),  get_grad_source_potential, self.K,self.D)
        
        
        return(sp.sparse.csc_matrix((r_k_m[0]*h**2,(np.zeros(len(r_k_m[0])),r_k_m[1])), shape=(1,len(net.s_blocks))),
               sp.sparse.csc_matrix((grad_r_k_m[0]*h**2 ,(np.zeros(len(grad_r_k_m[1])), grad_r_k_m[1])),shape=(1,len(net.s_blocks))),
               sp.sparse.csc_matrix((r_m_k[0]*h**2,(np.zeros(len(r_m_k[0])),r_m_k[1])), shape=(1,len(net.s_blocks))),
               sp.sparse.csc_matrix((grad_r_m_k[0]*h**2 ,(np.zeros(len(grad_r_m_k[1])), grad_r_m_k[1])),shape=(1,len(net.s_blocks)))
               )
    
    def get_J_k_m(self, k ,m):
        h=self.mesh_3D.h
        #The following returns the already integrated kernels
        
        rkm, grad_rkm, rmk, grad_rmk = self.get_interface_kernels(k,m)
        
        #return (grad_rmk - grad_rkm)/2/h + (rmk - rkm)/h**2 #This is a line sparse array 
        
        #REMOVE THE FOLLOWING AND REINSTATE THE PREVIOUS
        return  (rmk - rkm)/h**2 #This is a line sparse array 
    
# =============================================================================
#     def get_I_1(self, x,nodes, dual_neigh):
#         """Returns the kernels of the already interpolated part, il reste juste ajouter
#         le term rapide corrigé
#             - x is the point where the concentration is interpolated"""
#         
#         if len(nodes)==8:
#             
#             weights=trilinear_interpolation(x, np.array([self.h, self.h, self.h]))
#             if np.any(weights<0): pdb.set_trace()
#             kernel_q=np.array([])
#             kernel_C_v=np.array([])
#             kernel_s=np.array([])
#             col_q=np.array([], dtype=int)
#             col_C_v=np.array([], dtype=int)
#             col_s=np.array([], dtype=int)
#             for i in range(8): #Loop through each of the nodes
#             
#                 #This is to debug:
#                 #nodes[i].weight=weights[i]
#             
#                 U=get_uncommon(dual_neigh, nodes[i].neigh) 
#                 #The following variable will contain the data kernel for q, the data kernel
#                 #for C_v and the col kernel i.e. the sources 
#                 a=self.mesh_1D.kernel_point(x, U, get_source_potential, self.K, self.D)
#                 
#                 nodes[i].kernel_q=np.concatenate((nodes[i].kernel_q, a[0]))
#                 nodes[i].kernel_C_v=np.concatenate((nodes[i].kernel_C_v, a[1]))
#                 
#                 nodes[i].col_q=np.concatenate((nodes[i].col_q, a[2]))
#                 if np.any(a[1]): 
#                     nodes[i].col_C_v=np.concatenate((nodes[i].col_C_v, a[2]))
#                 
#                 nodes[i].kernel_s=np.array([1], dtype=float)
#                 nodes[i].col_s=np.array([nodes[i].block_3D])
#                 
#                 #This operation is a bit redundant
#                 nodes[i].multiply_by_value(weights[i])
#                 
#                 kernel_q=np.concatenate((kernel_q, nodes[i].kernel_q))
#                 kernel_C_v=np.concatenate((kernel_C_v, nodes[i].kernel_C_v))
#                 kernel_s=np.concatenate((kernel_s, nodes[i].kernel_s))
#                 
#                 col_q=np.concatenate((col_q, nodes[i].col_q))
#                 col_C_v=np.concatenate((col_C_v, nodes[i].col_C_v))
#                 col_s=np.concatenate((col_s, nodes[i].col_s))
#             return kernel_s,col_s,kernel_q, col_q,kernel_C_v,  col_C_v
#         
#         else: #There are not 8 nodes cause it lies in the boundary so there is no interpolation
#             return np.array([1]), np.array([nodes[0].block_3D]), np.array([]), np.array([]), np.array([]),np.array([])
#         
# =============================================================================
    
        
        
def get_I_1(x,nodes, dual_neigh, K, D, h_3D, mesh_1D_object):
    """Returns the kernels of the already interpolated part, il reste juste ajouter
    le term rapide corrigé
        - x is the point where the concentration is interpolated"""
    
    if len(nodes)==8:
        
        weights=trilinear_interpolation(x, np.array([h_3D, h_3D, h_3D]))
        if np.any(weights<0): pdb.set_trace()
        kernel_q=np.array([])
        kernel_C_v=np.array([])
        kernel_s=np.array([])
        col_q=np.array([], dtype=int)
        col_C_v=np.array([], dtype=int)
        col_s=np.array([], dtype=int)
        for i in range(8): #Loop through each of the nodes
           
            U=get_uncommon(dual_neigh, nodes[i].neigh) 
            #The following variable will contain the data kernel for q, the data kernel
            #for C_v and the col kernel i.e. the sources 
            a=mesh_1D_object.kernel_point(x, U, get_source_potential, K, D)
            ######## Changed sign on a[0] 26 mars 18:08
            nodes[i].kernel_q=np.concatenate((nodes[i].kernel_q, -a[0]))
            nodes[i].kernel_C_v=np.concatenate((nodes[i].kernel_C_v, a[1]))
            
            nodes[i].col_q=np.concatenate((nodes[i].col_q, a[2]))
            if np.any(a[1]): 
                nodes[i].col_C_v=np.concatenate((nodes[i].col_C_v, a[2]))
            
            nodes[i].kernel_s=np.array([1], dtype=float)
            nodes[i].col_s=np.array([nodes[i].block_3D])
            
            #This operation is a bit redundant
            nodes[i].multiply_by_value(weights[i])
            
            kernel_q=np.concatenate((kernel_q, nodes[i].kernel_q))
            kernel_C_v=np.concatenate((kernel_C_v, nodes[i].kernel_C_v))
            kernel_s=np.concatenate((kernel_s, nodes[i].kernel_s))
            
            col_q=np.concatenate((col_q, nodes[i].col_q))
            col_C_v=np.concatenate((col_C_v, nodes[i].col_C_v))
            col_s=np.concatenate((col_s, nodes[i].col_s))
        return nodes, kernel_s,col_s,kernel_q, col_q,kernel_C_v,  col_C_v
    
    else: #There are not 8 nodes cause it lies in the boundary so there is no interpolation
        return nodes, np.array([1]), np.array([nodes[0].block_3D]), np.array([]), np.array([]), np.array([]),np.array([])
            
    
        
        

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
        self.weight=value
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
                                
        
        