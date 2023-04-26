#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 16:17:50 2023

@author: pdavid
"""
import numpy as np 
import scipy as sp 
import pdb
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
    

    return sparse_arrs

def assemble_Dirichlet_1D(sparse_arrs, U, D, h, cells):
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
    
def assemble_vertices(U, D, h, cells, sparse_arrs, vertex_to_edge, R):
    pdb.set_trace()
    for i in vertex_to_edge:
        if len(i)==1: #Boundary condition
            #Mount the boundary conditions here
            ed=[i[0]]
            sparse_arrs=append_sparse(sparse_arrs, np.array([U[ed]+3*D/h, -D/h]), np.array([0,0]), np.array([0,1]))
            initial_value_Dirichlet=-U[ed]-2*D/h
            
        # =============================================================================
        #     sparse_arrs=append_sparse(sparse_arrs, np.array([-U/2-D/h,-U/2+3*D/h]), np.array([cells-1,cells-1]), np.array([cells-2,cells-1]))
        #     final_value_Dirichlet=U-2*D/h
        # =============================================================================
            if i[0]<0: ID=np.sum(cells[:ed+1])-1 #End vertex 
            else: ID=np.sum(cells[:ed]) #initial vertex
            
            sparse_arrs=append_sparse(sparse_arrs, np.array([-U[ed]-D/h,U[ed]+3*D/h]), np.array([cells-1,cells-1]), np.array([cells-2,cells-1]))
            final_value_Dirichlet=-2*D/h
            
        else: #Bifurcation between two or three vessels
            den=0
            num=np.zeros(len(i))
            
            initial=np.array([]) #The edges for which the bifurcation is the initial DoF
            final=np.array([])   #The edges for which the bifurcation is the final DoF
            
            #Loop to calculate the coefficients of gamma
            c=0
            DoF=np.zeros(3) #To store the actual DoF we are working on 
            for j in i:
                den+=2*D*np.pi*R[j]**2/h
                num[c]=2*D*np.pi*R[j]**2/h
                if j>1: #The bifurcation is initial
                    den+=U[np.abs(j)]*np.pi*R[j]**2
                    initial=np.append(initial, c)
                    DoF[c]=np.sum(cells[:np.abs(j)])
                else:
                    final=np.append(final, c)
                    DoF[c]=np.sum(cells[:np.abs(j)+1])-1
                
                c+=1
                
            gamma=num/den
            
            for k in initial:
                i=DoF[k]
                DoF_w=np.delete(DoF, k) #The other two DoFs
                #Normal Cylinder
                sparse_arrs =append_sparse(sparse_arrs, np.array([D/h, -D/h]), np.array([i,i]), np.array([i,i+1])) 
                #Bifurcation Flux
                sparse_arrs =append_sparse(sparse_arrs, np.array([2*D/h*(1-gamma[k])-U[k]*gamma[k]]), np.array([i]), np.array([i]))
                
                for j in DoF_w:
                    sparse_arrs =append_sparse(sparse_arrs, np.array([-2*D/h*gamma[j]-U[j]*gamma[j]]), np.array([i]), np.array([j]))
                
            for k in final:
                i=DoF[k]
                #Normal Cylinder
                sparse_arrs =append_sparse(sparse_arrs, np.array([-U-D/h,U+ 2*D/h]), np.array([i,i]), np.array([i-1,i]))
                DoF_w=np.delete(DoF, k) #The other two DoFs
                
                #Bifurcation Flux
                sparse_arrs =append_sparse(sparse_arrs, np.array([2*D/h*(1-gamma[k])]), np.array([i]), np.array([i]))
                
                for j in DoF_w:
                    sparse_arrs =append_sparse(sparse_arrs, np.array([-2*D/h*gamma[j]]), np.array([i]), np.array([j]))
            
            
            
    

def pre_processing_network(U, init, end, pos_vertex):
    vertex_to_edge=[]
    for i in range(len(pos_vertex)):
        if U[i]<0:
            temp=init[i]
            init[i]=end[i]
            end[i]=init[i]
            
        a=np.arange(init)[init==i] #edges for which this vertex is the initial 
        b=np.arange(end)[init==i]  #edges for which this vertex is the end
        
        vertex_to_edge+=[[a.tolist()+(-b).tolist()]]
        
    return init, end, pos_vertex
    


class flow():
    """This class acts as a flow solver, if the veocities (or flow) are imported from another simulation
    this class is not neccesary"""    
    def __init__(self, uid_list, value_list, L, diameters, startVertex, endVertex):
        self.bc_uid=uid_list
        self.bc_value=value_list
        self.L=L
        self.d=diameters
        self.viscosity=0.0012
        self.start=startVertex
        self.end=endVertex
        self.n_vertices=np.max(np.array([np.max(self.start)+1, np.max(self.end)+1]))
        
        
    def solver(self):
        A=np.zeros([self.n_vertices,self.n_vertices])
        P=np.zeros([self.n_vertices])
        for i in range(len(self.start)): #Loop that goes through each edge assembling the pressure matrix
        
            if self.start[i] not in self.bc_uid:
                A[self.start[i],self.start[i]]-=self.d[i]**4/self.L[i]
                A[self.start[i],self.end[i]]+=self.d[i]**4/self.L[i]
            if self.end[i] not in self.bc_uid:
                A[self.end[i],self.end[i]]-=self.d[i]**4/self.L[i]
                A[self.end[i],self.start[i]]+=self.d[i]**4/self.L[i]
        A[self.bc_uid,self.bc_uid]=1
        P[self.bc_uid]=self.bc_value
        
        self.A=A
        self.P=P
        
        return(A)
    
    def get_U(self):
        """Computes and returns the speed from the pressure values that have been previously computed"""
        pressures=np.linalg.solve(self.solver(), self.P)
        U=np.array([])
        for i in range(len(self.start)):
            vel=self.d[i]**2*(pressures[self.start[i]]-pressures[self.end[i]])/(32*self.viscosity*self.L[i])
            U=np.append(U,vel)
        self.P=pressures
        return(U)
       
                
#analytical=(np.exp(-Pe*x)-np.exp(-Pe))/(1-np.exp(-Pe))
U=np.array([1,1,1])
D=2
cells=np.array([8,9,10])
sparse_arrs=np.array([np.array([]),np.array([]),np.array([])])
vertex_to_edge=[[0],[0,1,2],[1],[2]]
R=np.array([1,2,3])
h=1
assemble_vertices(U, D, h, cells, sparse_arrs, vertex_to_edge, R)
            
    
    
    
