#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:09:23 2023

@author: pdavid
"""
import numpy as np 
import scipy as sp 
import pdb
from small_functions import append_sparse
from scipy.sparse import csc_matrix

from scipy.sparse.linalg import spsolve as dir_solve


def pre_processing_network(U, init, end, pos_vertex):
    """This function must be called before the assembly of the 1D matrices because
    they need all velocities to be positive for the proper assembly of the bifurcation
    model"""
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

def full_adv_diff_1D(U, D, h, cells, init, vertex_to_edge, R, BCs):
    sparse_arrs=assemble_transport_1D(U, D, h, cells)
    sparse_arrs, ind_array, DoF_list=assemble_vertices(U, D, h, cells, sparse_arrs, vertex_to_edge, R, init, BCs)
    
    return sparse_arrs, ind_array, DoF_list

def assemble_transport_1D(U, D, h, cells):
    """Assembles the linear matrix for convection-dispersion for a network only for the inner DoFs"""
    
    sparse_arrs=np.array([]), np.array( []), np.array([])
    for ed in range(len(cells)):
        initial=np.sum(cells[:ed])
        for i in initial+np.arange(cells[ed]-2)+1:
            sparse_arrs =append_sparse(sparse_arrs, np.array([-U[ed]-D/h[ed],U[ed]+ 2*D/h[ed], -D/h[ed]]), np.array([i,i,i]), np.array([i-1,i,i+1]))
    return sparse_arrs         
    

       
def assemble_vertices(U, D, h, cells, sparse_arrs, vertex_to_edge, R, init, BCs):

    #BCs is a two dimensional array where the first entry is the vertex and the second the value of the Dirichlet BC
    
    #We create here the independent array to store the BCs:
    ind_array=np.zeros(np.sum(cells))
    DoF_list=[] #List that stores the DoF that are in contact with this bifurcation
    vertex=0 #counter that indicates which vertex we are on
    for i in vertex_to_edge:
        if len(i)==1: #Boundary condition
            #Mount the boundary conditions here
            ed=i[0]
        # =============================================================================
        #     sparse_arrs=append_sparse(sparse_arrs, np.array([-U/2-D/h,-U/2+3*D/h]), np.array([cells-1,cells-1]), np.array([cells-2,cells-1]))
        #     final_value_Dirichlet=U-2*D/h
        # =============================================================================
            if init[i]!=vertex: #Then it must be the end Vertex of the edge 
                current_DoF=np.sum(cells[:ed])+cells[ed]-1 #End vertex 
                kk=-1
            else: 
                current_DoF=np.sum(cells[:ed]) #initial vertex
                kk=1
            
            sparse_arrs=append_sparse(sparse_arrs, np.array([-U[ed]-D/h[ed],U[ed]+3*D/h[ed]]), np.array([current_DoF,current_DoF]), np.array([current_DoF+kk,current_DoF]))
            value_Dirichlet=-2*D/h[ed]
            ind_array[current_DoF]=BCs[np.where(BCs[:,0]==vertex)[0][0],1]*value_Dirichlet #assigns the value of the BC multiplied by the factor 
            
        else: #Bifurcation between two or three vessels
            #pdb.set_trace()
            den=0
            num=np.zeros(len(i))
            
            exiting=np.array([], dtype=int) #The edges for which the bifurcation is the initial DoF
            entering=np.array([], dtype=int)   #The edges for which the bifurcation is the final DoF
            
            #Loop to calculate the coefficients of gamma
            c=0
            DoF=np.zeros(len(i), dtype=int) #To store the actual DoF we are working on 
            #We figure out which edges are exiting and which are entering so we
            #can know which have a advective term on the numerator
            for ed in i: #Goes through each of the vessel in the bifurcation
                den+=2*D*np.pi*R[ed]**2/h[ed] #We always add the diffusive term 
                num[c]=2*D*np.pi*R[ed]**2/h[ed] #Same in the numerator
                if init[ed]==vertex: #The bifurcation is the initial vertex/exiting vessel
                    den+=U[ed]*np.pi*R[ed]**2
                    exiting=np.append(exiting, i[c])
                    DoF[c]=np.sum(cells[:ed])
                else:
                    num[c]+=U[ed]*np.pi*R[ed]**2
                    entering=np.append(entering, i[c])
                    DoF[c]=np.sum(cells[:ed])+cells[ed]-1
                
                c+=1
            
            gamma=num/den
            DoF=DoF.astype(int)
            DoF_list+=list(DoF)
            print(gamma)
            for ed in exiting: #Exiting vessels
                local_ed=np.where(i==ed)[0][0] #position of the edge in list i
            
                current_gamma=gamma[local_ed]
                current_DoF=DoF[local_ed]
                DoF_w=np.delete(DoF, local_ed) #The other two DoFs
                #Exiting flux to the normal neighbouring cylinder 
                sparse_arrs =append_sparse(sparse_arrs, np.array([D/h[ed]+U[ed], -D/h[ed]]), np.array([current_DoF,current_DoF]), np.array([current_DoF,current_DoF+1])) 
                #Bifurcation Flux
                sparse_arrs =append_sparse(sparse_arrs, np.array([2*D/h[ed]*(1-current_gamma)-U[ed]*current_gamma]), np.array([current_DoF]), np.array([current_DoF]))
                
                for j in DoF_w:
                    vessel=np.where(DoF==j)[0][0] #same as local ed before but I wanted to give a different name. It is the local position of the vessel within the bifurcation (i)
                    sparse_arrs =append_sparse(sparse_arrs, np.array([-2*D/h[ed]*gamma[vessel]-U[ed]*gamma[vessel]]), np.array([current_DoF]), np.array([j]))
                
            for ed in entering:
                local_ed=np.where(i==ed)[0][0]
                
                current_gamma=gamma[local_ed]
                current_DoF=DoF[local_ed]
                #Normal Cylinder
                sparse_arrs =append_sparse(sparse_arrs, np.array([-U[ed]-D/h[ed],D/h[ed]]), np.array([current_DoF,current_DoF]), np.array([current_DoF-1,current_DoF]))
                DoF_w=np.delete(DoF, local_ed) #The other two DoFs
                
                #Bifurcation Flux
                sparse_arrs =append_sparse(sparse_arrs, np.array([2*D/h[ed]*(1-current_gamma)+U[ed]]), np.array([current_DoF]), np.array([current_DoF]))
                
                for j in DoF_w:
                    vessel=np.where(DoF==j)[0][0]
                    sparse_arrs =append_sparse(sparse_arrs, np.array([-2*D/h[ed]*gamma[vessel]]), np.array([current_DoF]), np.array([j]))
            
        vertex+=1
        
    return sparse_arrs, ind_array, DoF_list

#analytical=(np.exp(-Pe*x)-np.exp(-Pe))/(1-np.exp(-Pe))



   
    