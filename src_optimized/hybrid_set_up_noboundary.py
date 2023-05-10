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
from small_functions import for_boundary_get_normal, append_sparse
from scipy.sparse.linalg import spsolve as dir_solve
from assembly import Assembly_diffusion_3D_boundaries, Assembly_diffusion_3D_interior

import scipy as sp
from scipy.sparse import csc_matrix

import multiprocessing
from multiprocessing import Pool

from assembly_1D import full_adv_diff_1D



import matplotlib.pylab as pylab
plt.style.use('default')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15,15),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large', 
         'font.size': 24,
         'lines.linewidth': 2,
         'lines.markersize': 15}
pylab.rcParams.update(params)

class hybrid_set_up():
    def __init__(self, mesh_3D, mesh_1D,  BC_type, BC_value,n, D, K, BCs_1D):
        
        self.BCs_1D=BCs_1D
        self.K=K
        
        self.mesh_3D=mesh_3D
        self.h=mesh_3D.h
        
        self.mesh_1D=mesh_1D
        self.R=mesh_1D.R
        self.D=D
        self.BC_type=BC_type
        self.BC_value=BC_value
        
        self.n=n
        
        self.mesh_3D.get_ordered_connect_matrix()
        
        self.F=self.mesh_3D.size_mesh
        self.S=len(mesh_1D.pos_s)
        
        return
    
    def Assembly_A_B_C(self):
        A_matrix=self.Assembly_A()
        self.A_matrix=A_matrix
        
        
        B_matrix=self.Assembly_B()
        B_matrix=self.Assembly_B_boundaries(B_matrix)
        self.B_matrix=B_matrix
        
        C_matrix=csc_matrix((self.mesh_3D.size_mesh, len(self.mesh_1D.pos_s)))
        self.C_matrix=C_matrix
        
        A_B_C=sp.sparse.hstack((A_matrix, B_matrix, C_matrix))
        self.A_B_C_matrix=A_B_C
        return(A_B_C)
    
    def Interp_sources(self):
        """This function provides the interpolation to the sources """
    
    def Assembly_D_E_F(self):
        
        D=np.zeros([3,0])
        E=np.zeros([3,0])
        F=np.zeros([3,0])
        
        #The following are important matrices that are interesting to keep separated
        #Afterwards they can be used to assemble the D, E, F matrices
        G_ij=np.zeros([3,0])
        H_ij=np.zeros([3,0])
        Permeability=np.zeros([3,0])
        
        for j in range(len(self.mesh_1D.s_blocks)):
            kernel_s,col_s,kernel_q, col_q,kernel_C_v,  col_C_v=self.interpolate(self.mesh_1D.pos_s[j])
            D=append_sparse(D, kernel_s,np.zeros(len(col_s))+j, col_s)
            E=append_sparse(E, kernel_q,np.zeros(len(col_q))+j, col_q)
            
            G_ij=append_sparse(G_ij, kernel_q,np.zeros(len(col_q))+j, col_q)
            
            if len(kernel_C_v)!=len(col_C_v): pdb.set_trace()
            
            F=append_sparse(F, kernel_C_v,np.zeros(len(col_C_v))+j, col_C_v)
            H_ij=append_sparse(H_ij, kernel_C_v,np.zeros(len(col_C_v))+j, col_C_v)
            
            E=append_sparse(E, 1/self.K[self.mesh_1D.source_edge[j]] , j, j)
            Permeability=append_sparse(Permeability, 1/self.K[self.mesh_1D.source_edge[j]] , j, j)
        F=append_sparse(F, -np.ones(len(self.mesh_1D.s_blocks)) , np.arange(len(self.mesh_1D.s_blocks)), np.arange(len(self.mesh_1D.s_blocks)))
            
        self.D_matrix=csc_matrix((D[0], (D[1], D[2])), shape=(len(self.mesh_1D.pos_s), self.mesh_3D.size_mesh))
        self.E_matrix=csc_matrix((E[0], (E[1], E[2])), shape=(len(self.mesh_1D.pos_s), len(self.mesh_1D.pos_s)))
        self.F_matrix=csc_matrix((F[0], (F[1], F[2])), shape=(len(self.mesh_1D.pos_s), len(self.mesh_1D.pos_s)))
        
        self.G_ij=csc_matrix((G_ij[0], (G_ij[1], G_ij[2])), shape=(len(self.mesh_1D.pos_s), len(self.mesh_1D.pos_s)))
        self.H_ij=csc_matrix((H_ij[0], (H_ij[1], H_ij[2])), shape=(len(self.mesh_1D.pos_s), len(self.mesh_1D.pos_s)))
        self.Permeability=csc_matrix((Permeability[0], (Permeability[1], Permeability[2])), shape=(len(self.mesh_1D.pos_s), len(self.mesh_1D.pos_s)))
        
        return sp.sparse.hstack((self.D_matrix, self.E_matrix, self.F_matrix))
    
    def Assembly_G_H_I(self):
        I_matrix=self.Assembly_I()
        
        #WILL CHANGE WHEN CONSIDERING MORE THAN 1 VESSEL
        aux_arr=np.zeros(len(self.mesh_1D.pos_s))
        #H matrix for multiple vessels
        for ed in range(len(self.R)): #Loop through every vessel
            DoFs=np.arange(np.sum(self.mesh_1D.cells[:ed]),np.sum(self.mesh_1D.cells[:ed])+np.sum(self.mesh_1D.cells[ed])) #DoFs belonging to this vessel
            aux_arr[DoFs]=self.mesh_1D.h[ed]/(np.pi*self.mesh_1D.R[ed]**2)
        H_matrix=sp.sparse.diags(aux_arr, 0)
        self.H_matrix=H_matrix
        
        #Matrix full of zeros:
        G_matrix=csc_matrix(( len(self.mesh_1D.pos_s),self.mesh_3D.size_mesh))
        self.G_matrix=G_matrix
        self.G_H_I_matrix=sp.sparse.hstack((G_matrix, H_matrix, I_matrix))
        
        return(self.G_H_I_matrix)
    
    def reAssembly_matrices(self):
        Upper=sp.sparse.hstack((self.A_matrix, self.B_matrix, self.C_matrix))
        Middle=sp.sparse.hstack((self.D_matrix, self.E_matrix, self.F_matrix))
        Down=sp.sparse.hstack((self.G_matrix, self.H_matrix, self.I_matrix))
        
        Full_linear_matrix=sp.sparse.vstack((Upper,
                                             Middle,
                                             Down))
        
        return Full_linear_matrix
    
    def Assembly_problem(self):
        Full_linear_matrix=sp.sparse.vstack((self.Assembly_A_B_C(),
                                             self.Assembly_D_E_F(),
                                             self.Assembly_G_H_I()))
        self.Full_linear_matrix=Full_linear_matrix
        
        self.Full_ind_array=np.concatenate((self.I_ind_array, np.zeros(len(self.mesh_1D.pos_s)), self.III_ind_array))
        return
    def Solve_problem(self):
        sol=dir_solve(self.Full_linear_matrix, -self.Full_ind_array)
        self.s=sol[:self.mesh_3D.size_mesh]
        self.q=sol[self.mesh_3D.size_mesh:-self.S]
        self.Cv=sol[-self.S:]
        return
        
    def Assembly_A(self):
        """An h is missing somewhere to be consistent"""
        size=self.mesh_3D.size_mesh
        
        a=Assembly_diffusion_3D_interior(self.mesh_3D)
        b=Assembly_diffusion_3D_boundaries(self.mesh_3D, self.BC_type, self.BC_value)
        
        # =============================================================================
        #         NOTICE HERE, THIS IS WHERE WE MULTIPLY BY h so to make it dimensionally consistent relative to the FV integration
        A_matrix=csc_matrix((a[2]*self.mesh_3D.h, (a[0], a[1])), shape=(size,size)) + csc_matrix((b[2]*self.mesh_3D.h, (b[0], b[1])), shape=(size, size))
        #       We only multiply the non boundary part of the matrix by h because in the boundaries assembly we need to include the h due to the difference
        #       between the Neumann and Dirichlet boundary conditions. In short Assembly_diffusion_3D_interior returns data that needs to be multiplied by h 
        #       while Assembly_diffusion_3D_boundaries is already dimensionally consistent
        # =============================================================================
        self.I_ind_array=b[3]*self.mesh_3D.h
        
        
        return A_matrix
    
    def Assembly_B(self):
        """Assembly of the B matrix i.e. computation of the arrays J_k_m and 
        adding them to the B matrix"""
        B=csc_matrix((0, len(self.mesh_1D.s_blocks)))
        for k in range(self.mesh_3D.size_mesh):
            N_k= self.mesh_3D.ordered_connect_matrix[k] #Set with all the neighbours
            
            J_k=0
            
            for m in N_k:
                J_k+=self.get_J_k_m(k,m)
                #if np.sum(J_k.toarray())!=0: print(k,m)
            B=sp.sparse.vstack((B, J_k))
            
        #HERE IS WHERE WE MULTIPLY BY H!!!!!
        self.B_matrix=B*self.mesh_3D.h

        return self.B_matrix

            
    def Assembly_B_boundaries(self, B):

        B=B.tolil()
        
        normals=np.array([[0,0,1],  #for north boundary
                          [0,0,-1], #for south boundary
                          [0,1,0],  #for east boundary
                          [0,-1,0], #for west boundary
                          [1,0,0],  #for top boundary 
                          [-1,0,0]])#for bottom boundary
        mesh=self.mesh_3D
        h=mesh.h
        c=0
        
        for bound in self.mesh_3D.full_full_boundary:
        #This loop goes through each of the boundary cells, and it goes repeatedly 
        #through the edges and corners accordingly
            for k in bound: #Make sure this is the correct boundary variable
                k_neigh=get_neighbourhood(self.n, mesh.cells_x, mesh.cells_y, mesh.cells_z, k)
                
                pos_k=mesh.get_coords(k)
                normal=normals[c]
                pos_boundary=pos_k+normal*h/2
                if self.BC_type[c]=="Dirichlet":
                    r_k=self.mesh_1D.kernel_integral_surface(pos_boundary, normal,  k_neigh, 'P', self.K,self.D, self.mesh_3D.h)
                    
                if self.BC_type[c]=="Neumann":
                    r_k=self.mesh_1D.kernel_integral_surface(pos_boundary, normal,  k_neigh, 'G', self.K,self.D, self.mesh_3D.h)
                
                kernel=csc_matrix((r_k[0]*h**2,(np.zeros(len(r_k[0])),r_k[1])), shape=(1,len(self.mesh_1D.s_blocks)))
                B[k,:]-=kernel
            c+=1
            
        return(B)
        
     
    
    def Assembly_I(self):
        """Models intravascular transport. Advection-diffusion equation
        
        FOR NOW IT ONLY HANDLES A SINGLE VESSEL"""
        
        D=self.mesh_1D.D
        U=self.mesh_1D.U
        L=self.mesh_1D.L
        aa, ind_array, DoF=full_adv_diff_1D(U, D, self.mesh_1D.h, self.mesh_1D.cells, self.mesh_1D.startVertex, self.mesh_1D.vertex_to_edge, self.R, self.BCs_1D)
# =============================================================================
#         aa, bb =assemble_transport_1D(U, D, L[0]/len(self.mesh_1D.pos_s), len(self.mesh_1D.pos_s))
#         self.III_ind_array = np.zeros(len(self.mesh_1D.pos_s))
#         self.III_ind_array [0] = bb[0]
# =============================================================================
        self.III_ind_array=ind_array
        I = csc_matrix((aa[0], (aa[1], aa[2])), shape=(len(self.mesh_1D.pos_s), len(self.mesh_1D.pos_s)))
        
        self.I_matrix=I
        
        return I

    def get_coord_reconst(self,corners, resolution):
        """Corners given in order (0,0),(0,1),(1,0),(1,1)
        
        This function is to obtain the a posteriori reconstruction of the field
        
        OUTDATED, NOT PARALLEL"""
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
            
    

    
    def get_coord_reconst_chat(self, corners, resolution, num_processes=4):
        """Corners given in order (0,0),(0,1),(1,0),(1,1)
        
        This function is to obtain the a posteriori reconstruction of the field"""
        print("Number of processes= ", num_processes)
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
        
# =============================================================================
#         # Create a pool of worker processes
#         pool = Pool(processes=num_processes)
#         
#         # Use map function to apply interpolate_helper to each coordinate in parallel
#         results = pool.map(interpolate_helper, [(self, k) for k in crds])
#         
#         # Close the pool to free up resources
#         pool.close()
#         pool.join()
#         
#         # Convert the results to a numpy array
#         rec = np.array(results)
# =============================================================================
        rec=np.array([])        
        for k in crds:
            rec=np.append(rec, interpolate_helper((self, k)))
        
        return crds, rec     
    
    def get_coord_reconst_center(self, corners):
        """Corners given in order (0,0),(0,1),(1,0),(1,1)
        
        Returns the coarse reconstruction, cell center"""
        
        tau_1=corners[1]-corners[0]
        tau_1=tau_1/np.linalg.norm(tau_1)
        tau_2=corners[2]-corners[0]
        tau_2=tau_2/np.linalg.norm(tau_2)
        
        init=self.mesh_3D.get_id(corners[0]) #ID of the lowest block
        init_coords=self.mesh_3D.get_coords(init)
        crds=init_coords.copy()
        
        end_1=self.mesh_3D.get_id(corners[1]) #ID of the upper corner
        end_1_coords=self.mesh_3D.get_coords(end_1)
        end_2=self.mesh_3D.get_id(corners[2]) #ID of the upper corner
        end_2_coords=self.mesh_3D.get_coords(end_2)
        
        L1=end_1_coords-init
        L2=end_2_coords-init
        v=np.array([])
        for i in range(int(end_2-init)):
            for j in range(int(end_1-init)):
                cr=init_coords+self.mesh_3D.h*i*tau_2+self.mesh_3D.h*j*tau_1
                
                crds=np.vstack((crds, cr))
                
                k=self.mesh_3D.get_id(cr) #Current block
                
                if np.any(self.mesh_3D.get_coords(k)-cr): print("ERORROOROROROROR")
                
                a,b,c,d,_,_=self.interpolate(cr)
                
                value=0
                print(a)
                value+=np.dot(self.q[d], c)
                v=np.append(v, value)
        return crds, v  
    
    def rec_along_mesh(self,axis, crds_along_axis, s_field, q, C_v_array):
        """Outdated"""
        
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
        """For some reason, when the discretization size of the network is too small, 
        this function provides artifacts. I have not solved this yet"""
        #From the nodes coordinates, their neighbourhoods and their kernels we do the interpolation
        #Therefore, the kernels represent the slow term, the corrected rapid term must be calculated
        
        #INTERPOLATED PART:
        #kernel_s,col_s,kernel_q, col_q,kernel_C_v,  col_C_v=self.get_I_1(x, nodes, dual_neigh)
        self.nodes,kernel_s,col_s,kernel_q, col_q,kernel_C_v,  col_C_v=get_I_1(x,nodes, dual_neigh, self.K, self.D,
                                                                    self.mesh_3D.h, self.mesh_1D)
        
        #kernel_s,col_s,kernel_q, col_q,kernel_C_v,  col_C_v=np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
        #RAPID (NON INTERPOLATED) PART
        #q,C,sources=self.mesh_1D.kernel_point(x, dual_neigh, 'P', self.K, self.D)
        q,sources=self.mesh_1D.kernel_point(x, dual_neigh, 'P', self.K, self.D)
        kernel_q=np.concatenate((kernel_q, q))
        #kernel_C_v=np.concatenate((kernel_C_v, C))
        col_q=np.concatenate((col_q, sources))
        #col_C_v=np.concatenate((col_C_v, sources))
        
        
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
        
        #if np.linalg.norm(normal) > 1: print('ERROR!!!!!!!!!!!!!!!!!')
        if np.linalg.norm(normal) > 1.0000001: pdb.set_trace()
        
        r_k_m=net.kernel_integral_surface(pos_m/2+pos_k/2, normal,  get_uncommon(k_neigh, m_neigh), 'P', self.K,self.D, self.mesh_3D.h)
        r_m_k=net.kernel_integral_surface(pos_m/2+pos_k/2, normal,  get_uncommon(m_neigh, k_neigh), 'P', self.K,self.D, self.mesh_3D.h)
        
        #if np.sum(r_m_k[0]) and np.sum(r_k_m[0]): pdb.set_trace()
        
        grad_r_k_m=net.kernel_integral_surface(pos_m/2+pos_k/2, normal, get_uncommon(k_neigh, m_neigh), 'G', self.K,self.D, self.mesh_3D.h)
        grad_r_m_k=net.kernel_integral_surface(pos_m/2+pos_k/2, normal, get_uncommon(m_neigh, k_neigh),  'G', self.K,self.D, self.mesh_3D.h)
        #NOTICE HOW THE VALUES ARE MULTIPLIED BY h**2 (SURFACE OF THE INTERFACE) DOWN HERE SINCE THE FUNCTION SAMPSON DOES NOT DO IT
        return(csc_matrix((r_k_m[0]*h**2,(np.zeros(len(r_k_m[0])),r_k_m[1])), shape=(1,len(net.s_blocks))),
               csc_matrix((grad_r_k_m[0]*h**2 ,(np.zeros(len(grad_r_k_m[1])), grad_r_k_m[1])),shape=(1,len(net.s_blocks))),
               csc_matrix((r_m_k[0]*h**2,(np.zeros(len(r_m_k[0])),r_m_k[1])), shape=(1,len(net.s_blocks))),
               csc_matrix((grad_r_m_k[0]*h**2 ,(np.zeros(len(grad_r_m_k[1])), grad_r_m_k[1])),shape=(1,len(net.s_blocks)))
               )
    
    def Assemble_interface_kernels_optimized(self, k, m):
        
        net=self.mesh_1D #1D mesh object that contains the functions to calculate the integral kernels
        mesh=self.mesh_3D
        
        r_k_m, r_m_k, grad_r_k_m, grad_r_m_k=get_interface_kernels_optimized(k,m,mesh.get_coords(k), mesh.get_coords(m),mesh.h, self.n, mesh.cells_x, mesh.cells_y, mesh.cells_z,
                                            net.s_blocks, net.tau, net.h, net.pos_s, net.source_edge,self.D )
        return(csc_matrix((r_k_m[0]*mesh.h**2,(np.zeros(len(r_k_m[0])),r_k_m[1])), shape=(1,len(net.s_blocks))),
               csc_matrix((grad_r_k_m[0]*mesh.h**2 ,(np.zeros(len(grad_r_k_m[1])), grad_r_k_m[1])),shape=(1,len(net.s_blocks))),
               csc_matrix((r_m_k[0]*mesh.h**2,(np.zeros(len(r_m_k[0])),r_m_k[1])), shape=(1,len(net.s_blocks))),
               csc_matrix((grad_r_m_k[0]*mesh.h**2 ,(np.zeros(len(grad_r_m_k[1])), grad_r_m_k[1])),shape=(1,len(net.s_blocks)))
               )
    
    def get_J_k_m(self, k ,m):
        h=self.mesh_3D.h
        #The following returns the already integrated kernels
        
        rkm, grad_rkm, rmk, grad_rmk = self.get_interface_kernels(k,m)
        #if np.sum(rkm)!=0: pdb.set_trace()
        return (grad_rmk - grad_rkm)/2/h + (rmk - rkm)/h**2 #This is a line sparse array 
        

    
    def get_point_value_post(self, coords, rec,k):
        a,b,c,d,e,f=self.interpolate(coords)
        rec[k]=a.dot(self.s[b])+c.dot(self.q[d])
        return 
    
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
            a=mesh_1D_object.kernel_point(x, U, 'P', K, D)
            ######## Changed sign on a[0] 26 mars 18:08
            #I think the negative sign arises from the fact that this is the corrected part of the rapid term so it needs to be subtracted
            
            nodes[i].kernel_q=np.concatenate((nodes[i].kernel_q, -a[0]))
            #nodes[i].kernel_C_v=np.concatenate((nodes[i].kernel_C_v, a[1]))
            
            #nodes[i].col_q=np.concatenate((nodes[i].col_q, a[2]))
            nodes[i].col_q=np.concatenate((nodes[i].col_q, a[1]))
# =============================================================================
#             if len(a[1]): 
#                 nodes[i].col_C_v=np.concatenate((nodes[i].col_C_v, a[2]))
# =============================================================================
            
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
            
            if len(kernel_C_v)!=len(col_C_v): pdb.set_trace()
            if len(kernel_q)!=len(col_q): pdb.set_trace()
            if len(kernel_s)!=len(col_s): pdb.set_trace()
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
                                

def interpolate_helper(args):
    self, k = args
    a,b,c,d,e,f = self.interpolate(k)
    return a.dot(self.s[b]) + c.dot(self.q[d])        

class visualization_3D():
    def __init__(self,lim, res, prob, num_proc, vmax, *trans):
        
        self.vmax=vmax
        self.vmin=0
        self.lim=lim
        self.res=res
        a=(lim[1]-lim[0])*np.array([1,2,3])/4
        LIM_1=[lim[0], lim[0], lim[1], lim[1]]
        LIM_2=[lim[0], lim[1], lim[0], lim[1]]
        
        perp_x=np.zeros([3,4,3])
        for i in range(3):
            perp_x[i]=np.array([a[i]+np.zeros(4),LIM_1 , LIM_2]).T
            
        perp_y, perp_z=perp_x.copy(),perp_x.copy()
        
        perp_y[:,:,0]=perp_x[:,:,1]
        perp_y[:,:,1]=perp_x[:,:,0]
        
        perp_z[:,:,0]=perp_x[:,:,2]
        perp_z[:,:,2]=perp_x[:,:,0]
        
        if trans:
            perp_x+=trans
            perp_y+=trans
            perp_z+=trans
            
            
        data=np.empty([9, res, res])
        for i in range(3):
            for j in range(3):
                #cor are the corners of a square 
                if i==0: cor=perp_x
                if i==1: cor=perp_y
                if i==2: cor=perp_z
                
                a,b=prob.get_coord_reconst_chat(cor[j], res, num_processes=num_proc)
                
                data[i*3+j]=b.reshape(res, res)
                
        self.data=data
        
        self.perp_x=perp_x
        self.perp_y=perp_y
        self.perp_z=perp_z
        
        self.plot(data, lim)
        
        return
    
    def plot(self, data, lim):
        
        res=self.res
        perp_x, perp_y, perp_z=self.perp_x, self.perp_y, self.perp_z
        
        # Create a figure with 3 rows and 3 columns
        fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18,18))
        
        # Set the titles for each row of subplots
        row_titles = ['X', 'Y', 'Z']
        
        # Set the titles for each individual subplot
        subplot_titles = ['x={:.2f}'.format(perp_x[0,0,0]), 'x={:.2f}'.format(perp_x[1,0,0]), 'x={:.2f}'.format(perp_x[2,0,0]),
                          'y={:.2f}'.format(perp_y[0,1,1]), 'y={:.2f}'.format(perp_y[1,1,1]), 'y={:.2f}'.format(perp_y[2,1,1]),
                          'z={:.2f}'.format(perp_z[0,0,2]), 'z={:.2f}'.format(perp_z[1,0,2]), 'z={:.2f}'.format(perp_z[2,2,2])]
        
        
        # Loop over each row of subplots
        for i, ax_row in enumerate(axs):
            # Set the title for this row of subplots
            ax_row[0].set_title(row_titles[i], fontsize=16)
            
            # Loop over each subplot in this row
            for j, ax in enumerate(ax_row):
                # Plot some data in this subplot
                x = [1, 2, 3]
                y = [1, 4, 9]
                                             
                b=self.data[i*3+j]
                
                im=ax.imshow(b.reshape(res,res), origin='lower', vmax=self.vmax, vmin=self.vmin, extent=[lim[0], lim[1], lim[0], lim[1]])
                # Set the title and y-axis label for this subplot
                ax.set_title(subplot_titles[i*3 + j], fontsize=14)
                if i==0: ax.set_ylabel('z'); ax.set_xlabel('y')
                if i==1: ax.set_ylabel('z'); ax.set_xlabel('x')
                if i==2: ax.set_ylabel('x'); ax.set_xlabel('y')
                
        # Set the x-axis label for the bottom row of subplots
        #axs[-1, 0].set_xlabel('X-axis', fontsize=12)
        
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        
        # Show the plot
        plt.show()
    
from mesh_1D import kernel_integral_surface_optimized
def get_interface_kernels_optimized(k,m,coords_k, coords_m,h_3D, n, cells_x, cells_y, cells_z,
                                    s_blocks, tau, h_1D, pos_s, source_edge,D ):
    
    k_neigh=get_neighbourhood(n, cells_x, cells_y, cells_z, k)
    m_neigh=get_neighbourhood(n, cells_x, cells_y, cells_z, m)
    
    
    pos_k=coords_k
    pos_m=coords_m
    normal=(pos_m-pos_k)/h_3D
    
    #if np.linalg.norm(normal) > 1: print('ERROR!!!!!!!!!!!!!!!!!')
    if np.linalg.norm(normal) > 1.0000001: pdb.set_trace()
    
    r_k_m=kernel_integral_surface_optimized(s_blocks, tau, h_1D, pos_s, source_edge,
                                            pos_m/2+pos_k/2, normal,  get_uncommon(k_neigh, m_neigh), 'P', D, h_3D)
    r_m_k=kernel_integral_surface_optimized(s_blocks, tau, h_1D, pos_s, source_edge,
                                      pos_m/2+pos_k/2, normal,  get_uncommon(m_neigh, k_neigh), 'P', D, h_3D)
    
    #if np.sum(r_m_k[0]) and np.sum(r_k_m[0]): pdb.set_trace()
    
    grad_r_k_m=kernel_integral_surface_optimized(s_blocks, tau, h_1D, pos_s, source_edge,
                                                 pos_m/2+pos_k/2, normal, get_uncommon(k_neigh, m_neigh), 'G', D, h_3D)
    grad_r_m_k=kernel_integral_surface_optimized(s_blocks, tau, h_1D, pos_s, source_edge,
                                                 pos_m/2+pos_k/2, normal, get_uncommon(m_neigh, k_neigh),  'G', D, h_3D)

    return(r_k_m, r_m_k, grad_r_k_m, grad_r_m_k)
# =============================================================================
# def get_coord_reconst_parallel(problem_object,corners, resolution):
#     """Corners given in order (0,0),(0,1),(1,0),(1,1)
#     
#     This function is to obtain the a posteriori reconstruction of the field"""
#     crds=np.zeros((0,3))
#     tau=(corners[2]-corners[0])/resolution
#     
#     h=(corners[1]-corners[0])/resolution
#     L_h=np.linalg.norm((corners[1]-corners[0]))
#     
#     local_array= np.linspace(corners[0]+h/2, corners[1]-h/2 , resolution )
#     for j in range(resolution):
#         arr=local_array.copy()
#         arr[:,0]+=tau[0]*(j+1/2)
#         arr[:,1]+=tau[1]*(j+1/2)
#         arr[:,2]+=tau[2]*(j+1/2)
#         
#     crds=np.vstack((crds, arr))
#     with multiprocessing.Manager() as manager:
#         arr = manager.list(np.zeros(len(crds)))
#         with multiprocessing.Pool(processes=10) as pool:
#             pool.starmap(problem_object.get_point_value_post, [(crds[c],arr,c) for c in range(len(crds))])
#         rec = np.array(arr)
# 
# # =============================================================================
# #     for c in range(len(crds)):
# #         rec = problem_object.get_point_value_post(crds[c],rec,c)
# # =============================================================================
#     return crds, rec
# 
# =============================================================================
