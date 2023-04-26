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
from small_functions import trilinear_interpolation
from Green import get_grad_source_potential, get_source_potential
from small_functions import for_boundary_get_normal, append_sparse
from assembly import assemble_transport_1D

from assembly import Assembly_diffusion_3D_boundaries, Assembly_diffusion_3D_interior

import scipy as sp
from scipy.sparse import csc_matrix

import multiprocessing
from multiprocessing import Pool

print("NoLoc")

class hybrid_set_up():
    def __init__(self, mesh_3D, mesh_1D,  BC_type, BC_value, D, K):
        
        self.K=K
        
        self.mesh_3D=mesh_3D
        self.h=mesh_3D.h
        
        self.mesh_1D=mesh_1D
        self.D=D
        self.BC_type=BC_type
        self.BC_value=BC_value
        
        
        self.mesh_3D.get_ordered_connect_matrix()
        
        self.F=self.mesh_3D.size_mesh
        self.S=len(mesh_1D.pos_s)
        
        return
    

    
    def Assembly_problem(self):
        
        A_matrix=self.Assembly_A()
        self.A_matrix=A_matrix
        
        B_matrix=self.Assembly_B_boundaries()
        self.B_matrix=B_matrix
        
        C_matrix=csc_matrix((self.mesh_3D.size_mesh, len(self.mesh_1D.pos_s)))
        
        
        D_E_F_matrix=self.Assembly_D_E_F()
        I_matrix=self.Assembly_I()
        
        #WILL CHANGE WHEN CONSIDERING MORE THAN 1 VESSEL
        H_matrix=sp.sparse.identity(len(self.mesh_1D.pos_s))*self.mesh_1D.h[0]/(np.pi*self.mesh_1D.R[0]**2)
        self.H_matrix=H_matrix
        
        #Matrix full of zeros:
        G_matrix=csc_matrix(( len(self.mesh_1D.pos_s),self.mesh_3D.size_mesh))
        self.G_matrix=G_matrix
        
        Full_linear_matrix=sp.sparse.hstack((A_matrix, B_matrix, C_matrix))
        Full_linear_matrix=sp.sparse.vstack((Full_linear_matrix, D_E_F_matrix))
        self.G_H_I_matrix=sp.sparse.hstack((G_matrix, H_matrix, I_matrix))
        
        Full_linear_matrix=sp.sparse.vstack((Full_linear_matrix, self.G_H_I_matrix))
        
        self.Full_linear_matrix=Full_linear_matrix
        
        self.Full_ind_array=np.concatenate((self.I_ind_array, np.zeros(len(self.mesh_1D.pos_s)), self.III_ind_array))
        
        return
    
    def Assembly_A(self):
        """An h is missing somewhere to be consistent"""
        size=self.mesh_3D.size_mesh
        
        a=Assembly_diffusion_3D_interior(self.mesh_3D)
        b=Assembly_diffusion_3D_boundaries(self.mesh_3D, self.BC_type, self.BC_value)
        
        # =============================================================================
        #         NOTICE HERE, THIS IS WHERE WE MULTIPLY BY h so to make it dimensionally consistent relative to the FV integration
        A_matrix=csc_matrix((a[2]*self.mesh_3D.h, (a[0], a[1])), shape=(size,size)) + csc_matrix((b[2], (b[0], b[1])), shape=(size, size))
        #       We only multiply the non boundary part of the matrix by h because in the boundaries assembly we need to include the h due to the difference
        #       between the Neumann and Dirichlet boundary conditions. In short Assembly_diffusion_3D_interior returns data that needs to be multiplied by h 
        #       while Assembly_diffusion_3D_boundaries is already dimensionally consistent
        # We do not multiply the b side i.e. the boundary part of the matrix by h cause it is multiplied inside the function already, 
        # that is, the kernel already has the dimensional values!
        # =============================================================================
        self.I_ind_array=b[3]
        
        
        return A_matrix
    

            
    def Assembly_B_boundaries(self):

        B=sp.sparse.lil_matrix((self.mesh_3D.size_mesh,len(self.mesh_1D.pos_s)))
        
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
                
                pos_k=mesh.get_coords(k)
                normal=normals[c]
                pos_boundary=pos_k+normal*h/2
                if self.BC_type[c]=="Dirichlet":
                    r_k=self.mesh_1D.kernel_integral_surface(pos_boundary, normal,  self.mesh_1D.uni_s_blocks, get_source_potential, self.K,self.D, mesh.h)
                    kernel=csc_matrix((r_k[0]*2*h,(np.zeros(len(r_k[0])),r_k[1])), shape=(1,len(self.mesh_1D.s_blocks)))
                if self.BC_type[c]=="Neumann":
                    r_k=self.mesh_1D.kernel_integral_surface(pos_boundary, normal,  self.mesh_1D.uni_s_blocks, get_grad_source_potential, self.K,self.D, mesh.h)
                    if np.any(r_k[0]>0): pdb.set_trace()
                    kernel=csc_matrix((r_k[0]*h**2,(np.zeros(len(r_k[0])),r_k[1])), shape=(1,len(self.mesh_1D.s_blocks)))
                B[k,:]-=kernel
                #if np.any(kernel.toarray()<0): pdb.set_trace()
            c+=1
            
        return(B)
        
     
        
    def Assembly_D_E_F(self):
        
        D=np.zeros([3,0])
        E=np.zeros([3,0])
        F=np.zeros([3,0])
        
   
        
        for j in range(len(self.mesh_1D.s_blocks)):
            kernel_s,col_s,kernel_q, col_q,kernel_C_v,  col_C_v=self.interpolate(self.mesh_1D.pos_s[j])
            D=append_sparse(D, kernel_s,np.zeros(len(col_s))+j, col_s)
            
            E=append_sparse(E, kernel_q,np.zeros(len(col_q))+j, col_q)
            
            if len(kernel_C_v)!=len(col_C_v): pdb.set_trace()
            
            F=append_sparse(F, kernel_C_v,np.zeros(len(col_C_v))+j, col_C_v)
            
            E=append_sparse(E, 1/self.K[self.mesh_1D.source_edge[j]] , j, j)
        F=append_sparse(F, -np.ones(len(self.mesh_1D.s_blocks)) , np.arange(len(self.mesh_1D.s_blocks)), np.arange(len(self.mesh_1D.s_blocks)))
            
        self.D_matrix=csc_matrix((D[0], (D[1], D[2])), shape=(len(self.mesh_1D.pos_s), self.mesh_3D.size_mesh))
        self.E_matrix=csc_matrix((E[0], (E[1], E[2])), shape=(len(self.mesh_1D.pos_s), len(self.mesh_1D.pos_s)))
        self.F_matrix=csc_matrix((F[0], (F[1], F[2])), shape=(len(self.mesh_1D.pos_s), len(self.mesh_1D.pos_s)))
        
        return sp.sparse.hstack((self.D_matrix, self.E_matrix, self.F_matrix))
    
    def Assembly_I(self):
        """Models intravascular transport. Advection-diffusion equation
        
        FOR NOW IT ONLY HANDLES A SINGLE VESSEL"""
        
        D=self.mesh_1D.D
        U=self.mesh_1D.U
        L=self.mesh_1D.L
        
        
        aa, bb =assemble_transport_1D(U, D, L[0]/len(self.mesh_1D.pos_s), len(self.mesh_1D.pos_s))
        
        I = csc_matrix((aa[0], (aa[1], aa[2])), shape=(len(self.mesh_1D.pos_s), len(self.mesh_1D.pos_s)))

        self.III_ind_array = np.zeros(len(self.mesh_1D.pos_s))
        self.III_ind_array [0] = bb[0]
        
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
        
        # Create a pool of worker processes
        pool = Pool(processes=num_processes)
        # Use map function to apply interpolate_helper to each coordinate in parallel
        results = pool.map(interpolate_helper, [(self, k) for k in crds])
        
        # Close the pool to free up resources
        pool.close()
        pool.join()
# =============================================================================
#         c=0
#         for k in crds:
#             interpolate_helper((self, k))
#             c+=1
# =============================================================================

        # Convert the results to a numpy array
        rec = np.array(results)
        
        return crds, rec     
    
# =============================================================================
#     def get_coord_reconst_center(self, corners):
#         """Corners given in order (0,0),(0,1),(1,0),(1,1)
#         
#         Returns the coarse reconstruction, cell center"""
#         
#         tau_1=corners[1]-corners[0]
#         tau_1=tau_1/np.linalg.norm(tau_1)
#         tau_2=corners[2]-corners[0]
#         tau_2=tau_2/np.linalg.norm(tau_2)
#         
#         init=self.mesh_3D.get_id(corners[0]) #ID of the lowest block
#         init_coords=self.mesh_3D.get_coords(init)
#         crds=init_coords.copy()
#         
#         end_1=self.mesh_3D.get_id(corners[1]) #ID of the upper corner
#         end_1_coords=self.mesh_3D.get_coords(end_1)
#         end_2=self.mesh_3D.get_id(corners[2]) #ID of the upper corner
#         end_2_coords=self.mesh_3D.get_coords(end_2)
#         
#         L1=end_1_coords-init
#         L2=end_2_coords-init
#         v=np.array([])
#         for i in range(int(end_2-init)):
#             for j in range(int(end_1-init)):
#                 cr=init_coords+self.mesh_3D.h*i*tau_2+self.mesh_3D.h*j*tau_1
#                 
#                 crds=np.vstack((crds, cr))
#                 
#                 k=self.mesh_3D.get_id(cr) #Current block
#                 
#                 if np.any(self.mesh_3D.get_coords(k)-cr): print("ERORROOROROROROR")
#                 
#                 a,b,c,d,_,_=self.interpolate(cr)
#                 
#                 value=0
#                 print(a)
#                 value+=np.dot(self.q[d], c)
#                 v=np.append(v, value)
#         return crds, v  
# =============================================================================
    
   
    def interpolate(self, x):
        """returns the kernels to obtain an interpolation on the point x. 
        In total it will be 6 kernels, 3 for columns and 3 with data for s, q, and
        C_v respectively"""
        bound_status=self.get_bound_status(x)
        if len(bound_status): #boundary interpolation
        
            blocks=np.array([self.mesh_3D.get_id(x)], dtype=int)
            kernel_s=np.array([1])
            
        else: #no boundary node (standard interpolation)
            blocks=self.mesh_3D.get_8_closest(x)
            kernel_s=trilinear_interpolation(x, np.ones(3)*self.mesh_3D.h)
        
        q_array, C_v_array, sources=self.mesh_1D.kernel_point(x, self.mesh_1D.uni_s_blocks, get_source_potential, self.K, self.D)
            
        #if np.all(x==np.array([1.5, 4.5, 2.5])): pdb.set_trace()
        return kernel_s, blocks, q_array, sources, C_v_array, sources
    
    
  
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
    
    
    def get_point_value_post(self, coords, rec,k):
        a,b,c,d,e,f=self.interpolate(coords)
        rec[k]=a.dot(self.s[b])+c.dot(self.q[d])
        return 
    
     

def interpolate_helper(args):
    self, k = args
    a,b,c,d,e,f = self.interpolate(k)
    return a.dot(self.s[b]) + c.dot(self.q[d])        

class visualization_3D():
    def __init__(self,lim, res, prob, num_proc, vmax):
        
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
        data=np.empty([9, res, res])
        for i in range(3):
            for j in range(3):
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
