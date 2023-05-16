#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 09:22:23 2023

@author: pdavid
"""
import os 
path=os.path.dirname(__file__)
os.chdir(path)
import numpy as np 
import pdb 
import matplotlib.pyplot as plt
from neighbourhood import GetNeighbourhood, GetUncommon
from small_functions import TrilinearInterpolation, auto_TrilinearInterpolation
from small_functions import FromBoundaryGetNormal, AppendSparse, GetBoundaryStatus
from scipy.sparse.linalg import spsolve as dir_solve
from assembly import AssemblyDiffusion3DBoundaries, AssemblyDiffusion3DInterior
from Second_eq_functions import node, InterpolateFast,GetInterpolationKernelFast,GetI1Fast, InterpolatePhiBarBlock

import scipy as sp
from scipy.sparse import csc_matrix

import multiprocessing
from multiprocessing import Pool
from assembly_1D import FullAdvectionDiffusion1D

from numba.typed import List

from mesh import GetID, Get8Closest
from mesh_1D import KernelIntegralSurfaceFast, KernelPointFast
from small_functions import in1D
from GreenFast import SimpsonSurface

from numba import njit, prange
from numba.experimental import jitclass
from numba import int64, float64
from numba import int64, types, typed
import numba as nb
import matplotlib.pylab as pylab

import dask
from dask import delayed

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

import time

@dask.delayed
def InterpolateHelper(prob, coords):
    a,b,c,d = prob.Interpolate(coords)
    return a.dot(prob.s[b]) + c.dot(prob.q[d]) 

class Visualization3D():
    def __init__(self,lim, res, prob, num_proc, vmax, *trans):
        
        self.vmax=vmax
        self.vmin=0
        self.lim=lim
        self.res=res
        self.prob=prob
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
                
                a,b=prob.GetCoordReconst_chat(cor[j], res, num_processes=num_proc)
                
                data[i*3+j]=b.reshape(res, res)
                
        self.data=data
        
        self.perp_x=perp_x
        self.perp_y=perp_y
        self.perp_z=perp_z
        
        self.plot(data, lim)
        
        return
    
    def getCoordReconstChat(self, corners, resolution, num_processes=4):
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
#         # Use map function to apply Interpolate_helper to each coordinate in parallel
#         results = pool.map(Interpolate_helper, [(self, k) for k in crds])
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
            rec=np.append(rec, InterpolateHelper((self.prob, k)))
        
        return crds, rec     
    
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

def get_coord_reconst_chat(prob, corners, resolution, num_processes=4):
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
    #         results = pool.map(interpolate_helper, [(prob, k) for k in crds])
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
        rec=np.append(rec, InterpolateHelper((prob, k)))
    
    return crds, rec     