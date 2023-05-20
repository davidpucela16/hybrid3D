#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 10:08:11 2023

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


import os
path=os.path.dirname(__file__)
os.chdir(os.path.join(path, "../../Kleinfeld_1"))
from Load_Kleinfeld import get_problem
os.chdir('../src_full_opt')
from mesh_1D import KernelPointFast
from neighbourhood import GetNeighbourhood, GetUncommon
from mesh import GetID, Get8Closest
from hybridFast import GetBoundaryStatus, PhiBarHelper
from numba import types, typed
from numba.typed import List
from small_functions import TrilinearInterpolation
import pdb
from numba import int64, float64
from numba.experimental import jitclass
from multiprocessing import Pool
import time
import dask
prob=get_problem(20,1)
mat_path='/home/pdavid/Bureau/Code/hybrid3d/Kleinfeld_1/matrices'

#%%
t=np.zeros(10)
t[0]=time.time()
prob.AssemblyABC()
t[1]=time.time()


# Load the saved sparse matrix from the file
sp.sparse.save_npz(mat_path +"/B_matrix_cells{}_n{}.npz".format(prob.mesh_3D.cells_x,prob.n), prob.B_matrix.tocsr())

#%%
t[2]=time.time()
prob.AssemblyDEFFast(mat_path)
t[3]=time.time()

#%%


kernel_q=np.zeros(0, dtype=np.float64)
kernel_s=np.zeros(0, dtype=np.float64)
#The kernels with the positions
col_s=np.zeros(0, dtype=np.int64)
col_q=np.zeros(0, dtype=np.int64)

row_s=np.zeros(0, dtype=np.int64)
row_q=np.zeros(0, dtype=np.int64)



def RetrievePhiBar(mat_path, S, size_mesh):
    c=0
    for i in prob.mesh_1D.uni_s_blocks:
        print("block: ", i)
        kernel_s, kernel_q, col_s, col_q, row_s, row_q=retrieve_block_phi_bar(mat_path,i)
        c+=1
        phi_bar_s=sp.sparse.csc_matrix((S, size_mesh))
        phi_bar_q=sp.sparse.csc_matrix((S, S))
        if c%200==0:
            phi_bar_s+=sp.sparse.csc_matrix((kernel_s,(row_s, col_s)), shape=(prob.S, prob.mesh_3D.size_mesh))
            phi_bar_q+=sp.sparse.csc_matrix((kernel_q,(row_q, col_q)), shape=(prob.S, prob.S))
            kernel_q=np.zeros(0, dtype=np.float64)
            kernel_s=np.zeros(0, dtype=np.float64)
            #The kernels with the positions
            col_s=np.zeros(0, dtype=np.int64)
            col_q=np.zeros(0, dtype=np.int64)
    
            row_s=np.zeros(0, dtype=np.int64)
            row_q=np.zeros(0, dtype=np.int64)
   
def retrieve_block_phi_bar(path, block):
    kernel_s=np.load(path + '/{}_kernel_s.npy'.format(block))
    kernel_q=np.load(path + '/{}_kernel_q.npy'.format(block))
    col_s=np.load(path + '/{}_col_s.npy'.format(block))
    col_q=np.load(path + '/{}_col_q.npy'.format(block))
    row_s=np.load(path + '/{}_row_s.npy'.format(block))
    row_q=np.load(path + '/{}_row_q.npy'.format(block))
    return kernel_s, kernel_q, col_s, col_q, row_s, row_q
     
#%%

phi_bar_kernel=sp.sparse.hstack((phi_bar_s, phi_bar_q))

#%%
import pstats
import cProfile
if __name__ == '__main__':
    # run the profiler on the code and save results to a file
    cProfile.run('InterpolatePhiBarFast(prob.n, prob.mesh_3D.cells_x, prob.mesh_3D.cells_y, prob.mesh_3D.cells_z, prob.mesh_3D.h,prob.mesh_3D.pos_cells,prob.mesh_1D.s_blocks, prob.mesh_1D.source_edge,prob.mesh_1D.tau, prob.mesh_1D.pos_s, prob.mesh_1D.h, prob.R, prob.D)', filename=path + '/profile_results.prof')


stats = pstats.Stats(path + '/profile_results.prof')
stats.strip_dirs().sort_stats('cumulative').print_stats(100)


#%%import pstats
if __name__ == '__main__':
    # run the profiler on the code and save results to a file
    cProfile.run('InterpolatePhiBarFast_3(prob.n, prob.mesh_3D.cells_x, prob.mesh_3D.cells_y, prob.mesh_3D.cells_z, prob.mesh_3D.h,prob.mesh_3D.pos_cells,prob.mesh_1D.s_blocks, prob.mesh_1D.source_edge,prob.mesh_1D.tau, prob.mesh_1D.pos_s, prob.mesh_1D.h, prob.R, prob.D, prob.max_size)', filename=path + '/profile_results.prof')


stats = pstats.Stats(path + '/profile_results.prof')
stats.strip_dirs().sort_stats('cumulative').print_stats(100)



#%%

if __name__ == '__main__':
    # run the profiler on the code and save results to a file
    cProfile.run('InterpolatePhiBarBlock(6742,prob.n, prob.mesh_3D.cells_x, prob.mesh_3D.cells_y, prob.mesh_3D.cells_z, prob.mesh_3D.h, prob.mesh_3D.pos_cells,prob.mesh_1D.s_blocks, prob.mesh_1D.source_edge,prob.mesh_1D.tau, prob.mesh_1D.pos_s, prob.mesh_1D.h, prob.mesh_1D.R, 1, prob.mesh_1D.sources_per_block, prob.mesh_1D.quant_sources_per_block)', filename=path + '/profile_results.prof')


stats = pstats.Stats(path + '/profile_results.prof')
stats.strip_dirs().sort_stats('cumulative').print_stats(100)


#%%

InterpolatePhiBarFast(prob.n, prob.mesh_3D.cells_x, prob.mesh_3D.cells_y, prob.mesh_3D.cells_z, prob.mesh_3D.h,
                                                                             prob.mesh_3D.pos_cells,prob.mesh_1D.s_blocks, prob.mesh_1D.source_edge,prob.mesh_1D.tau, prob.mesh_1D.pos_s, prob.mesh_1D.h, 
                                                                             prob.R, prob.D)
