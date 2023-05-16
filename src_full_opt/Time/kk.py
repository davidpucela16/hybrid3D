#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 18:12:14 2023

@author: pdavid
"""


import dask
from dask import delayed
import numpy as np 
from time import sleep 
from numba import jit, types

@jit(nopython=True)
def numba_jitted_function(x):
    a = np.float64(0)
    for i in x:
        a += i
    return a

# Wrap the Numba-jitted function with dask.delayed
delayed_function = delayed(numba_jitted_function)

# Create delayed objects for the function calls
delayed_result1 = delayed_function(np.random.random((10000)).astype(np.float64))
delayed_result2 = delayed_function(np.random.random((10000)).astype(np.float64))

# Compute the results using Dask
results = dask.compute(delayed_result1, delayed_result2)



#%%

data = [1, 2, 3, 4, 5, 6, 7, 8]

def PhiBarHelper(args):
    block, lst=args
    path,n, cells_x, cells_y, cells_z, h_3D,pos_cells,s_blocks, source_edge,tau, pos_s, h_1D, R, D,sources_per_block, quant_sources_per_block=lst