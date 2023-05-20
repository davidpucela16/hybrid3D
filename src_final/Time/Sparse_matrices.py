#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 10:37:35 2023

@author: pdavid
"""

import numpy as np
from numba import njit
from scipy.sparse import coo_matrix


def create_sparse_matrix(n):
    a=np.array([0], dtype=np.float64)
    for i in np.arange(n)+1:
        b=np.arange(i, dtype=np.float64)
        a=np.concatenate((a,b))
    return a

@njit
def kk(matrix):
    for i in matrix:
        b=np.array(i, dtype=np.float64)
    



