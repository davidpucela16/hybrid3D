#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 18:47:23 2023

@author: pdavid
"""

import numpy as np
from assembly import assemble_transport_1D
import matplotlib.pyplot as plt
import scipy as sp
import scipy.sparse.linalg
import pdb

#%% - This is the validation of the 1D transport eq without reaction
U = 1
D = 1
L = 10
cells = 50

x = np.linspace(L/2/cells, L-L/2/cells, cells)

Pe = U*L/D

analytical = (np.exp(Pe*x/L)-np.exp(Pe))/(1-np.exp(Pe))

plt.plot(x, analytical)


a, b = assemble_transport_1D(U, D, L/cells, cells)

A = sp.sparse.csc_matrix((a[0], (a[1], a[2])), shape=(cells, cells))

B = np.zeros(cells)
B[0] = -b[0]

sol = sp.sparse.linalg.spsolve(A, B)

#sol = np.hstack((np.array((1)), sol, np.array((0))))

plt.plot(x,sol, label='numerical')
plt.plot(x, analytical, label='analytical')
plt.legend()

# plt.plot(x,sol)
