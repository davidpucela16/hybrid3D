#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:56:31 2023

@author: pdavid


This file is meant to test the intravascular numerical model for networks. In short, this test was 
written after the validation of the intravascular model for 1 vessel. The goal here is to make sure 
all the mess about the assembly of the bifurcations is properly done.
"""
import numpy as np 
import scipy as sp 
import pdb

from scipy.sparse import csc_matrix

from scipy.sparse.linalg import spsolve as dir_solve

import os 
path=os.path.dirname(__file__)
path_src=os.path.join(path, '../')
os.chdir(path_src)

from assembly_1D import assemble_transport_1D, assemble_vertices
import matplotlib.pyplot as plt
#%% - Validation single vessel with simplest bifurcation:
D=1
U=np.empty(2)
U[0]=0.5
R=np.array([1,1])

L=np.array([10,10])

U[1]=(U[0]*R[0]**2)/R[1]**2

vertex_to_edge=[[0],[0,1],[1]]

cells=np.array([100,100])

h=L[0]/cells[0]+np.zeros(len(cells))

init=np.array([0,1])

sparse_arrs=assemble_transport_1D(U, D, h, cells)
a, III_ind_array,kk=assemble_vertices(U, D, h, cells, sparse_arrs, vertex_to_edge,R, init, np.array([[0,1],[2,0]]))
# =============================================================================
# III_ind_array[0]*=1
# III_ind_array[-1]*=0
# III_ind_array[np.sum(cells[:2])-1]*=0
# =============================================================================
I=csc_matrix((a[0], (a[1], a[2])), shape=(np.sum(cells), np.sum(cells)))

sol=dir_solve(I, -III_ind_array)

plt.plot(sol)
#%%

s = np.linspace(np.sum(L)/2/np.sum(cells), np.sum(L)-np.sum(L)/2/np.sum(cells), np.sum(cells))

Pe = U[0]*np.sum(L)/D

analytical = (np.exp(Pe*s/np.sum(L))-np.exp(Pe))/(1-np.exp(Pe))

plt.plot(s, analytical, label="analytical")
plt.plot(s, sol, label='numerical')
plt.show()


#%% - Now for a small network 
#%- Validation
D=1
U=np.empty(3)
U[0]=2
R=np.array([1,1,1])

L=np.array([10,10,10])

U[2]=(U[0]*R[0]**2)/R[2]**2/2
U[1]=U[2]

Pe=U*L[0]/D

vertex_to_edge=[[0],[0,1,2],[1],[2]]

cells=np.array([100,100,100])

h=L[0]/cells[0]+np.zeros(len(cells))

init=np.array([0,1,1])

BCs_1D=np.array([[0,1],
                 [2,0],
                 [3,0]])

sparse_arrs=assemble_transport_1D(U, D, h, cells)
a, III_ind_array,kk=assemble_vertices(U, D, h, cells, sparse_arrs, vertex_to_edge,R, init, BCs_1D)

I=csc_matrix((a[0], (a[1], a[2])), shape=(np.sum(cells), np.sum(cells)))

sol=dir_solve(I, -III_ind_array)

plt.plot(sol)
plt.show()
#%%
A,B,C,d=np.linalg.solve(np.array([[1,1,0,0],
                                  [np.exp(Pe[0]),1,-1,-1],
                                  [0,0,np.exp(Pe[1]), 1],
                                  [np.exp(Pe[0]), 0, -1, 0]]), np.array([1,0,0,0]))
x=np.linspace(0,L[0]*(1-1/cells[0]), cells[0])+L[0]/cells[0]/2

analyt=np.concatenate((A*np.exp(Pe[0]*x/L[0])+B, C*np.exp(Pe[1]*x/L[1])+d,C*np.exp(Pe[1]*x/L[0])+d))
plt.plot(analyt, label='analyt')
plt.plot(sol, label='sol')
plt.legend()
plt.show()

#%% - Now with simplest bifurcation:
from assembly_1D import full_adv_diff_1D

a, III_ind_array,kk=full_adv_diff_1D(U, D, h, cells, init, vertex_to_edge, R, BCs_1D)

I=csc_matrix((a[0], (a[1], a[2])), shape=(np.sum(cells), np.sum(cells)))

sol=dir_solve(I, -III_ind_array)

plt.plot(sol)
plt.show()