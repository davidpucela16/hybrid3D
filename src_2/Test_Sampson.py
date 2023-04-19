#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 18:43:43 2023

@author: pdavid
"""



#%%
import scipy as sp 
def append_sparse(arr, d, r, c):
    data_arr=np.append(arr[0], d)
    row_arr=np.append(arr[1], r)
    col_arr=np.append(arr[2], c)
    return(data_arr, row_arr, col_arr)
def assemble_transport_1D(U, D, h, cells):
    """Assembles the linear matrix for convection-dispersion for a single vessel with Dirichlet
    boundary conditions on both sides"""
    
    sparse_arrs=np.array([]), np.array( []), np.array([])
    
    for i in np.arange(cells-2)+1:
        sparse_arrs =append_sparse(sparse_arrs, np.array([-U-D/h,U+ 2*D/h, -D/h]), np.array([i,i,i]), np.array([i-1,i,i+1]))
    
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
#%%
from Green import get_grad_source_potential, log_line, grad_point, Simpson_surface, get_source_potential
arg=a,b,0.075,math.inf
t=Simpson_surface(arg, get_source_potential, np.array([0,1,0]), 1,np.array([0,1,0]), 1)




#%%
a=np.array([0,0,0])
b=np.array([0,0,0.1])

pot=np.zeros(100)
dip=np.zeros(100)
normal=np.array([0,1,0])
c=0
for i in np.linspace(0.1,4,100):
    pos=np.array([0,i,0.5])
    tup_args=a,b,pos,0.075,math.inf
# =============================================================================
#     pot[c]=get_source_potential(tup_args, 1)[0]
#     dip[c]=get_grad_source_potential(tup_args,1)[0].dot(normal)
# =============================================================================
    
    pot[c]=log_line((pos, a, b))
    dip[c]=grad_point((pos,(a+b)/2)).dot(normal)
    
    c+=1

plt.plot(np.linspace(0.1,4,100),-dip)
plt.plot(np.linspace(0.1,4,100),pot)