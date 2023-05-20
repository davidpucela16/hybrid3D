#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 09:41:23 2023

@author: pdavid
"""

import numpy as np
from numba import njit

import pdb
from scipy.integrate import quad, dblquad

from GreenFast import SimpsonSurface, GetGradSourcePotential, GetSourcePotential

# =============================================================================
# @njit
# def GetSourcePotential(x, x_star, R, h, D):
#     # Convert input arrays to float arrays
#     x = x.astype(np.float64)
#     x_star = x_star.astype(np.float64)
#     
#     return h*R**2/4/(4*np.pi*D*np.linalg.norm(x-x_star))
# =============================================================================
@njit
def LogLine(x,a,b):
    """Returns the average value of the integral without the coefficient i.e. to the result one would have to multiply
    by the surface of the open cylinder (2 \pi R_j L_j)/(4*pi*D) to obtain a proper single layer potential
    
    DONT FORGET THE DIFUSSION COEFFICIENT"""
    ra=np.linalg.norm(x-a)
    rb=np.linalg.norm(x-b)
    
    L=np.linalg.norm(b-a)
    tau=(b-a)/L
    log=np.log((np.max(np.array([ra, rb])) + L/2 + np.abs(np.dot((a+b)/2-x,tau)))/(np.min(np.array([ra, rb])) - L/2 + np.abs(np.dot((a+b)/2-x,tau))))
    
    return np.float64(log)
@njit
def Gjerde(x,a,b,R):
    """Returns the average value of the integral without the coefficient i.e. to the result one would have to multiply
    by the surface of the open cylinder (2 \pi R_j L_j)/(4*pi*D) to obtain a proper single layer potential
    
    DONT FORGET THE DIFUSSION COEFFICIENT"""
    ra=np.linalg.norm(x-a)
    rb=np.linalg.norm(x-b)
    
    L=np.linalg.norm(b-a)
    tau=(b-a)/L
    
    log=np.log((np.max([ra, rb]) + L/2 + np.abs(np.dot((a+b)/2-x,tau)))/(np.min([ra, rb]) - L/2 + np.abs(np.dot((a+b)/2-x,tau))))
    #log=np.log((rb+L+np.dot(tau, a-x))/(ra+np.dot(tau, a-x)))
    return log*R/2



@njit
def line_source(inc_x, tau_1, tau_2, h, center):
    c=center+tau_2*inc_x
    a=c+tau_1*h/2
    b=c-tau_1*h/2
    ra=np.linalg.norm(a)
    rb=np.linalg.norm(b)
    
    return np.log((np.max(np.array([ra, rb])) + h/2 + np.abs(np.dot((a+b)/2,tau_1)))/(np.min(np.array([ra, rb])) - h/2 + np.abs(np.dot((a+b)/2,tau_1))))/4/np.pi


def Integral_potential_interface_square( x_source, center_square,normal_square, h_square):
    
    tau_1, tau_2=np.zeros(3), np.zeros(3)
    tau_1[np.where(normal_square==0)[0][0]]=1
    tau_2[np.where(normal_square==0)[0][1]]=1
    
    f = lambda inc_x: line_source(inc_x, tau_1, tau_2, h_square, center_square-x_source)
    integral, error = quad(f, -h_square/2, h_square/2)
    return integral


    
    
@njit
def Integral_grad_potential_interface_square_prev(x_source, center_square,normal_square, h_square):
    """This formula is not always valid"""
    d=np.linalg.norm(x_source-center_square)
    h_prime=np.dot(center_square-x_source, normal_square)*h_square/d
    theta=np.arctan(h_prime/2/d)
    
    alpha=h_prime/2/d
    
    return -np.arcsin(alpha**2/np.sqrt(2*(1+alpha**2)))/np.pi

@njit
def Integral_grad_potential_interface_square_prev(x_source, center_square,normal_square, h_square):
    d=np.linalg.norm(x_source-center_square)
    h_prime=np.dot(center_square-x_source, normal_square)*h_square/d
    theta=np.arctan(h_prime/2/d)
    
    alpha=h_prime/2/d
    
    return -np.arcsin(alpha**2/np.sqrt(2*(1+alpha**2)))/np.pi





#%%

def unit_test_Simpson(h,D):
    
    arg=np.array([0,-0.2,0]),np.array([0,0.2,0]),0.1,D
    a=SimpsonSurface(arg, GetGradSourcePotential, np.array([0,0,-h/2]),h, np.array([0,0,-1]), D)*h**2
    
    b=GradPoint((np.array([0,0,0]), np.array([0,h/2,0])))[1]*h**2*16/36
    
    b+=GradPoint((np.array([0,0,0]), np.array([-h/2,h/2,0])))[1]*h**2*4/36
    b+=GradPoint((np.array([0,0,0]), np.array([0,h/2,-h/2])))[1]*h**2*4/36
    b+=GradPoint((np.array([0,0,0]), np.array([0,h/2,h/2])))[1]*h**2*4/36
    b+=GradPoint((np.array([0,0,0]), np.array([-h/2,h/2,0])))[1]*h**2*4/36
    
    b+=GradPoint((np.array([0,0,0]), np.array([-h/2,h/2,-h/2])))[1]*h**2/36
    b+=GradPoint((np.array([0,0,0]), np.array([-h/2,h/2,h/2])))[1]*h**2/36
    b+=GradPoint((np.array([0,0,0]), np.array([h/2,h/2,-h/2])))[1]*h**2/36
    b+=GradPoint((np.array([0,0,0]), np.array([h/2,h/2,h/2])))[1]*h**2/36
    
    from scipy.integrate import dblquad
    def integrand(y, x):
        
        p_1=np.array([0,0,0])
        p_2=np.array([x,y,h/2])
        r = np.dot(GradPoint((p_2, p_1)), np.array([0,0,1]))
        return r / (4*np.pi*D)
    
    scp, _ = dblquad(integrand, -h/2,h/2 , -h/2,  h/2)
    #The following is the exact result of the Simpson's integration done by hand (notebook)
# =============================================================================
#     print("Solid angle method", Integral_grad_potential_interface_square(np.array([0,0,0], dtype=float),
#                                                                          np.array([h/2,0,0], dtype=float),
#                                                                          np.array([1,0,0], dtype=float), 
#                                                                          h))
# =============================================================================
    print("With scipy integration: ", scp)
    print("Manually with the grad point function", -b/4/np.pi/D)
    print("Analytical Simpson= ", -(27**-0.5+2**0.5+4)/(9*np.pi*D))
    #The following is the value returned by the function 
    print("Calculated Simpson= ", a/np.linalg.norm(arg[0]-arg[1])) #We have to divide by the length of the source since it is included in the integral
    #The following is the analytical value of the integral
    print("Analytical (Gauss)= ", -1/(6*D))
    return

def another_unit_test_Simpson():
    """integral over the faces of a cube to make sure the gradient of the greens function
    over a closed surface is 0"""
    normal=np.array([[0,0,1],
                     [0,0,-1],
                     [0,1,0],
                     [0,-1,0],
                     [1,0,0],
                     [-1,0,0]])
    integral=0
    for h in np.array([1,2,3,9]): #size of the cube
        #pdb.set_trace()    
        arg=np.array([0,-0.2,0]),np.array([0,0.2,0]),0.1,1
        L=0.4
        integral=0
        for i in range(6):
            no=normal[i]
            center=no*h/2
            integral+=SimpsonSurface(arg, GetGradSourcePotential, center, h, no, 1)*h**2/L
        print("This must be 1 due to properties of delta", integral)
    print("We expect around a 20% error")
        
        

def unit_test_single_layer(h, a):
    """h is the size of the square
    a is the separation from the square of the point where the source is located"""
    #Here we want to test the integration of the potential function
    from scipy.integrate import dblquad
    import time
    
    def integrand(y, x):
        r = np.sqrt(x**2 + y**2 + a**2)
        return 1 / (4*np.pi*r)
    
    t=np.zeros(5)
    
    t[0]=time.time()
    integral, _ = dblquad(integrand, -h/2,h/2 , -h/2,  h/2)
    t[1]=time.time()
    
    arg=np.array([-0.1,0,0]),np.array([0.1,0,0]),1,1
    
    t[2]=time.time()
    mine=SimpsonSurface(arg, GetSourcePotential, np.array([0,0,a]), h, np.array([0,0,1]), 1)*h**2/np.linalg.norm(arg[1]-arg[0])
    t[3]=time.time()
    fast=Integral_potential_interface_square(np.array([0,0,0], dtype=float),
                                            np.array([a,0,0], dtype=float),
                                            np.array([1,0,0], dtype=float), 
                                            h)*h**2
    t[4]=time.time()
    
    
    print("Scipy integral: ", integral)
    print("Simpson integral", mine)
    print("fast", fast)
    return t[1]-t[0], t[3]-t[2], t[4]-t[3]

#%%
import cProfile

def my_function():
    # Some code to be profiled
    ...

cProfile.runctx('result = my_function', globals(), locals())

# Print the profiling results
import pstats
p = pstats.Stats()
p.strip_dirs().sort_stats('cumulative').print_stats(10)

#%% - Validation gradient

@njit
def GradPoint(x, x_j):
    return -(x-x_j)/(np.linalg.norm(x-x_j)**3)
def unit_test_gradient_any_point(x, center_square, normal_square, h_square,D):
    
    tau_1, tau_2=np.zeros(3), np.zeros(3)
    tau_1[np.where(normal_square==0)[0][0]]=1
    tau_2[np.where(normal_square==0)[0][1]]=1
    
    def integrand(y, x):
        p_1=np.array([0,0,0])
        p_2=center_square + x*tau_1 + y*tau_2
        r = np.dot(GradPoint(p_2, p_1), normal_square)
        return r / (4*np.pi*D)
    scp, _ = dblquad(integrand, -h_square/2,h_square/2 , -h_square/2,  h_square/2)
    #The following is the exact result of the Simpson's integration done by hand (notebook)
    print("With scipy integration: ", scp)
    
    print("Mid point rule", np.dot(GradPoint(center_square, x), normal_square)*h_square**2)
    return



#%%
import numba
@njit
def try_code():
    b=[]
    for i in range(10**5):
        b.append(LogLine(np.array([0.1+i,0,0], dtype=np.float64),np.array([0,0,0], dtype=np.float64),np.array([0,0.1,0], dtype=np.float64)))
    return b
        
        #%%
@njit
def my_func():
    my_list = []
    for i in range(10):
        my_list.append(LogLine(np.array([0.1+i,0,0], dtype=np.float64),np.array([0,0,0], dtype=np.float64),np.array([0,0.1,0], dtype=np.float64)))
    return np.array(my_list)



#%%

def GetNeighbourhood(n, cells_x, cells_y, cells_z, block_ID):
    """Will return the neighbourhood of a given block for a given n
    in a mesh made of square cells

    It will assume cells_x=ylen"""
    
    step_y=cells_z
    step_x=cells_z*cells_y
    
    pad_x = np.concatenate((np.zeros((n)) - 1, np.arange(cells_x), np.zeros((n)) - 1)).astype(int)
    pad_y = np.concatenate((np.zeros((n)) - 1, np.arange(cells_y), np.zeros((n)) - 1)).astype(int)
    pad_z = np.concatenate((np.zeros((n)) - 1, np.arange(cells_z), np.zeros((n)) - 1)).astype(int)
    pos_x, pos_y, pos_z =int( block_ID/(cells_y*cells_z)), int(int(block_ID%step_x)/cells_z), int(block_ID%cells_z)
    loc_x = pad_x[pos_x : pos_x + 2 * n + 1]
    loc_x = loc_x[np.where(loc_x >= 0)]
    loc_y = pad_y[pos_y : pos_y + 2 * n + 1]
    loc_y = loc_y[np.where(loc_y >= 0)]
    loc_z = pad_z[pos_z : pos_z + 2 * n + 1]
    loc_z = loc_z[np.where(loc_z >= 0)]

    cube = np.zeros((len(loc_z),len(loc_y), len(loc_x)), dtype=int)
    
    d=0
    for i in loc_x:
        c=0
        for j in loc_y:
            
            cube[:,c,d] = loc_z + step_x*i + step_y*j
            c+=1
        d+=1
    # print("the neighbourhood", square)
    return np.ndarray.flatten(cube)
@njit
def GetNeighbourhood_opt(n, cells_x, cells_y, cells_z, block_ID):
    """Will return the neighbourhood of a given block for a given n
    in a mesh made of square cells

    It will assume cells_x=ylen
    
    50 times faster than the non optimized"""
    
    step_y=cells_z
    step_x=cells_z*cells_y
    pad_x = np.concatenate((np.zeros(n)-1,  np.arange(cells_x), np.zeros(n)-1))
    pad_y = np.concatenate((np.zeros(n)-1,  np.arange(cells_y), np.zeros(n)-1))
    pad_z = np.concatenate((np.zeros(n)-1,  np.arange(cells_z), np.zeros(n)-1))
    pos_x, pos_y, pos_z =int( block_ID/(cells_y*cells_z)), int(int(block_ID%step_x)/cells_z), int(block_ID%cells_z)
    
    loc_x = pad_x[pos_x : pos_x + 2 * n + 1]
    loc_x = loc_x[np.where(loc_x >= 0)]
    loc_y = pad_y[pos_y : pos_y + 2 * n + 1]
    loc_y = loc_y[np.where(loc_y >= 0)]
    loc_z = pad_z[pos_z : pos_z + 2 * n + 1]
    loc_z = loc_z[np.where(loc_z >= 0)]

    cube = np.zeros((len(loc_z)*len(loc_y)*len(loc_x)), dtype=np.int64)
    
    c=0
    for i in loc_x:
        for j in loc_y:
            cube[c*len(loc_z): (c+1)*len(loc_z)] = loc_z + step_x*i + step_y*j
            c+=1
        
    return cube


#%%

from numba import boolean

@njit
def in1D(arr1, arr2):
    """Return a boolean array indicating which elements of `arr1` are in `arr2`."""
    arr2_sorted = np.sort(arr2)
    indices = np.searchsorted(arr2_sorted, arr1)
    return (arr2_sorted[indices] == arr1)
@njit
def KernelIntegralSurfaceFast(s_blocks, tau, h_net, pos_s, source_edge, center, normal, neighbourhood, function, K, D, h):
    """Returns the kernel that multiplied (scalar, dot) by the array of fluxes (q) returns
    the integral of the rapid term over the surface
    
    Main function used to calculate J
    
    h must be the disc size of the mesh_3D
    """
    
    sources=np.arange(len(s_blocks))[in1D(s_blocks, neighbourhood)]
    #sources=in1D(s_blocks, neighbourhood)
    q_array=np.zeros(len(sources), dtype=float)
    c=0
    for i in sources:
        ed=source_edge[i]
        a,b= pos_s[i]-tau[ed]*h_net[ed]/2, pos_s[i]+tau[ed]*h_net[ed]/2
        q_array[c]=SimpsonSurface(a,b,function, center,h, normal,D)
        c+=1
    return q_array,  sources



