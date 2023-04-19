#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 12:26:41 2023

@author: pdavid
"""

import numpy as np
import matplotlib.pyplot as plt 
import pdb

import sys
sys.path.append('../src/')

from small_functions import append_sparse

# =============================================================================
# def Green_line_orig(arg, D):
#     """Returns the average value of the integral without the coefficient i.e. to the result one would have to multiply
#     by the surface of the open cylinder (2 \pi R_j L_j)/(4*pi*D) to obtain a proper single layer potential
#     
#     DONT FORGET THE DIFUSSION COEFFICIENT
#     
#     THIS IS THE ORIGINAL FUNCTION RIGHT AFTER INTEGRATION WHICH WILL CONTAIN SINGULARITIES"""
#     x, a, b=arg
#     ra=np.linalg.norm(x-a)
#     rb=np.linalg.norm(x-b)
#     
#     L=np.linalg.norm(b-a)
#     tau=(b-a)/L
#     
#     num=ra + L/2 + np.dot(x-(a+b)/2,tau)
#     den=rb-L/2+np.dot(x-(a+b)/2,tau)
#     
#     return np.log(num/den)/(4*np.pi*D)
# =============================================================================
#from numba import njit

#@njit
def log_line(arg):
    """Returns the average value of the integral without the coefficient i.e. to the result one would have to multiply
    by the surface of the open cylinder (2 \pi R_j L_j)/(4*pi*D) to obtain a proper single layer potential
    
    DONT FORGET THE DIFUSSION COEFFICIENT"""
    x, a, b=arg
    ra=np.linalg.norm(x-a)
    rb=np.linalg.norm(x-b)
    
    L=np.linalg.norm(b-a)
    tau=(b-a)/L
    
    log=np.log((np.max([ra, rb]) + L/2 + np.abs(np.dot((a+b)/2-x,tau)))/(np.min([ra, rb]) - L/2 + np.abs(np.dot((a+b)/2-x,tau))))
    
    return log

def grad_point(arg):
    x,x_j=arg
    return -(x-x_j)/(np.linalg.norm(x-x_j)**3)


def get_source_potential(tup_args,D):
    """Returns two arrays, one to multiply to q and another to multiply to Cv
    It includes both the single layer and double layer
        - center_source is self explanatory, center of the cylinder/segment
        - R is the radius of the cylinder
        - D is the diffusion coefficient
        - K is the effective diffusivity of the wall [m^2 s^-1]
        - x is the point where the potential is calculated
    """
    a,b,x,R,K=tup_args
    L=np.linalg.norm(a-b)
    tau=(b-a)/L
    Dj=np.dot((grad_point((x,b))-grad_point((x,a))),tau)*R**2/4
    Sj=log_line((x, a,b))/(4*np.pi*D)
    
    
    #return(Sj-Dj/K, Dj)
    return(Sj, 0)

def get_grad_source_potential( tup_args, D):
    
    a,b,x,_,_=tup_args
    tau=(b-a)/np.linalg.norm(b-a)
    x_j=(a+b)/2
    #The second empty array that is returned is the corresponding to the C_v
    return np.linalg.norm(b-a)*grad_point((x,x_j))/(4*np.pi*D), np.array([])
    #I have multiplied the grad point by the length of the segment, Im pretty sure it needs to be done to be consistent dimensionally 


def Simpson_surface(arg, function, center, h, normal, D):
    """Assumes a square surface
    
    As always, the integral must be multiplied by the surface OUTSIDE of the
    function
    
    source must be given as a tuple. If the function is:
        - Green_3D
        - grad_Green_3D
        - get_grad_source_potential
        
    h and normal define the square surface where the function is integrated 
        - source is the position of the center of the source 
        - function is the function to be integrated
        - center is the surface of integration
        - h is the size of the square surface
        - normal is the normal vector to the surface
        - D is the diffusion coefficient
 
    else: arg=x_j 
        """
    a,b,R,K=arg    
        
    w_i=np.array([1,4,1,4,16,4,1,4,1])/36
    
    corr = np.array([[-1,-1,],
                [0, -1],
                [1, -1],
                [-1, 0],
                [0, 0],
                [1, 0],
                [-1, 1],
                [0, 1],
                [1, 1]])*h/2
    #This only works because it is a Cartesian grid and the normal has to be parallel to one of the axis
    #Other wise we would need another way to calculate this tangential vector
    #tau represents one of the prependicular vectors to normal 
    tau_1, tau_2=np.zeros(3), np.zeros(3)
    tau_1[np.where(normal==0)[0][1]]=1
    tau_2[np.where(normal==0)[0][1]]=1
    integral=0
    if function==get_source_potential:
        
        for i in range(len(w_i)):
            pos=center+corr[i,0]*tau_1+corr[i,1]*tau_2
            tup_args=a,b,pos,R,K
            #The function returns two kernels that cannot be multiplied 
            w_integral,_=function(tup_args, D)
            integral+=w_integral*w_i[i]
    elif function==get_grad_source_potential:
        
        for i in range(len(w_i)):
            pos=center+corr[i,0]*tau_1+corr[i,1]*tau_2
            tup_args=a,b,pos,R,K
            #The function returns two kernels that cannot be multiplied 
            grad,_=function(tup_args, D)
            integral+=np.dot(grad,normal )*w_i[i]
            
    return integral


def unit_test_Simpson(h,D):
    normal=np.array([0,1,0])
    a=np.dot(Simpson_surface(np.array([0,0,0]), grad_point, np.array([0,h/2,0]),h, normal, D)[0],normal)*h**2
    #The following is the exact result of the Simpson's integration done by hand (notebook)
    print("Analytical Simpson= ", -(27**-0.5+2**0.5+4)/(9*np.pi*D))
    #The following is the value returned by the function 
    print("Calculated Simpson= ", a)
    #The following is the analytical value of the integral
    print("Analytical (Gauss)= ", -1/(6*D))
    return


def get_self_influence(R,L, D):
    #####################################
    # REVIEW THE GEOMETRICAL COEFFICIENTS
    ####################################"
    a=np.array([0,0,0])
    b=np.array([0,0,L])
    x1=np.array([0,R,0])
    x2=np.array([0,R,L/2])
    x3=np.array([0,R,L])
    G1=log_line((x1,a,b))
    G2=log_line((x2,a,b))
    G3=log_line((x3,a,b))
    return (G1+4*G2+G3)/6/(4*np.pi*D)