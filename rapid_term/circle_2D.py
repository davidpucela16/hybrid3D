#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 13:21:51 2022

@author: pdavid

This script is basically to prove to myself the properties of the Green's function and its normal gradient when integrated over a closed surface
"""

import numpy as np
from numba import njit
import pdb

@njit
def grad_Green_2D(x, j, epsilon):
    d=np.float64(np.linalg.norm(x-j))
    if d<epsilon:
        a=np.float64(0)
        print("EPSILON")
    else:
        a=(2*np.pi*d)**-1
    e_r=(x-j)/d
    return(a*e_r)

    

def Green_2D(x, j, epsilon):
    d=np.float64(np.linalg.norm(x-j))
    if d<epsilon:
        a=np.zeros(len(x), dtype=np.float64)
    else:
        a=np.array(1/(2*np.pi)*np.log(1/d), dtype=float)
    return(a)

def circle_2D(x, points_theta, epsilon,R):
    """x is expected to be 0<x<1
    points_theta is the expected discretization of the circumference
    It integrates over the whole circumference """

    theta=np.linspace(0,2*np.pi*(points_theta-1)/points_theta,points_theta)
    
    dip=0
    dth=2*np.pi/points_theta
    G_3D=0
    #pdb.set_trace()
    surf=np.float64(0)
    for th in theta:
        normal=np.array([np.sin(th), np.cos(th)])
        current_point=np.array([R*np.sin(th), R*np.cos(th)])

        dip+=grad_Green_2D(x, current_point, epsilon).dot(normal)*dth*R
        G_3D+=Green_2D(x, current_point, epsilon)*dth*R
        surf+=dth*R
    print(surf)
    #pdb.set_trace()
    return(np.array([dip, G_3D, dth],dtype=np.float64))

R=0.01
r=np.linspace(2*R, 1,100)
G=0
dip=0
th=0
G_array=np.zeros(len(r))
dip_array=np.zeros(len(r))
c=0
for i in r:
    arr=circle_2D(np.array([0,i]), 20,R*0.5,R)
    G=arr[1]
    dip=arr[0]
    th=arr[2]
    G_array[c]=G
    dip_array[c]=dip
    c+=1
    
    
