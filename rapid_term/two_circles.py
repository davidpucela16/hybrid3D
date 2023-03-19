#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:38:56 2022

@author: pdavid

variation of the 2D green's function's gradient around a circle
"""

import numpy as np 
import matplotlib.pyplot as plt 

i=np.logspace(0,1,7)+1
R=10
theta=np.linspace(0,2*np.pi,300)
arr=np.array([])

plt.figure(figsize=(12,12))
plt.title('gradient of the single layer potential over a circle at a distance \n a=L/R')
for a in i:
    f=-(a*np.cos(theta)+1)/(a**2+2*a*np.cos(theta) + 1)
    plt.plot(theta/2/np.pi,f, label='a={}'.format(np.around(a,2)))
    plt.xlabel('radians')
    arr=np.append(arr,np.sum(f[:-1]))
    
    plt.legend()
    
plt.plot(theta/2/np.pi, np.zeros(len(theta)))
plt.show()
plt.plot(i,arr)
plt.ylim(-0.01,0.01)
plt.title("integral of this gradient over the circle\n")
    
    