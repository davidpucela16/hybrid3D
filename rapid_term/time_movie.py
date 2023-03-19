#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 14:51:41 2021

@author: pdavid
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

D=1

points=100
R=np.linspace(1/points,1,points)
T=np.linspace(1/points,1,points)

Sol=np.zeros((points, points))
Sol_ss=np.zeros(points)

def G_temp_3D(r, t, D):
    return(np.exp(-r**2/(4*D**2*t))/(4*np.pi*D**2))

for t in range(len(T)):
    for r in range(len(R)):
        Sol[t,r]=G_temp_3D(R[r], T[t], D)

Sol_ss[1:]=np.log(1/R[1:])/(2*np.pi*D)
        
        

fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.1)
line, = ax.plot(0, 0)


x_data = []
y_data = []

def animation_frame(i):
    ind=int(np.where(np.arange(0,10,0.1)==i)[0][0])
    print(ind)
    x_data=R
    y_data=Sol[ind,:]*Sol_ss
    
    line.set_xdata(x_data[1:])
    line.set_ydata(y_data[1:])
    return line, 

animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(0, 10, 0.1), interval=10)
plt.show()


# =============================================================================
# fig, ax = plt.subplots()
# ax.set_xlim(0, 105)
# ax.set_ylim(0, 12)
# line, = ax.plot(0, 0)
# 
# 
# x_data = []
# y_data = []
# 
# def animation_frame(i):
#     print(i)
#     x_data.append(i * 10)
#     y_data.append(i)
#     
#     line.set_xdata(x_data)
#     line.set_ydata(y_data)
#     return line, 
# 
# animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(0, 10, 0.1), interval=10)
# plt.show()
# 
# =============================================================================
