#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:08:11 2023

@author: pdavid
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:26:11 2023

@author: pdavid
"""

from mesh_1D import mesh_1D
from mesh import cart_mesh_3D
import pdb
import numpy as np
import matplotlib.pyplot as plt

L=4
D=1
mesh=cart_mesh_3D(np.array([L,L,L+2]), 20)
mesh.assemble_boundary_vectors()

#%%

startVertex=np.array([0])
endVertex=np.array([1])
pos_vertex=np.array([[L/2,L/2,0],[L/2,L/2,L]])
vertex_to_edge=[[0],[0,1,2],[1,],[2]]
diameters=np.array([0.15])
h=np.array([0.1])



a=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h)
a.pos_arrays(mesh)

#%%
slice_perp=mesh.get_y_slice(L/2)

mesh_2D=np.zeros((0,3))
SL=np.zeros(len(slice_perp))
DL=np.zeros(len(slice_perp))
c=0
for i in slice_perp:
    mesh_2D=np.vstack((mesh_2D, mesh.get_coords(i)))
    
    for j in range(len(a.pos_s)):
        tup_args=(a.a_array[j], a.b_array[j], a.R[a.source_edge[j]], D, mesh.get_coords(i))
        r,s=a.get_source_potential(tup_args,D)
        SL[c]+=r
        DL[c]+=s
        
    c+=1
    
#%%
plt.imshow(SL.reshape(mesh.cells_x, mesh.cells_y)); plt.colorbar()

