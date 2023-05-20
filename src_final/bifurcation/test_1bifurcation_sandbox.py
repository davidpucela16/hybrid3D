#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 18:52:07 2023

@author: pdavid
"""

import os 
path=os.path.dirname(__file__)
os.chdir(path)

path_src=os.path.join(path, '../')
os.chdir(path_src)



import numpy as np

import matplotlib.pyplot as plt
import scipy as sp
import scipy.sparse.linalg
import math

import pdb

import matplotlib.pylab as pylab
plt.style.use('default')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15,15),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large', 
         'font.size': 24,
         'lines.linewidth': 2,
         'lines.markersize': 15}
pylab.rcParams.update(params)


from assembly import AssemblyDiffusion3DInterior, AssemblyDiffusion3DBoundaries
from mesh import cart_mesh_3D
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve as dir_solve
from scipy.sparse.linalg import bicg
import numpy as np
import matplotlib.pyplot as plt

import math
from assembly_1D import FullAdvectionDiffusion1D
from mesh_1D import mesh_1D
from GreenFast import GetSourcePotential
from Second_eq_functions import InterpolateFast,InterpolatePhiBarBlock
import pdb

from hybridFast import hybrid_set_up, Visualization3D

from neighbourhood import GetNeighbourhood, GetUncommon





L_vessel=240
# - This is the validation of the 1D transport eq without reaction
alpha=50
R_vessel=L_vessel/alpha
R_1D=np.array([1,0.5,0.5])*R_vessel
D_1D = 10
K=np.array([1,1,1])*R_1D/R_vessel

U = np.array([2,2,2])/L_vessel*10

U[1]=U[0]/2*(R_1D[0]/R_1D[1])**2
U[2]=U[1]


startVertex=np.array([0,1,1])
endVertex=np.array([1,2,3])
aux=L_vessel*15**0.5/4
pos_vertex=np.array([[L_vessel/2, 0, L_vessel/2],
                     [L_vessel/2, L_vessel,L_vessel/2],
                     [L_vessel/2, L_vessel+aux, L_vessel*3/4],
                     [L_vessel/2, L_vessel+aux,L_vessel/4]
                     ])

vertex_to_edge=[[0],[0,1,2], [1], [2]]
diameters=2*R_1D

cells_per_vessel=50
h=np.zeros(3)+L_vessel/cells_per_vessel

net=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h[0],D_1D)
net.U=U
net.D=D_1D


cells_3D=10
n=1
L_3D=np.array([L_vessel, 2*L_vessel, L_vessel])
mesh=cart_mesh_3D(L_3D,cells_3D)

mesh.AssemblyBoundaryVectors()

net.PositionalArraysFast(mesh)

BCs_1D=np.array([[0,1],
                 [2,0],
                 [3,0]])

BC_type=np.array(["Neumann", "Neumann", "Neumann","Neumann","Neumann","Neumann"])
BC_type=np.array(["Dirichlet", "Dirichlet","Neumann","Neumann", "Dirichlet","Dirichlet"])
BC_value=np.array([0,0,0,0,0,0])


prob=hybrid_set_up(mesh, net, BC_type, BC_value,n,1, np.zeros(len(diameters))+K, BCs_1D)
#%%
#%%
prob.AssemblyI()


tot_cell=cells_per_vessel*len(startVertex)
x=net.pos_s[:net.cells[0],1]
Pe_0=U[0]*L_vessel/D_1D
Pe_1=U[1]*L_vessel/D_1D
A,B,C,d=np.linalg.solve(np.array([[1,1,0,0],
                                  [np.exp(Pe_0), 1, -1, -1],
                                  [0,0,np.exp(Pe_1), 1],
                                  [np.exp(Pe_0)*Pe_0, 0, -Pe_1, 0]]),
                        np.array([1,0,0,0]))


analytical_0 = lambda s : A*np.exp(Pe_0*s/L_vessel)+B
analytical_1 = lambda s : C*np.exp(Pe_1*s/L_vessel)+d


full_analytical=np.concatenate((analytical_0(x), analytical_1(x), analytical_1(x)))

text_1 = 'vessel 0'
text_2 = 'vessel 1'
text_3 = 'vessel 2'
props = dict(boxstyle='round', facecolor='white', alpha=0.5)

plt.title("Converged estimation of the flux")
plt.plot(full_analytical, label="analytical")

plt.plot(dir_solve(prob.I_matrix, -prob.III_ind_array), label="numerical")
plt.text(0, 0.6, text_1,  fontsize=40,
        verticalalignment='top', bbox=props)

plt.text(10, 0.6, text_2,  fontsize=40,
        verticalalignment='top', bbox=props)

plt.text(20, 0.6, text_3,  fontsize=40,
        verticalalignment='top', bbox=props)
plt.legend()
plt.show()

#%%

#%%
from assembly_1D import FullAdvectionDiffusion1D, AssemblyTransport1D,AssemblyVertices,AssemblyTransport1DFast
data, row, col=AssemblyTransport1D(np.ndarray.flatten(U), 1, net.h, net.cells)
data_2, row_2, col_2=AssemblyTransport1DFast(np.ndarray.flatten(U), 1, net.h, net.cells)


one=sp.sparse.csc_matrix((data, (row, col)), shape=(np.sum(net.cells), np.sum(net.cells)))
two=sp.sparse.csc_matrix((data_2, (row_2, col_2)), shape=(np.sum(net.cells), np.sum(net.cells)))

#%%
# =============================================================================
# from hybridFast import AssemblyBArrays_parallel
# from numba.typed import List
# prob=hybrid_set_up(mesh, net, BC_type, BC_value,n,1, np.zeros(len(diameters))+K, BCs_1D)
# mesh.GetOrderedConnectivityMatrix()
# AssemblyBArrays_parallel(List(mesh.ordered_connect_matrix), mesh.size_mesh, n,1,
#                                 mesh.cells_x, mesh.cells_y, mesh.cells_z, mesh.pos_cells, mesh.h, 
#                                 net.s_blocks, net.tau, net.h, net.pos_s, net.source_edge)
# =============================================================================

#%%
#prob.AssemblyProblem()

#print("If all BCs are newton the sum of all coefficients divided by the length of the network should be close to 1", np.sum(prob.B_matrix.toarray())/net.L)

import pstats
import cProfile
if __name__ == '__main__':
    # run the profiler on the code and save results to a file
    cProfile.run('prob.AssemblyProblem()', filename=path + '/profile_results.prof')



stats = pstats.Stats(path + '/profile_results.prof')

stats.strip_dirs().sort_stats('cumulative').print_stats(30)

#%% - We solve the 2D problem in a 3D setting to be able to validate. Ideally, there is no transport in the y direction
C_v_array=np.ones(len(net.pos_s)) #Though this neglects completely the contribution from the dipoles

L1=sp.sparse.hstack((prob.A_matrix,prob.B_matrix))
L2=sp.sparse.hstack((prob.D_matrix, prob.E_matrix))

Li=sp.sparse.vstack((L1,L2))

ind=np.concatenate((prob.I_ind_array, prob.F_matrix.dot(C_v_array)))

sol=dir_solve(Li, -ind)


plt.plot(net.pos_s[:,1],sol[-prob.S:], label='hybrid reaction')
plt.title("q(s) with C_v=1")
plt.legend()
plt.show()

prob.s=sol[:-prob.S]
prob.q=sol[-prob.S:]


phi_bar=np.array([])
for i in range(prob.S):
    phi_bar=np.append(phi_bar, -prob.q[i]/prob.K[net.source_edge[i]]+1)
    

plt.plot(prob.q, label="q")
plt.plot(phi_bar, label="phi_bar")
plt.legend()
#%%
import dask
@dask.delayed
def PhiBarHelper(args):
    block, lst=args
    path,n, cells_x, cells_y, cells_z, h_3D,pos_cells,s_blocks, source_edge,tau, pos_s, h_1D, R, D,sources_per_block, quant_sources_per_block=lst
    print("block", block)
    kernel_s,row_s, col_s,kernel_q, row_q, col_q=InterpolatePhiBarBlock(block,n, cells_x, cells_y, cells_z, h_3D, 
                                 pos_cells,s_blocks, source_edge,tau, pos_s, h_1D, R, D, 
                                 sources_per_block, quant_sources_per_block)
    
    np.save(path + '/{}_kernel_s'.format(block), kernel_s)
    np.save(path + '/{}_row_s'.format(block), row_s)
    np.save(path + '/{}_col_s'.format(block), col_s)
    
    np.save(path + '/{}_kernel_q'.format(block), kernel_q)
    np.save(path + '/{}_row_q'.format(block), row_q)
    np.save(path + '/{}_col_q'.format(block), col_q)
    
    return kernel_s,row_s, col_s,kernel_q, row_q, col_q

#%%
pdb.set_trace()
data=np.random.random((10,3))*1100

#%%
lst=path+'/matrices', prob.n, prob.mesh_3D.cells_x, prob.mesh_3D.cells_y, prob.mesh_3D.cells_z, prob.mesh_3D.h,prob.mesh_3D.pos_cells,prob.mesh_1D.s_blocks, prob.mesh_1D.source_edge,prob.mesh_1D.tau, prob.mesh_1D.pos_s, prob.mesh_1D.h, prob.mesh_1D.R, 1,prob.mesh_1D.sources_per_block, prob.mesh_1D.quant_sources_per_block
results=[]
for block in prob.mesh_1D.uni_s_blocks:
    results.append(PhiBarHelper((block, lst)))
    
#%%
import time
begin=time.time()
dask.compute(results)
end=time.time()
    
#%%
# =============================================================================
# #%%  - Validation with 2D code
# 
# Lin_matrix=sp.sparse.hstack((prob.A_matrix, prob.B_matrix))
# Lin_matrix=sp.sparse.vstack((Lin_matrix, sp.sparse.hstack((prob.D_matrix, prob.E_matrix))))
# 
# prob.C_v_array=np.ones(len(net.pos_s))
# 
# 
# 
# sol=dir_solve(Lin_matrix, -np.concatenate((prob.I_ind_array, prob.F_matrix.dot(prob.C_v_array))))
# 
# prob.s=sol[:prob.F]
# prob.q=sol[prob.F:]
# plt.plot(prob.q)
# 
# 
# =============================================================================
#%%

prob.AssemblyProblem()
prob.H_matrix/=4
Lin_matrix=prob.ReAssemblyMatrices()
prob.Full_linear_matrix=Lin_matrix
prob.SolveProblem()

for i in range(prob.S):
    phi_bar=np.append(phi_bar, -prob.q[i]/prob.K[net.source_edge[i]]+1)
    

plt.plot(prob.q, label="q")
plt.plot(phi_bar, label="phi_bar")
plt.legend()

a=Visualization3D([0, L_vessel], 50, prob, 12, 0.08, np.array([0,L_vessel/2,0]))

#%%
prob.AssemblyDEFFast(path + '/matrices')
for i in range(cells_per_vessel*3):
    b=np.where(prob.D_E_F_matrix.toarray()[i]!=prob.Middle.toarray()[i])[0]
    if np.any(b):
        print(b)
        print(prob.D_E_F_matrix.toarray()[i,b]-prob.Middle.toarray()[i,b])
