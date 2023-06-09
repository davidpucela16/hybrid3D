#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 18:52:07 2023

@author: pdavid
"""

import os 
path=os.path.dirname(__file__)
os.chdir(path)
from Potentials_module import Classic
from Potentials_module import Gjerde
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
          'figure.figsize': (10,10),
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
from Green import GetSourcePotential
import pdb

from hybrid_set_up_noboundary import hybrid_set_up, Visualization3D

from neighbourhood import GetNeighbourhood, GetUncommon
import copy 


BC_type=np.array(["Neumann", "Neumann", "Neumann","Neumann","Neumann","Neumann"])
BC_type=np.array(["Dirichlet", "Dirichlet","Neumann","Neumann", "Dirichlet","Dirichlet"])
BC_value=np.array([0,0,0,0,0,0])
L_vessel=240
cells_3D=5
n=20
L_3D=np.array([L_vessel, 3*L_vessel, L_vessel])
mesh=cart_mesh_3D(L_3D,cells_3D)

mesh.AssemblyBoundaryVectors()


#%%
# - This is the validation of the 1D transport eq without reaction

D = 1
K=np.array([0.0001,10,0.0001])

U = np.array([2,2,2])*100/L_vessel
alpha=10
R_vessel=L_vessel/alpha
R_1D=np.zeros(3)+R_vessel

startVertex=np.array([0,1,2])
endVertex=np.array([1,2,3])
pos_vertex=np.array([[L_vessel/2, 0, L_vessel/2],
                     [L_vessel/2, L_vessel,L_vessel/2],
                     [L_vessel/2, 2*L_vessel, L_vessel/2],
                     [L_vessel/2, L_vessel*3,L_vessel/2]
                     ])

vertex_to_edge=[[0],[0,1], [1,2], [2]]
diameters=np.array([2*R_vessel, 2*R_vessel, 2*R_vessel])





#%%
for alpha in np.array([10,20,50,100]):
    for k in np.array([1,2,5,10,15,20,25]):
        for l in np.array([1,2,5,10,15,20,25]):
            
            U = np.array([1,1,1])*5*l/L_vessel
            K=np.array([0.0001,k,0.0001])
            
            cells_per_vessel_fine=100
            cells_per_vessel_coarse=10
            
            h_fine=np.zeros(3)+L_vessel/cells_per_vessel_fine
            h_coarse=np.zeros(3)+L_vessel/cells_per_vessel_coarse
            
            net_fine=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h_fine,1)
            net_fine.U=U
            net_fine.D=D
            net_fine.PositionalArrays(mesh)
            
            net_coarse=mesh_1D(startVertex, endVertex, vertex_to_edge ,pos_vertex, diameters, h_coarse,1)
            net_coarse.U=U
            net_coarse.D=D
            net_coarse.PositionalArrays(mesh)
            
            BCs_1D=np.array([[0,1],
                             [3,0]])
            
            prob_fine=hybrid_set_up(mesh, net_fine, BC_type, BC_value,n,1, np.zeros(len(diameters))+K, BCs_1D)
            #mesh.GetOrderedConnectivityMatrix()
            
            prob_fine.AssemblyProblem()
            prob_fine.SolveProblem()
            q_line_fine=prob_fine.q.copy()
            Cv_line_fine=prob_fine.Cv.copy()
            
            
            P=Classic(3*L_vessel, R_vessel)
            
            G_ij_fine=P.get_single_layer_vessel(len(net_fine.pos_s))/2/np.pi/R_vessel
            #The factor 2*np.pi*R_vessel arises because we consider q as the total flux and not the point gradient of concentration
            
            new_E_matrix=G_ij_fine+prob_fine.Permeability
            
            H_ij=P.get_double_layer_vessel(len(net_fine.pos_s))
            
            F_matrix_line=prob_fine.F_matrix.copy()
            E_matrix_line=prob_fine.E_matrix.copy()
            
            F_matrix_cyl=prob_fine.F_matrix+H_ij
            E_matrix_cyl=new_E_matrix-H_ij*1/K[net_fine.source_edge]
            
            prob_fine.E_matrix=E_matrix_cyl
            prob_fine.F_matrix=F_matrix_cyl
            Exact_full_linear=prob_fine.ReAssemblyMatrices()
            
            sol_cyl=dir_solve(Exact_full_linear, -prob_fine.Full_ind_array)
            
            q_cyl_fine=sol_cyl[-prob_fine.S*2:-prob_fine.S]
            Cv_cyl_fine=sol_cyl[-prob_fine.S:]
            
            
            
            plt.plot(q_cyl_fine, label='q_cyl')
            plt.plot(q_line_fine, label='q_line')
            plt.legend()
            plt.show()
            
            P=Classic(3*L_vessel, R_vessel)
            G_ij_coarse=P.get_single_layer_vessel_coarse(30, 10)/2/np.pi/R_vessel
            #The factor 2*np.pi*R_vessel arises because we consider q as the total flux and not the point gradient of concentration
            
            prob_coarse=hybrid_set_up(mesh, net_coarse, BC_type, BC_value,n,1, np.zeros(len(diameters))+K, BCs_1D)
            prob_coarse.AssemblyProblem()
            prob_coarse.SolveProblem()
            q_line_coarse=prob_coarse.q.copy()
            Cv_line_coarse=prob_coarse.Cv.copy()
            
            
            new_E_matrix=G_ij_coarse+prob_coarse.Permeability
            
            H_ij_coarse=P.get_double_layer_vessel_coarse(len(net_coarse.pos_s), 10)
            
            F_matrix_line=prob_coarse.F_matrix.copy()
            E_matrix_line=prob_coarse.E_matrix.copy()
            
            # =============================================================================
            # F_matrix_cyl=prob_coarse.F_matrix+H_ij_coarse
            # E_matrix_cyl=new_E_matrix-H_ij_coarse*1/K[net_coarse.source_edge]
            # =============================================================================
            
            E_matrix_cyl=new_E_matrix
            F_matrix_cyl=prob_coarse.F_matrix+H_ij_coarse
            
            prob_coarse.E_matrix=E_matrix_cyl
            prob_coarse.F_matrix=F_matrix_cyl
            Exact_full_linear=prob_coarse.ReAssemblyMatrices()
            
            sol_cyl=dir_solve(Exact_full_linear, -prob_coarse.Full_ind_array)
            
            q_cyl_coarse=sol_cyl[-prob_coarse.S*2:-prob_coarse.S]
            Cv_cyl_coarse=sol_cyl[-prob_coarse.S:]
            
            
            
            plt.plot(net_coarse.pos_s[:,1],q_cyl_coarse, label='q_cyl')
            plt.plot(net_coarse.pos_s[:,1],q_line_coarse, label='q_line')
            plt.plot(net_fine.pos_s[:,1],q_cyl_fine, label='exact')
            plt.legend()
            plt.title("q for Pe={}, Da_m={} and L/R={}".format(np.around(U[1]*L_vessel, decimals=2), np.around(K[1]*alpha/2/np.pi, decimals=2), alpha))
            plt.savefig(path+"/Pe={}_Da_m={}_LdivR={}.pdf".format(int(U[1]*L_vessel), int(K[1]*alpha/2/np.pi), alpha))
            plt.show()





