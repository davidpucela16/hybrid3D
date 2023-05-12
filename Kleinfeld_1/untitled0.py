#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 17:29:09 2023

@author: pdavid
"""


import os 
path=os.path.dirname(__file__)
os.chdir(path)

import pandas as pd
import numpy as np 

#%%

filename='/home/pdavid/Bureau/PhD/Network_Flo/All_files/Network1_Figure_Data.txt'
#%%
import os

def split_file(filename, output_dir):
    with open(filename, 'r') as file:
        output_files = []
        current_output = None
        line_counter = 0

        for line in file:
            line_counter += 1

            if line_counter < 25:
                continue

            if line.startswith('@'):
                if current_output:
                    current_output.close()
                output_filename = f"output_{len(output_files)}.txt"
                output_path = os.path.join(output_dir, output_filename)
                current_output = open(output_path, 'w')
                output_files.append(output_path)

            if current_output:
                current_output.write(line)

        if current_output:
            current_output.close()

        return output_files

# Usage:
output_dir = '/home/pdavid/Bureau/PhD/Network_Flo/All_files/'  # Specify the output directory here
os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
output_files = split_file(filename, output_dir)

print("Split files:")
for file in output_files:
    print(file)



#%%
df = pd.read_csv('./test_net.txt', skiprows=24,sep="\s+", names=["x", "y", "z"])

# Print the DataFrame
print(df)

path_src=os.path.join(path, '../src_full_opt')
os.chdir(path_src)






#%%

import scipy.sparse as sps
from numba import njit

def test_1():
    A=sps.lil_matrix((10**5,10**5), dtype=np.int64)
    for i in range(10**3):
        A[i, np.arange(10)+i]=np.arange(10)
    return(A)

@njit
def test_2():
    A_1=np.zeros(0, dtype=np.int64)
    A_2=np.zeros(0, dtype=np.int64)
    A_3=np.zeros(0, dtype=np.int64)
    
    for i in range(10**3):
        A_1=np.concatenate((A_1, np.arange(10)))
        A_2=np.concatenate((A_2, np.zeros(10)+i))
        A_1=np.concatenate((A_3, np.arange(10)))
        
    return A_1, A_2, A_3