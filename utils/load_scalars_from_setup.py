# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 14:10:08 2024

@author: jbk5816

 # Load the .mat file
"""

from scipy.io import loadmat

def load_scalars_from_setup(file_path):
    # Load the .mat file
    data = loadmat(file_path, squeeze_me=True)
    
    # Assuming 'struct_c' is stored directly at the top level of the .mat file
    struct_c = data['struct_c']
    
    # Extract scalar values
    Nt = struct_c['Nt'].item()   # Number of transmitters
    N = struct_c['N'].item()     # Number of time samples
    Nr = struct_c['Nr'].item()   # Number of receivers
    M = struct_c['M'].item()     # Numer of targets
    Lj = struct_c['Lj'].item()   # l_M - l_1 + N
    
    return Nt, N, Nr, M, Lj