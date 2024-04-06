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
    
    # Create a dictionary to hold your constants
    constants = {
        'Nt': struct_c['Nt'].item(),  # Number of transmitters
        'N': struct_c['N'].item(),    # Number of time samples
        'Nr': struct_c['Nr'].item(),  # Number of receivers
        'M': struct_c['M'].item(),    # Number of targets
        'Lj': struct_c['Lj'].item(),  # l_M - l_1 + N
    }
    
    constants["Ls"] = constants['Nt']*constants['N']
    constants["Lw"] = constants['Lj']*constants['Nr']
    
    return constants

def load_param_from_setup(file_path):
    # Load the .mat file
    data = loadmat(file_path, squeeze_me=True)
    struct_c = data['struct_c']
    
class Struct_c:
    def __init__(self, data):
        struct_c = data['struct_c']
        self.Nt = struct_c['Nt'].item()  # Number of transmitters
        self.N = struct_c['N'].item()    # Number of time samples
        self.Nr = struct_c['Nr'].item()  # Number of receivers
        self.M = struct_c['M'].item()    # Number of targets
        self.K = struct_c['K'].item()    # Number of interference
        self.Lj = struct_c['Lj'].item()  # l_M - l_1 + N

class Struct_k:
    def __init__(self, data):
        struct_k = data['struct_k']
        self.r = struct_k['r']
        self.sigma_k = struct_k['sigma_k']
    