# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 23:28:14 2024

@author: jbk5816
"""


from scipy.io import loadmat
import torch

def load_mapVector(file_path):
    # Load the .mat file
    data = loadmat(file_path, squeeze_me=True)
    
    # Extract the variable v_M from the file
    v_M = torch.tensor(data['v_M'], dtype=torch.float32)
    
    # Set Lv based on the first dimension of v_M
    Lv = v_M.shape[0]
    
    return v_M, Lv