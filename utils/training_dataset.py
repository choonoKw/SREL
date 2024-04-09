# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:22:51 2024

@author: jbk5816
Read training data for each case
"""

from torch.utils.data import Dataset
from scipy.io import loadmat
import torch

class TrainingDataSet(Dataset):
    def __init__(self, mat_file, stage=1):
        data = loadmat(mat_file)
        self.phi_list = torch.tensor(data['phi_list'], dtype=torch.float64)
        self.G_M_list = torch.tensor(data['G_M_list'], dtype=torch.complex64)
        self.H_M_list = torch.tensor(data['H_M_list'], dtype=torch.complex64)
        self.w_M_list = torch.tensor(data['w_M_list'], dtype=torch.complex64)
        
        self.y_M= torch.tensor(data['y_M'], dtype=torch.float32)
        self.Ly = self.y_M.shape[0]
        
        
    def __len__(self):
        return self.G_M_list.shape[-1]  # Number of training samples

    def __getitem__(self, idx):
    
        phi = self.phi_list[:,idx]
        w_M = self.w_M_list[:,:,idx]
        G_M = self.G_M_list[:, :, :, idx]
        H_M = self.H_M_list[:, :, :, idx]
            
        return phi, w_M, G_M, H_M