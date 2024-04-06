# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 22:32:19 2024

@author: jbk5816
"""

import torch
import numpy as np

from utils.training_dataset import TrainingDataSet
from torch.utils.data import DataLoader

from utils.functions import sum_of_sinr_reciprocal

def test(constants,model_test):
    dataset = TrainingDataSet('data/data_trd_1e+02_val.mat')
    N_data = len(dataset)
    modulus = constants['modulus']
    
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    y_M = dataset.y_M
    
    N_iter = 2e2
    
    f_sinr_stack_list = np.zeros((N_data, N_iter))    
    
    # # Ly = dataset.Ly.to(device)
    with torch.no_grad():  # Disable gradient computation
        for idx_data, (
                phi_batch, w_M_batch, G_M_batch, H_M_batch
                ) in enumerate(val_loader):
            # phi = phi_batch.squeeze()
            
            for idx_iter in range(N_iter):
                s_batch = modulus*torch.exp(1j *phi_batch) 
                S_tilde = np.reshape(np.concatenate(
                    (s.reshape(-1, 1), np.zeros((Nt * (lm[M-1] - lm[0]), 1)))
                    ), (Nt, Lj))
                
                f_sinr = sum_of_sinr_reciprocal(G_M_batch, H_M_batch, s_batch)
                phi_batch = model_test(
                    phi_batch, w_M_batch, y_M, G_M_batch, H_M_batch
                    )
                
                f_sinr_stack_list[idx_]
                
    # G_M_list = dataset.G_M_list
    # H_M_list = dataset.H_M_list
    # phi_list = dataset.phi_list
    # w_M_list = dataset.w_M_list
    # y_M = dataset.y_M
    
    # for idx_data, (G_M, H_M, phi, w_M) in enumerate(zip(
    #         torch.unbind(G_M_list, dim=-1),torch.unbind(H_M_list, dim=-1),
    #         torch.unbind(phi_list, dim=-1),torch.unbind(w_M_list, dim=-1)
    #         )): 
        
    #     for idx_iter in range(N_iter):
    #         s = modulus*torch.exp(1j *phi) 
    #         f_sinr = sum_of_sinr_reciprocal(G_M, H_batch, s)
    #         phi_batch = model_test(
    #             phi_batch, w_M_batch, y_M, G_M_batch, H_M_batch
                )
        
        
    #     print(1);
        
    # return 0