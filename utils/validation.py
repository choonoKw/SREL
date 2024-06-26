# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 10:38:22 2024

@author: jbk5816
"""

from utils.training_dataset import TrainingDataSet

import torch
import numpy as np
from torch.utils.data import DataLoader

from utils.worst_sinr import worst_sinr_function
from utils.custom_loss import sum_of_reciprocal

def validation(constants,model_val):
    model_val.eval()
    device = model_val.device
    
    dataset = TrainingDataSet('data/data_trd_1e+02_val.mat')
    batch_size = 100
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    y_M = dataset.y_M.to(device)
    # Ly = dataset.Ly.to(device)
    with torch.no_grad():  # Disable gradient computation
        for phi_batch, w_M_batch, G_M_batch, H_M_batch in val_loader:
            # s_batch = modulus * torch.exp(1j * phi_batch)
            phi_batch = phi_batch.to(device)
            G_M_batch = G_M_batch.to(device)
            H_M_batch = H_M_batch.to(device)
            w_M_batch = w_M_batch.to(device)
            
            
            model_outputs = model_val(
                phi_batch, w_M_batch, y_M, G_M_batch, H_M_batch
                )
            
    
    # validation of rho values
    rho_M_stack_batch = model_outputs['rho_M_stack_batch']
    rho_M_stack_avg = torch.sum(rho_M_stack_batch, dim=0)/batch_size
    print('rho values = ')
    for n in range(model_val.N_step):
        for m in range(model_val.M):
            print(f'{rho_M_stack_avg[n,m].item():.4f}', end=",      ")
        print('')
        
    # SINR values for each step
    s_stack_batch = model_outputs['s_stack_batch']
    worst_sinr_stack_list= np.zeros((batch_size,model_val.N_step+1))
    f_stack_list = np.zeros((batch_size,model_val.N_step+1))
    for update_step in range(model_val.N_step+1):
        s_batch = s_stack_batch[:,update_step,:]
        worst_sinr_batch = worst_sinr_function(constants, s_batch, G_M_batch, H_M_batch)
        sinr_avg_db = 10*np.log10( np.sum(worst_sinr_batch)/batch_size )
        print(f'Step {update_step:02d}, SINR = {sinr_avg_db:.4f} dB')
        
        worst_sinr_stack_list[:,update_step] = worst_sinr_batch
        f_stack_list[:,update_step] = sum_of_reciprocal(constants, G_M_batch, H_M_batch, s_batch)
    
    return worst_sinr_stack_list, f_stack_list


def validation_intra_phase2(constants,model_val):
    model_val.eval()
    device = model_val.device
    
    dataset = TrainingDataSet('data/data_trd_1e+02_val.mat')
    batch_size = 100
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    y_M = dataset.y_M.to(device)
    # Ly = dataset.Ly.to(device)
    with torch.no_grad():  # Disable gradient computation
        for phi_batch, w_M_batch, G_M_batch, H_M_batch in val_loader:
            # s_batch = modulus * torch.exp(1j * phi_batch)
            phi_batch = phi_batch.to(device)
            G_M_batch = G_M_batch.to(device)
            H_M_batch = H_M_batch.to(device)
            w_M_batch = w_M_batch.to(device)
            
            
            model_outputs = model_val(
                phi_batch, w_M_batch, y_M, G_M_batch, H_M_batch
                )
            
    
    # validation of rho values
    rho_M_stack_batch = model_outputs['rho_M_stack_batch']
    rho_M_stack_avg = torch.sum(rho_M_stack_batch, dim=0)/batch_size
    print('rho values = ')
    for n in range(model_val.N_step):
        for m in range(model_val.M):
            print(f'{rho_M_stack_avg[n,m].item():.4f}', end=",      ")
        print('')
        
    # SINR values for each step
    s_stack_batch = model_outputs['s_stack_batch']
    worst_sinr_stack_list= np.zeros((batch_size,model_val.N_step+1))
    f_stack_list = np.zeros((batch_size,model_val.N_step+1))
    for update_step in range(model_val.N_step+1):
        s_batch = s_stack_batch[:,update_step,:]
        worst_sinr_batch = worst_sinr_function(constants, s_batch, G_M_batch, H_M_batch)
        sinr_avg_db = 10*np.log10( np.sum(worst_sinr_batch)/batch_size )
        print(f'Step {update_step:02d}, SINR = {sinr_avg_db:.4f} dB')
        
        worst_sinr_stack_list[:,update_step] = worst_sinr_batch
        f_stack_list[:,update_step] = sum_of_reciprocal(constants, G_M_batch, H_M_batch, s_batch)
    
    return worst_sinr_stack_list, f_stack_list


# def validation(constants,model_val):
#     model_val.eval()
#     device = model_val.device
    
#     dataset = TrainingDataSet('data/data_trd_1e+02_val.mat')
#     batch_size = 100
#     val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
#     y_M = dataset.y_M.to(device)
#     # Ly = dataset.Ly.to(device)
#     with torch.no_grad():  # Disable gradient computation
#         for phi_batch, w_M_batch, G_M_batch, H_M_batch in val_loader:
#             # s_batch = modulus * torch.exp(1j * phi_batch)
#             phi_batch = phi_batch.to(device)
#             G_M_batch = G_M_batch.to(device)
#             H_M_batch = H_M_batch.to(device)
#             w_M_batch = w_M_batch.to(device)
            
            
#             model_outputs = model_val(phi_batch, w_M_batch, y_M)
            
    
#     # validation of rho values
#     rho_M_stack_batch = model_outputs['rho_M_stack_batch']
#     rho_M_stack_avg = torch.sum(rho_M_stack_batch, dim=0)/batch_size
#     print('rho values = ')
#     for n in range(model_val.N_step):
#         for m in range(model_val.M):
#             print(f'{rho_M_stack_avg[n,m].item():.4f}', end=",      ")
#         print('')
        
#     # SINR values for each step
#     s_stack_batch = model_outputs['s_stack_batch']
#     for update_step in range(model_val.N_step+1):
#         s_batch = s_stack_batch[:,update_step,:]
#         worst_sinr_batch = worst_sinr_function(constants, s_batch, G_M_batch, H_M_batch)
#         sinr_avg_db = 10*np.log10( np.sum(worst_sinr_batch)/batch_size )
#         print(f'Step {update_step:02d}, SINR = {sinr_avg_db:.4f} dB')
#     sinr_avg_db_opt = sinr_avg_db
    
#     return sinr_avg_db_opt

# def validation_sred(constants,model_val):
#     model_val.eval()
#     device = model_val.device
    
#     dataset = TrainingDataSet('data/data_trd_1e+02_val.mat')
#     batch_size = 100
#     val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
#     y_M = dataset.y_M.to(device)
#     # Ly = dataset.Ly.to(device)
#     with torch.no_grad():  # Disable gradient computation
#         for phi_batch, w_M_batch, G_M_batch, H_M_batch in val_loader:
#             # s_batch = modulus * torch.exp(1j * phi_batch)
#             phi_batch = phi_batch.to(device)
#             G_M_batch = G_M_batch.to(device)
#             H_M_batch = H_M_batch.to(device)
#             w_M_batch = w_M_batch.to(device)
            
            
#             model_outputs = model_val(phi_batch, w_M_batch, y_M, G_M_batch, H_M_batch)
            
    
#     # validation of rho values
#     rho_M_stack_batch = model_outputs['rho_M_stack_batch']
#     rho_M_stack_avg = torch.sum(rho_M_stack_batch, dim=0)/batch_size
#     print('rho values = ')
#     for n in range(model_val.N_step):
#         for m in range(model_val.M):
#             print(f'{rho_M_stack_avg[n,m].item():.4f}', end=",      ")
#         print('')
        
#     # SINR values for each step
#     s_stack_batch = model_outputs['s_stack_batch']
    
#     worst_sinr_stack_list= np.zeros((batch_size,model_val.N_step+1))
#     f_stack_list = np.zeros((batch_size,model_val.N_step+1))
#     for update_step in range(model_val.N_step+1):
#         s_batch = s_stack_batch[:,update_step,:]
#         worst_sinr_batch = worst_sinr_function(constants, s_batch, G_M_batch, H_M_batch)
#         sinr_avg_db = 10*np.log10( np.sum(worst_sinr_batch)/batch_size )
#         print(f'Step {update_step:02d}, SINR = {sinr_avg_db:.4f} dB')
#         worst_sinr_stack_list[:,update_step] = worst_sinr_batch
        
#         f_stack_list[:,update_step] = sum_of_reciprocal(constants, G_M_batch, H_M_batch, s_batch)
#     # sinr_avg_db_opt = sinr_avg_db
    
#     return worst_sinr_stack_list, f_stack_list
            
