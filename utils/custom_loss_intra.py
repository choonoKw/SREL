# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 16:10:00 2024

@author: jbk5816
"""

import torch

from utils.custom_loss import reciprocal_sinr, regularizer_eta


def custom_loss_function(constants, G_batch, H_batch, hyperparameters, model_outputs):
    N_step = constants['N_step']
    s_stack_batch = model_outputs['s_stack_batch']
    s_stack_batch = s_stack_batch.to(G_batch.device)
    eta_stack_batch = model_outputs['eta_stack_batch']
    eta_stack_batch = eta_stack_batch.to (G_batch.device)
    batch_size = s_stack_batch.size(0)
    
    loss_sum = 0.0
    
    for idx_batch in range(batch_size):
        G = G_batch[idx_batch]
        H = H_batch[idx_batch]
        
        s_stack = s_stack_batch[idx_batch]
        eta_stack = eta_stack_batch[idx_batch]
        
        
        s = s_stack[0]
        eta = eta_stack[0]
    
        f_eta = regularizer_eta(G, H, s, eta)
        f_sinr = 0.0
    
        for n in range(N_step-1):
            s = s_stack[n+1]
            eta = eta_stack[n+1]
            
            f_eta += regularizer_eta(G, H, s, eta)
            f_sinr += reciprocal_sinr(G, H, s)
        
        s = s_stack[N_step]
        f_sinr_opt = reciprocal_sinr(G, H, s)
    
        loss = f_sinr_opt + \
            hyperparameters['lambda_sinr']*f_sinr/(N_step-1) + hyperparameters['lambda_eta']*f_eta/N_step
            
        loss_sum += loss
    
    return loss_sum/ batch_size

# def sinr_function(constants, G_batch, H_batch, s_batch):
#     batch_size = s_batch.size(0)
    
#     sinr_batch = torch.empty(batch_size)
#     for idx_batch in range(batch_size):
#         G = G_batch[idx_batch]
#         H = H_batch[idx_batch]
        
#         s = s_batch[idx_batch]
        
#         numerator = torch.abs(torch.vdot(s, torch.matmul(H, s)))
#         denominator = torch.abs(torch.vdot(s, torch.matmul(G, s)))
#         sinr_batch[idx_batch] = numerator / denominator

#         # Average the loss over the batch
        
#     return sinr_batch