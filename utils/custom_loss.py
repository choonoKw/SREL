# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:56:07 2024

@author: jbk5816
"""

import torch

def sum_of_reciprocal(constants, G_M_batch, H_M_batch, s_batch):
    batch_size = s_batch.size(0)
    M = constants['M']
    total_loss = 0.0

    # Process each item in the batch
    for batch_idx in range(batch_size):
        s_optimal = s_batch[batch_idx]
        G_M = G_M_batch[batch_idx]
        H_M = H_M_batch[batch_idx]
        
        f = 0.0
        for m in range(M):
            numerator = torch.abs(torch.vdot(s_optimal, torch.matmul(G_M[:, :, m], s_optimal)))
            denominator = torch.abs(torch.vdot(s_optimal, torch.matmul(H_M[:, :, m], s_optimal)))
            f += numerator / denominator
        
        # Accumulate loss for each batch item
        total_loss += f 

    # Average the loss over the batch
    return total_loss / batch_size

def regularizer_eta(constants, G_M_batch, H_M_batch, s_batch, eta_M_batch):
    batch_size = s_batch.size(0)
    M = constants['M']
    total_loss = 0.0

    # Process each item in the batch
    for batch_idx in range(batch_size):
        s = s_batch[batch_idx]
        G_M = G_M_batch[batch_idx]
        H_M = H_M_batch[batch_idx]
        eta_M = eta_M_batch[batch_idx]
        
        f = 0.0
        for m in range(M):
            G = G_M[:, :, m]
            H = H_M[:, :, m]
            eta = eta_M[:,m]
            
            # Using torch.vdot for complex dot products where one operand is conjugated
            sGs = torch.vdot(s, torch.matmul(G, s))  # Equivalent to s'*G*s in MATLAB
            sHs = torch.vdot(s, torch.matmul(H, s))  # Equivalent to s'*H*s in MATLAB
            Gs = torch.matmul(G, s)  # G*s
            Hs = torch.matmul(H, s)  # H*s
            
            # Calculating eta_tilde using torch.vdot
            eta_tilde = 2 / (sGs ** 2) * torch.imag((sHs * Gs - sGs * Hs) * torch.conj(s))
            f += torch.norm(eta_tilde - eta) ** 2
            
        # Accumulate loss for each batch item
        total_loss += f 
        
    # Average the loss over the batch
    return total_loss / batch_size

def custom_loss_function(constants, G_M_batch, H_M_batch, hyperparameters, model_outputs):
    N_step = constants['N_step']
    s_list_batch = model_outputs['s_list_batch']
    eta_M_list_batch = model_outputs['eta_M_list_batch']
    
    s_batch = s_list_batch[:,:,0]
    eta_M_batch = eta_M_list_batch[:,:,:,0]
    
    f_sinr = 0.0
    f_eta = regularizer_eta(constants, G_M_batch, H_M_batch, s_batch, eta_M_batch)
    
    for n in range(N_step-1):
        s_batch = s_list_batch[:,:,n+1].squeeze()
        eta_M_batch = eta_M_list_batch[:,:,:,n+1].squeeze()
        
        f_eta += regularizer_eta(constants, G_M_batch, H_M_batch, s_batch, eta_M_batch)
        
        f_sinr += sum_of_reciprocal(constants, G_M_batch, H_M_batch, s_batch)
        
    s_batch = s_list_batch[:,:,-1]
    f_sinr_opt = sum_of_reciprocal(constants, G_M_batch, H_M_batch, s_batch)
            
    # Compute the regularization loss
    # regularization_loss = regularizer_eta(constants, s_batch, G_M_batch, H_M_batch, eta_M_batch)
    # Combine the losses
    
    loss = f_sinr_opt + hyperparameters['lambda_sinr']*f_sinr/(N_step-1) + hyperparameters['lambda_eta']*f_eta/N_step
    
    return loss