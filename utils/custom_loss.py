# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:56:07 2024

@author: jbk5816
"""

import torch

def sum_of_reciprocal(constants, s_optimal_batch, G_M_batch, H_M_batch):
    batch_size = s_optimal_batch.size(0)
    M = constants['M']
    total_loss = 0.0

    # Process each item in the batch
    for i in range(batch_size):
        s_optimal = s_optimal_batch[i]
        G_M = G_M_batch[i]
        H_M = H_M_batch[i]
        
        f = 0.0
        for m in range(M):
            numerator = torch.abs(torch.vdot(s_optimal, torch.matmul(G_M[:, :, m], s_optimal)))
            denominator = torch.abs(torch.vdot(s_optimal, torch.matmul(H_M[:, :, m], s_optimal)))
            f += numerator / denominator
        
        # Accumulate loss for each batch item
        total_loss += f 

    # Average the loss over the batch
    return total_loss / batch_size

def regularizer_eta(constants, s_batch, G_M_batch, H_M_batch, eta_M_batch):
    batch_size = s_batch.size(0)
    M = constants['M']
    total_loss = 0.0

    # Process each item in the batch
    for i in range(batch_size):
        s = s_batch[i]
        G_M = G_M_batch[i]
        H_M = H_M_batch[i]
        eta_M = eta_M_batch[i]
        
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

def custom_loss_function(constants, G_M_batch, H_M_batch, lambda_eta, model_outputs):
    primary_loss = sum_of_reciprocal(constants, s_batch, G_M_batch, H_M_batch)
    # Compute the regularization loss
    regularization_loss = regularizer_eta(constants, s_batch, G_M_batch, H_M_batch, eta_M_batch)
    # Combine the losses
    loss = primary_loss + lambda_eta * regularization_loss
    
    return loss