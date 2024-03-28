# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:36:45 2024

@author: jbk5816
"""

import torch
import numpy as np

def reciprocal_sinr(G_batch, H_batch, s_batch):
    batch_size = s_batch.size(0)
    
    f_sinr_batch = np.zeros(batch_size)
    for idx_batch in range(batch_size):
        G = G_batch[idx_batch].squeeze()
        H = H_batch[idx_batch].squeeze()
        s = s_batch[idx_batch].squeeze()
        
        numerator = torch.abs(torch.vdot(s, torch.matmul(G, s)))
        denominator = torch.abs(torch.vdot(s, torch.matmul(H, s)))
    
        f_sinr_batch[idx_batch] = numerator / denominator
        
    return f_sinr_batch

def sinr(G_batch, H_batch, s_batch):
    batch_size = s_batch.size(0)
    
    sinr_batch = np.zeros(batch_size)
    for idx_batch in range(batch_size):
        G = G_batch[idx_batch].squeeze()
        H = H_batch[idx_batch].squeeze()
        s = s_batch[idx_batch].squeeze()
        
        numerator = torch.abs(torch.vdot(s, torch.matmul(H, s)))
        denominator = torch.abs(torch.vdot(s, torch.matmul(G, s)))
    
        sinr_batch[idx_batch] = numerator / denominator
        
    return sinr_batch


def sum_of_reciprocal(constants, G_M_batch, H_M_batch, s_batch):
    batch_size = s_batch.size(0)
    f_sr_batch = np.zeros(batch_size)
    for m in range(constants['M']):
        G_batch = G_M_batch[:,:,:,m]
        H_batch = H_M_batch[:,:,:,m]
        f_sr_batch+= reciprocal_sinr(G_batch, H_batch, s_batch)
        
    return f_sr_batch


        

def regularizer_eta(G, H, s, eta):
    # Using torch.vdot for complex dot products where one operand is conjugated
    sGs = torch.abs(torch.vdot(s, torch.matmul(G, s)))  # Equivalent to s'*G*s in MATLAB
    sHs = torch.abs(torch.vdot(s, torch.matmul(H, s)))  # Equivalent to s'*H*s in MATLAB
    Gs = torch.matmul(G, s)  # G*s
    Hs = torch.matmul(H, s)  # H*s
        
    # Calculating eta_tilde using torch.vdot
    eta_tilde = 2 / (sHs ** 2) * torch.imag((sHs * Gs - sGs * Hs) * torch.conj(s))
        
    # Average the loss over the batch
    return torch.norm(eta_tilde - eta) ** 2