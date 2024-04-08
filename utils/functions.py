# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:02:19 2024

@author: jbk5816

functions for SREL Algorithm

1) eta_sred(): compute the descent direction as the same as SRED
"""

import torch

def eta_sred(G_batch, H_batch, s_batch):
    s_batch_unsqueezed = s_batch.unsqueeze(-1)
    
    
    # I suspect this part might not be correct
    Gs_batch = torch.bmm(G_batch, s_batch_unsqueezed).squeeze()
    Hs_batch = torch.bmm(H_batch, s_batch_unsqueezed).squeeze()
    
    sGs_batch = torch.abs(torch.sum(torch.conj(s_batch) * Gs_batch, dim=1)).unsqueeze(-1)
    sHs_batch = torch.abs(torch.sum(torch.conj(s_batch) * Hs_batch, dim=1)).unsqueeze(-1)
    
    eta_batch = 2 / (sHs_batch ** 2)* torch.imag(
        (sHs_batch * Gs_batch - sGs_batch * Hs_batch) * torch.conj(s_batch)
        )
    
    return eta_batch

def sum_of_sinr_reciprocal(G_M_batch, H_M_batch, s_batch):
    s_batch_unsqueezed = s_batch.unsqueeze(-1)
    
    f_sinr = 0.0
    for m, (G_batch, H_batch) in enumerate(zip(
            torch.unbind(G_M_batch, dim=-1),torch.unbind(H_M_batch, dim=-1)
            )): 
        
        Gs_batch = torch.bmm(G_batch, s_batch_unsqueezed).squeeze()
        Hs_batch = torch.bmm(H_batch, s_batch_unsqueezed).squeeze()
        
        sGs_batch = torch.abs(torch.sum(torch.conj(s_batch) * Gs_batch, dim=1)).unsqueeze(-1)
        sHs_batch = torch.abs(torch.sum(torch.conj(s_batch) * Hs_batch, dim=1)).unsqueeze(-1)
         
        f_sinr += sGs_batch / sHs_batch
     
    return f_sinr