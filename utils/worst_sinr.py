# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:09:40 2024

@author: jbk5816
"""

import torch

def worst_sinr_function(constants, s_optimal_batch, G_M_batch, H_M_batch):
    batch_size = s_optimal_batch.size(0)
    M = constants['M']

    # Process each item in the batch
    sum_worst_sinr = 0.0
    for idx_batch in range(batch_size):
        s_optimal = s_optimal_batch[idx_batch]
        G_M = G_M_batch[idx_batch]
        H_M = H_M_batch[idx_batch]
        
        sinr = torch.zeros((M,1))
        for m in range(M):
            numerator = torch.abs(torch.vdot(s_optimal, torch.matmul(H_M[:, :, m], s_optimal)))
            denominator = torch.abs(torch.vdot(s_optimal, torch.matmul(G_M[:, :, m], s_optimal)))
            sinr[m,:] = numerator / denominator
        
        # Accumulate loss for each batch item
        sum_worst_sinr += torch.min(sinr).item()

    # Average the loss over the batch
    return sum_worst_sinr / batch_size
