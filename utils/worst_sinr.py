# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:09:40 2024

@author: jbk5816
"""

import torch

def worst_sinr_function(constants, s_optimal_batch, G_M_batch, H_M_batch):
    batch_size = s_optimal_batch.size(0)
    M = constants['M']
    total_loss = 0.0

    # Process each item in the batch
    for i in range(batch_size):
        s_optimal = s_optimal_batch[i]
        G_M = G_M_batch[i]
        H_M = H_M_batch[i]
        
        sinr = torch.zeros((M,1))
        for m in range(M):
            numerator = torch.abs(torch.vdot(s_optimal, torch.matmul(H_M[:, :, m], s_optimal)))
            denominator = torch.abs(torch.vdot(s_optimal, torch.matmul(G_M[:, :, m], s_optimal)))
            sinr[m,:] = numerator / denominator
        
        # Accumulate loss for each batch item
        worst_sinr = torch.min(sinr)

    # Average the loss over the batch
    return worst_sinr / batch_size
    # return 10*torch.log10(worst_sinr / batch_size)
