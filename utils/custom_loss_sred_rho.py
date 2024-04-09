# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:51:20 2024

@author: jbk5816
"""


import torch
# from utils.custom_loss_intra import reciprocal_sinr
from utils.custom_loss_batch import reciprocal_sinr

# def reciprocal_sinr(G,H,s):
#     numerator = torch.abs(torch.vdot(s, torch.matmul(G, s)))
#     denominator = torch.abs(torch.vdot(s, torch.matmul(H, s)))
#     return numerator / denominator
def reciprocal_sinr_M(G_M_batch, H_M_batch, s_batch):
    batch_size = s_batch.size(0)
    device = s_batch.device
    
    f_sinr_batch = torch.zeros(batch_size).to(device)
    for m, (G_batch, H_batch) in enumerate(zip(torch.unbind(G_M_batch, dim=3),
                                               torch.unbind(H_M_batch, dim=3))):
        f_sinr_batch += reciprocal_sinr(G_batch, H_batch, s_batch)
        
    return f_sinr_batch
        

def custom_loss_sred_mono(constants, G_M_batch, H_M_batch, hyperparameters, model_outputs):
    N_step = constants['N_step']
    s_stack_batch = model_outputs['s_stack_batch']
    rho_M_stack_batch = model_outputs['rho_M_stack_batch']
    batch_size = s_stack_batch.size(0)
    M = rho_M_stack_batch.size(-1)
    
    f_sinr_sum = 0.0
    
    s_batch =  s_stack_batch[:,0,:]
    f_sinr_t1_batch = reciprocal_sinr_M(G_M_batch, H_M_batch, s_batch)
    
    N_vl_sum = 0 # number of violation of monotonicity
    for update_step in range(N_step):
        s_batch =  s_stack_batch[:,update_step+1,:]
            
        f_sinr_t2_batch = reciprocal_sinr_M(G_M_batch, H_M_batch, s_batch)
        
        f_sinr_sum += torch.sum(
            torch.exp(
                hyperparameters['lambda_mono']*(f_sinr_t2_batch-f_sinr_t1_batch)
            )
        )
        
        N_vl_sum += torch.sum(
            (f_sinr_t2_batch - f_sinr_t1_batch > 0).int()
            )
        
    # s_batch =  s_stack_batch[:,-1,:]
    
    f_sinr_opt = torch.sum(reciprocal_sinr_M(G_M_batch, H_M_batch, s_batch))
    
    
    # sinr_opt_avg = torch.sum(1/f_sinr_opt_batch)/batch_size
    
    var_rho_avg = torch.sum(
        torch.var(rho_M_stack_batch, dim=0, unbiased=False)
        )/M
    
    loss = (
        f_sinr_opt
        + hyperparameters['lambda_sinr']*f_sinr_sum/(N_step-1)
        + hyperparameters['lambda_var_rho']*var_rho_avg
        )
    
    loss_avg = loss / batch_size 
    
    N_vl_avg = N_vl_sum / batch_size
    
    return loss_avg, N_vl_avg


def custom_loss_sred(constants, G_M_batch, H_M_batch, hyperparameters, model_outputs):
    N_step = constants['N_step']
    s_stack_batch = model_outputs['s_stack_batch']
    rho_M_stack_batch = model_outputs['rho_M_stack_batch']
    batch_size = s_stack_batch.size(0)
    M = rho_M_stack_batch.size(-1)
    
    f_sinr_sum = 0.0
    
    
    for update_step in range(N_step-1):
        s_batch =  s_stack_batch[:,update_step+1,:]
            
        f_sinr_sum += torch.sum(reciprocal_sinr_M(G_M_batch, H_M_batch, s_batch))
        
        
    s_batch =  s_stack_batch[:,-1,:]
    
    # f_sinr_opt_batch = reciprocal_sinr(G_batch, H_batch, s_batch)
    f_sinr_opt = torch.sum(reciprocal_sinr_M(G_M_batch, H_M_batch, s_batch))
    
    
    # sinr_opt_avg = torch.sum(1/f_sinr_opt_batch)/batch_size
    
    var_rho_avg = torch.sum(
        torch.var(rho_M_stack_batch, dim=0, unbiased=False)
        )/M
    
    loss = (
        f_sinr_opt
        + hyperparameters['lambda_sinr']*f_sinr_sum/(N_step-1)
        + hyperparameters['lambda_var_rho']*var_rho_avg
        )
    
    loss_avg = loss / batch_size 
    
    return loss_avg

def custom_loss_function_element(constants, G_M_batch, H_M_batch, hyperparameters, model_outputs):
    N_step = constants['N_step']
    s_stack_batch = model_outputs['s_stack_batch']
    # rho_M_stack_batch = model_outputs['rho_M_stack_batch']
    batch_size = s_stack_batch.size(0)
    
    loss_sum = 0.0
    
    for idx_batch in range(batch_size):
        G_M = G_M_batch[idx_batch]
        H_M = H_M_batch[idx_batch]
        
        s_stack = s_stack_batch[idx_batch]
        
        f_sinr = 0.0
    
        for n in range(N_step-1):
            s = s_stack[n+1]
            
            for m, (G, H) in enumerate(zip(torch.unbind(G_M, dim=2),
                                                    torch.unbind(H_M, dim=2))):
            
                f_sinr += reciprocal_sinr(G, H, s)
        
        # compute the f_sinr with the output s
        s = s_stack[-1]
        f_sinr_opt = 0.0
        for m, (G, H) in enumerate(zip(torch.unbind(G_M, dim=2),
                                                torch.unbind(H_M, dim=2))):
            f_sinr_opt += reciprocal_sinr(G, H, s)
            
            
        # # compute the variance of rho
        # rho_M_stack = rho_M_stack_batch[idx_batch]
        # var_rho_avg = torch.sum(torch.var(rho_M_stack, dim=0, unbiased=False))
    
        loss = f_sinr_opt + hyperparameters['lambda_sinr']*f_sinr/(N_step-1)
        # + hyperparameters['lambda_var_rho']*var_rho_avg
        
            
        loss_sum += loss
    
    return loss_sum/ batch_size