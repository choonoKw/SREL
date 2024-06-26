# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 22:32:19 2024

@author: jbk5816
"""

import torch
import numpy as np

from utils.load_scalars_from_setup import load_param_from_setup
from utils.training_dataset import TrainingDataSet
# from torch.utils.data import DataLoader

from sred.functions import sum_of_sinr_reciprocal, sinr_values, derive_s, derive_w

from sred.function_of_s import make_Gamma_M, make_Psi_M, make_Sigma, make_Sigma_opt
from sred.function_of_w import make_Phi_M, make_Theta_M

import time
from utils.format_time import format_time


def test(constants, model_test, eps_f):
    device = model_test.device
    
    struct_c, struct_m, struct_k, aqaqh, aqhaq, bqbqh, bqhbq, Upsilon = load_param_from_setup(
        'data/data_setup.mat', device)
    
    dataset = TrainingDataSet('data/data_trd_1e+02_val.mat')
    N_data = len(dataset)
    M = constants['M']
    # modulus = constants['modulus']
    
    # val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    y_M = dataset.y_M
    
    N_iter = 10
    
    f_sinr_stack_list = np.zeros((N_data, N_iter+1))    
    
    # # Ly = dataset.Ly.to(device)
    # with torch.no_grad():  # Disable gradient computation
        # for idx_data, (
        #         phi_batch, w_M_batch, G_M_batch, H_M_batch
        #         ) in enumerate(val_loader):
        #     # phi = phi_batch.squeeze()
            
        #     for idx_iter in range(N_iter):
        #         s_batch = modulus*torch.exp(1j *phi_batch) 
        #         S_tilde = np.reshape(np.concatenate(
        #             (s.reshape(-1, 1), np.zeros((Nt * (lm[M-1] - lm[0]), 1)))
        #             ), (Nt, Lj))
                
        #         f_sinr = sum_of_sinr_reciprocal(G_M_batch, H_M_batch, s_batch)
        #         phi_batch = model_test(
        #             phi_batch, w_M_batch, y_M, G_M_batch, H_M_batch
        #             )
                
                
                
        #         f_sinr_stack_list[idx_data,idx_iter] = f_sinr
    with torch.no_grad():            
        G_M_list = dataset.G_M_list.to(device)
        H_M_list = dataset.H_M_list.to(device)
        phi_list = dataset.phi_list.to(device)
        w_M_list = dataset.w_M_list.to(device)
        y_M = dataset.y_M.to(device)
        
        for idx_data, (G_M, H_M, phi, w_M) in enumerate(zip(
                torch.unbind(G_M_list, dim=-1),torch.unbind(H_M_list, dim=-1),
                torch.unbind(phi_list, dim=-1),torch.unbind(w_M_list, dim=-1)
                )): 
            
            s, S_tilde = derive_s(constants, phi, struct_c, struct_m)
            f_sinr = sum_of_sinr_reciprocal(G_M, H_M, s)
            f_sinr_stack_list[idx_data,0] = f_sinr
            
            sinr_db_M = 10*torch.log10(sinr_values(G_M, H_M, s))
            
            for m in range(M):
                print(f'sinr_{m+1:d} = {sinr_db_M[m].item():.2f}',end=', ')
                
            print(' ')
            
            ##### test
            # Sigma = make_Sigma(struct_c,struct_k,S_tilde,bqhbq)
            
            # Gamma_M = make_Gamma_M(struct_c,struct_m,S_tilde,aqhaq)
            
            # Psi_M = make_Psi_M(struct_c,struct_m,S_tilde,aqhaq,Sigma,Upsilon)
            
            # w_M, W_M_tilde = derive_w(struct_c ,Psi_M, Gamma_M, device)
            
            # G_M = make_Phi_M(struct_c,struct_m,struct_k,w_M,W_M_tilde,aqaqh,bqbqh,Upsilon)
            # H_M = make_Theta_M(struct_c,struct_m,W_M_tilde,aqaqh)
            ####
            start_time_epoch = time.time()
            for idx_iter in range(N_iter):
                # s = modulus*torch.exp(1j *phi) 
                
                # we need to make new functions so that we do not need to squeeze/unsqueeze
                # G_M_batch = G_M.unsqueeze(0)
                # H_M_batch = H_M.unsqueeze(0)
                # s_batch = s.unsqueeze(0)
                # phi_batch = phi.unsqueeze(0)
                # w_M_batch = w_M.unsqueeze(0)
                
                # f_sinr = sum_of_sinr_reciprocal(G_M_batch, H_M_batch, s_batch)
                
                phi = model_test(phi, w_M, y_M, G_M, H_M)
                
                # record values
                s, S_tilde = derive_s(constants, phi, struct_c, struct_m)
                f_sinr = sum_of_sinr_reciprocal(G_M, H_M, s).item()
                f_sinr_stack_list[idx_data,idx_iter+1] = f_sinr
                f_sinr_db = 10*np.log10(f_sinr)
                
                sinr_db_M = 10*torch.log10(sinr_values(G_M, H_M, s))
                
                
                print(f'idx_iter={idx_iter}, '
                      f'f_sinr = {f_sinr_db:.2f}')
                for m in range(M):
                    print(f'sinr_{m+1:d} = {sinr_db_M[m].item():.2f}',end=', ')
                    
                print(' ')
                
                
                if abs(
                        f_sinr_stack_list[idx_data,idx_iter+1]
                        -f_sinr_stack_list[idx_data,idx_iter])<eps_f:
                    break
                
                Sigma = make_Sigma_opt(struct_c,struct_k,S_tilde,bqhbq)
                
                Gamma_M = make_Gamma_M(struct_c,struct_m,S_tilde,aqhaq)
                
                Psi_M = make_Psi_M(struct_c,struct_m,S_tilde,aqhaq,Sigma,Upsilon)
                
                w_M, W_M_tilde = derive_w(struct_c ,Psi_M, Gamma_M, device)
                
                G_M = make_Phi_M(struct_c,struct_m,struct_k,w_M,W_M_tilde,aqaqh,bqbqh,Upsilon)
                H_M = make_Theta_M(struct_c,struct_m,W_M_tilde,aqaqh)
                
                sinr_db_M = 10*torch.log10(sinr_values(G_M, H_M, s))
                
                print('w_M is updated')
                for m in range(M):
                    print(f'sinr_{m+1:d} = {sinr_db_M[m].item():.2f}',end=', ')
                print(' ')
                
                # print(f'idx_iter={idx_iter}, '
                #       f'f_sinr = {f_sinr_db:.2f}')
                
                time_spent_epoch = time.time() - start_time_epoch  # Time spent in the current inner loop iteration
                time_left = time_spent_epoch * (N_iter - idx_iter - 1)  # Estimate the time left
                formatted_time_left = format_time(time_left)
                print(f"{formatted_time_left} left")
                
            
                
    return f_sinr_stack_list