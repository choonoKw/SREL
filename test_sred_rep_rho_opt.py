# -*- coding: utf-8 -*-
"""
Created on April 2 2024

@author: jbk5816
"""

import torch
# import torch.optim as optim
import numpy as np
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader, Subset

# from utils.complex_valued_dataset import ComplexValuedDataset
# from utils.training_dataset import TrainingDataSet
from utils.load_scalars_from_setup import load_scalars_from_setup
# from utils.load_mapVector import load_mapVector

# from model.sred_rho import SRED_rho
# print('SRED_rho OG.')

# from model.sred_rho_DO import SRED_rho
# print('SRED_rho with Drop Out (DO)')

from model.sred import SRED_rep_rho, SRED_vary_rho

# from model.sred import SRED_rep_rho, SRED_vary_rho

from utils.check_module_structure import is_single_nn

from model.srel_intra_infer import SREL_intra_phase1_infer

from sred.functions import sum_of_sinr_reciprocal, sinr_values, derive_s, derive_w

from sred.function_of_s import make_Gamma_M, make_Psi_M, make_Sigma
from sred.function_of_s import make_Gamma_M_opt, make_Psi_M_opt, make_Sigma_opt

from sred.function_of_w import make_Phi_M, make_Theta_M
from sred.function_of_w import make_Phi_M_opt, make_Theta_M_opt


# from utils.custom_loss_intra import custom_loss_intra_phase1
# from utils.worst_sinr import worst_sinr_function
from utils.load_scalars_from_setup import load_param_from_setup
from utils.training_dataset import TrainingDataSet
# from utils.worst_sinr import worst_sinr_function

# from torch.utils.tensorboard import SummaryWriter #tensorboard
# tensorboard --logdir=runs/SREL --reload_interval 5
# tensorboard --logdir=runs/SREL_intra
# from visualization.plotting import plot_losses # result plot

# import datetime
import time
from utils.format_time import format_time

import os
import argparse

# from utils.joint_design import test
# from utils.save_result_mat import save_result_mat

# from utils.format_time import format_time

# import torch.nn as nn

def main(weightdir):
    # Load dataset
    constants = load_scalars_from_setup('data/data_setup.mat')
    # y_M, Ly = load_mapVector('data/data_mapV.mat')
    # data_num = 1e1
    
    
    # loading constant
    constants['Ly'] = 570
    Nt = constants['Nt']
    # M = constants['M']
    N = constants['N']
    eps_f = 1e-7
    
    constants['modulus'] = 1 / torch.sqrt(torch.tensor(Nt * N, dtype=torch.float))
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ###############################################################
    ## Load weight
    ###############################################################
    # Load the bundled dictionary
    if weightdir:
        dir_dict_saved = weightdir
        loaded_dict = torch.load(os.path.join(dir_dict_saved,'model_with_attrs.pth'), 
                                 map_location=device)
    else:
        dir_dict_saved = (
            'weights/sred_rep_rho/'
            '20240408-154610_Nstep10_batch05_sinr_15.44dB')
        loaded_dict = torch.load(os.path.join(dir_dict_saved,'model_with_attrs.pth'), 
                                 map_location=device)
            
    state_dict = loaded_dict['state_dict']
    N_step = loaded_dict['N_step']
    
    ###############################################################
    ## Control Panel
    ###############################################################
    # Initialize model
    # N_step = 10
    constants['N_step'] = N_step
    
    if is_single_nn(state_dict,'est_rho_modules'):
        model_intra_phase1 = SRED_rep_rho(constants)
          
    else:
        model_intra_phase1 = SRED_vary_rho(constants)
        
    model_intra_phase1.load_state_dict(loaded_dict['state_dict']) 
        
    
    ###############################################################
    
    model_intra_phase1.to(device)
    model_intra_phase1.device = device
    # # for results

    
    # validation
    model_intra_phase1.eval()  # Set model to evaluation mode
    model_intra_tester = SREL_intra_phase1_infer(constants, model_intra_phase1)
    model_intra_tester.device = device

    
    # worst_sinr_stack_list, f_stack_list = test(constants,model_intra_tester,1e-7)
    
    (struct_c, struct_m, struct_k, aqaqh, aqhaq, bqbqh, bqhbq, Upsilon,
     AQAQH_M, AQHAQ_M, BQBQH_K, BQHBQ_K) = load_param_from_setup(
        'data/data_setup.mat', device)
    
    dataset = TrainingDataSet('data/data_trd_1e+02_val.mat')
    N_data = len(dataset)
    M = constants['M']
    # modulus = constants['modulus']
    
    # val_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    y_M = dataset.y_M
    
    N_iter = 1
    
    f_sinr_stack_list = np.zeros((N_data, N_iter+1))    
    
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
            f_sinr_db = 10*torch.log10(f_sinr)
            
            sinr_db_M = 10*torch.log10(sinr_values(G_M, H_M, s))
            
            
            
            ##### test
            # Sigma = make_Sigma(struct_c,struct_k,S_tilde,bqhbq)
            
            # Gamma_M = make_Gamma_M(struct_c,struct_m,S_tilde,aqhaq)
            
            # Psi_M = make_Psi_M(struct_c,struct_m,S_tilde,aqhaq,Sigma,Upsilon)
            
            # w_M, W_M_tilde = derive_w(struct_c ,Psi_M, Gamma_M, device)
            
            # G_M = make_Phi_M(struct_c,struct_m,struct_k,w_M,W_M_tilde,aqaqh,bqbqh,Upsilon)
            # H_M = make_Theta_M(struct_c,struct_m,W_M_tilde,aqaqh)
            ####
            start_time_iter = time.time()
            for idx_iter in range(N_iter):
                # s = modulus*torch.exp(1j *phi) 
                
                # we need to make new functions so that we do not need to squeeze/unsqueeze
                # G_M_batch = G_M.unsqueeze(0)
                # H_M_batch = H_M.unsqueeze(0)
                # s_batch = s.unsqueeze(0)
                # phi_batch = phi.unsqueeze(0)
                # w_M_batch = w_M.unsqueeze(0)
                
                # f_sinr = sum_of_sinr_reciprocal(G_M_batch, H_M_batch, s_batch)
                
                phi = model_intra_tester(phi, w_M, y_M, G_M, H_M)
                
                # record values
                s, S_tilde = derive_s(constants, phi, struct_c, struct_m)
                f_sinr = sum_of_sinr_reciprocal(G_M, H_M, s).item()
                f_sinr_stack_list[idx_data,idx_iter+1] = f_sinr
                f_sinr_db = 10*np.log10(f_sinr)
                
                sinr_db_M = 10*torch.log10(sinr_values(G_M, H_M, s))
                
                
                #print(f'idx_iter={idx_iter}, '
                #      f'f_sinr = {f_sinr_db:.2f}')
                #for m in range(M):
                #    print(f'sinr_{m+1:d} = {sinr_db_M[m].item():.2f}',end=', ')
                    
                
                
                
                # if abs(
                #         f_sinr_stack_list[idx_data,idx_iter+1]
                #         -f_sinr_stack_list[idx_data,idx_iter])<eps_f:
                #     break
                
                Sigma = make_Sigma_opt(struct_c,struct_k,S_tilde,BQHBQ_K)
                
                Gamma_M = make_Gamma_M_opt(struct_c,struct_m,S_tilde,AQHAQ_M)
                
                Psi_M = make_Psi_M_opt(struct_c,struct_m,S_tilde,AQHAQ_M,Sigma,Upsilon)
                
                w_M, W_M_tilde = derive_w(struct_c ,Psi_M, Gamma_M, device)
                
                start_time = time.time()
                G_M = make_Phi_M(struct_c,struct_m,struct_k,w_M,W_M_tilde,aqaqh,bqbqh,Upsilon)
                time_spent1 = time.time() - start_time
                
                start_time = time.time()
                G_M2 = make_Phi_M_opt(struct_c,struct_m,struct_k,w_M,W_M_tilde,AQAQH_M,BQBQH_K,Upsilon)
                time_spent2 = time.time() - start_time
                
                gap = torch.max(torch.abs(G_M-G_M2))
                mean = torch.mean(torch.abs(G_M))
                
                print(f'G_M, gap = {gap} with mean {mean}')
                
                print(f'G_M took {time_spent1} seconds, '
                      f'G_M2 took {time_spent2} seconds, ')
                
                start_time = time.time()
                H_M = make_Theta_M(struct_c,struct_m,W_M_tilde,aqaqh)
                time_spent1 = time.time() - start_time
                
                start_time = time.time()
                H_M2 = make_Theta_M_opt(struct_c, struct_m, W_M_tilde, AQAQH_M)
                time_spent2 = time.time() - start_time
                
                gap = torch.max(torch.abs(H_M-H_M2))
                mean = torch.mean(torch.abs(H_M))
                
                print(f'H_M, gap = {gap} with mean {mean}')
                
                print(f'H_M took {time_spent1} seconds, '
                      f'H_M2 took {time_spent2} seconds, ')
                
                sinr_db_M = 10*torch.log10(sinr_values(G_M, H_M, s))
                
                # print('w_M is updated')
                print(f'idx_iter={idx_iter}, '
                      f'f_sinr = {f_sinr_db:.2f}')
                for m in range(M):
                    print(f'sinr_{m+1:d} = {sinr_db_M[m].item():.2f}',end=', ')
                print(' ')
                
                # print(f'idx_iter={idx_iter}, '
                #       f'f_sinr = {f_sinr_db:.2f}')
                
                time_spent_epoch = time.time() - start_time_iter  # Time spent in the current inner loop iteration
                time_left = time_spent_epoch * (N_iter - idx_iter - 1)  # Estimate the time left
                formatted_time_left = format_time(time_left)
                print(f"{formatted_time_left} left")
                
        time_spent_total_form = format_time(time.time() - start_time_iter)
        
        print(f"Computation time: {time_spent_total_form}")
        print('-----------------------------------------------')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a model.")
    
    parser.add_argument("--weightdir", type=str, 
                        help="Save the model weights after training")
    # parser.add_argument("--save-mat", action="store_true",
    #                     help="Save mat file including worst-sinr values")
    
    args = parser.parse_args()
    
    main(weightdir=args.weightdir)
    
