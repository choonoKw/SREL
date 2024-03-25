# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:49:41 2024

@author: jbk5816
"""

import torch
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from utils.complex_valued_dataset import ComplexValuedDataset
from utils.training_dataset import TrainingDataSet
from utils.load_scalars_from_setup import load_scalars_from_setup
from utils.load_mapVector import load_mapVector

# from model.sred_rho import SRED_rho
# print('SRED_rho OG.')

# from model.sred_rho_DO import SRED_rho
# print('SRED_rho with Drop Out (DO)')

from model.sred import SRED_rep_rho
# print('SRED_rho with Batch Normalization (BN)')



from utils.custom_loss_sred_rho import custom_loss_function
from utils.worst_sinr import worst_sinr_function

# from utils.worst_sinr import worst_sinr_function

from torch.utils.tensorboard import SummaryWriter #tensorboard
# tensorboard --logdir=runs/SREL --reload_interval 5
# tensorboard --logdir=runs/SREL_intra
from visualization.plotting import plot_losses # result plot

import datetime
import time
import os

# import torch.nn as nn

def main(batch_size):
    # Load dataset
    constants = load_scalars_from_setup('data/data_setup.mat')
    y_M, Ly = load_mapVector('data/data_mapV.mat')
    data_num = 2e3
    
    
    dataset = ComplexValuedDataset(f'data/data_trd_{data_num:.0e}.mat')
    y_M, Ly = load_mapVector('data/data_mapV.mat')
    
    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,  # 20% for validation
        random_state=42
    )
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    # batch_size = 10
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    
    # loading constant
    constants['Ly'] = 570
    Nt = constants['Nt']
    N = constants['N']
    constants['modulus'] = 1 / torch.sqrt(torch.tensor(Nt * N, dtype=torch.float))

    ###############################################################
    ## Control Panel
    ###############################################################
    # Initialize model
    N_step = 10
    constants['N_step'] = N_step
    model_sred = SRED_rep_rho(constants)
#    model_sred.apply(init_weights)
    num_epochs = 10
    # Initialize the optimizer
    learning_rate=1e-5
    print(f'learning_rate={learning_rate:.0e}')
    optimizer = optim.Adam(model_sred.parameters(), lr=learning_rate)
    
    # loss setting
    lambda_sinr = 1e-2
    lambda_var_rho = 1e1
    hyperparameters = {
        'lambda_sinr': lambda_sinr,
        'lambda_var_rho': lambda_var_rho
    }    
    ###############################################################
    # for results
    # Get the current time
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') 
    print(f'current time: {current_time}')
    
    # Create a unique directory name using the current time and the N_step value
    log_dir = (
        f'runs/SRED_rho/data{data_num}/{current_time}'
        f'_Nstep{constants["N_step"]:02d}_batch{batch_size:02d}'
        f'_lr_{learning_rate:.0e}'
    )
    writer = SummaryWriter(log_dir)
    
    dir_weight_save = f'weights/SRED_rho/data{data_num}/Nstep{N_step:02d}_{current_time}'
    os.makedirs(dir_weight_save, exist_ok=True)
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_sred.to(device)
    model_sred.device = device
    
    
    
    # List to store average loss per epoch
    training_losses = []
    validation_losses = []
    
    
    start_time = time.time()
    # Training loop
    for epoch in range(num_epochs):
        # case_num = epoch % num_case + 1
        # dataset = TrainingDataSet(f'data/data_trd_{data_num:.0e}_case{case_num:02d}.mat')
        
        # # Split dataset into training and validation
        # train_indices, val_indices = train_test_split(
        #     range(len(dataset)),
        #     test_size=0.2,  # 20% for validation
        #     random_state=42
        # )
        # train_dataset = Subset(dataset, train_indices)
        # val_dataset = Subset(dataset, val_indices)
        
        # # batch_size = 10
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        model_sred.train()  # Set model to training mode
        total_train_loss = 0.0
        
        
        for phi_batch, w_M_batch, G_M_batch, H_M_batch in train_loader:
            # if torch.cuda.is_available():
            phi_batch = phi_batch.to(device)
            G_M_batch = G_M_batch.to(device)
            H_M_batch = H_M_batch.to(device)
            w_M_batch = w_M_batch.to(device)
            y_M = y_M.to(device)  # If y_M is a tensor that requires to be on the GPU
            
            # Perform training steps
            optimizer.zero_grad()
            
            
            model_outputs = model_sred(phi_batch, w_M_batch, y_M, G_M_batch, H_M_batch)
            
            s_stack_batch = model_outputs['s_stack_batch']
            loss = custom_loss_function(
                constants, G_M_batch, H_M_batch, hyperparameters, model_outputs)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            
        # Compute average loss for the epoch
        average_train_loss = total_train_loss / len(train_loader) / model_sred.M
        
        # Log the loss
        writer.add_scalar('Loss/Training [dB]', 10*np.log10(average_train_loss), epoch)
        # writer.flush()
        
        training_losses.append(average_train_loss)
        
        
        
        
        # Validation phase
        model_sred.eval()  # Set model to evaluation mode
        
        total_val_loss = 0.0
        sum_of_worst_sinr_avg = 0.0  # Accumulate loss over all batches
        
        with torch.no_grad():  # Disable gradient computation
            for phi_batch, w_M_batch, G_M_batch, H_M_batch in test_loader:
                # s_batch = modulus * torch.exp(1j * phi_batch)
                phi_batch = phi_batch.to(device)
                G_M_batch = G_M_batch.to(device)
                H_M_batch = H_M_batch.to(device)
                w_M_batch = w_M_batch.to(device)
                y_M = y_M.to(device)  # If y_M is a tensor that requires to be on the GPU
                
                
                model_outputs = model_sred(phi_batch, w_M_batch, y_M, G_M_batch, H_M_batch)
                
                s_stack_batch = model_outputs['s_stack_batch']
                
                val_loss = custom_loss_function(
                    constants, G_M_batch, H_M_batch, hyperparameters, model_outputs)
                total_val_loss += val_loss.item()
                
                s_optimal_batch = s_stack_batch[:,-1,:].squeeze()
                
                sum_of_worst_sinr_avg += worst_sinr_function(constants, s_optimal_batch, G_M_batch, H_M_batch)
                    
        average_val_loss = total_val_loss / len(test_loader) / model_sred.M
        validation_losses.append(average_val_loss)
    
        # Log the loss
        writer.add_scalar('Loss/Testing [dB]', 10*np.log10(average_val_loss), epoch)
        writer.flush()
        
        worst_sinr_avg_db = 10*torch.log10(sum_of_worst_sinr_avg/ len(test_loader))  # Compute average loss for the epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              # f'Train Loss = {average_train_loss:.4f}, '
              f'average_worst_sinr = {worst_sinr_avg_db:.4f} dB')
        
        
        
        
    # End time
    end_time = time.time()
    
    # Calculate the duration
    duration = end_time - start_time
    
    print(f"Training completed in: {duration:.2f} seconds")
    
    # sinr values w.r.t. update steps
#     sum_of_worst_sinr_avg_list = torch.zeros(N_step).to(device)
#     for phi_batch, w_M_batch, G_M_batch, H_M_batch in test_loader:
#         # s_batch = modulus * torch.exp(1j * phi_batch)
#         phi_batch = phi_batch.to(device)
#         G_M_batch = G_M_batch.to(device)
#         H_M_batch = H_M_batch.to(device)
#         w_M_batch = w_M_batch.to(device)
#         y_M = y_M.to(device)  # If y_M is a tensor that requires to be on the GPU
        
        
#         model_outputs = model_sred(phi_batch, w_M_batch, y_M, G_M_batch, H_M_batch)
        
#         s_stack_batch = model_outputs['s_stack_batch']
        
#         val_loss = custom_loss_function(constants, G_M_batch, H_M_batch, hyperparameters, s_stack_batch)
#         total_val_loss += val_loss.item()
        
#         for idx_step in range (N_step):
#             s_optimal_batch = s_stack_batch[:,idx_step,:].squeeze()        
#             sum_of_worst_sinr_avg_list[idx_step] += worst_sinr_function(constants, s_optimal_batch, G_M_batch, H_M_batch)
            
#     worst_sinr_avg_list_dB = 10*torch.log10(sum_of_worst_sinr_avg_list/ len(test_loader))
#     print(' worst_sinr [dB]')
#     for idx_step in range(N_step):
#         print(f'Update step {idx_step:02d}, {worst_sinr_avg_list_dB[idx_step].item():.4f}')
    
        
    # After completing all epochs, plot the training loss
    
    plot_losses(training_losses, validation_losses)
    
    if data_num=='2e3':
        # save model's information
        save_dict = {
            'state_dict': model_sred.state_dict(),
            'N_step': model_sred.N_step,
            # Include any other attributes here
        }
        # save
        dir_weight_save = (
            f'weights/SRED_rho/{current_time}'
            f'_Nstep{N_step:02d}_batch{batch_size:02d}'
            f'_sinr_{worst_sinr_avg_db:.2f}dB'
        )
        os.makedirs(dir_weight_save, exist_ok=True)
        torch.save(save_dict, os.path.join(dir_weight_save, 'model_with_attrs.pth'))
    
    # validation of rho values
    rho_M_stack_batch = model_outputs['rho_M_stack_batch']
    rho_M_stack_avg = torch.sum(rho_M_stack_batch, dim=0)/batch_size
    print('rho values = ')
    for n in range(model_sred.N_step):
        for m in range(model_sred.M):
            print(f'{rho_M_stack_avg[n,m].item():.4f}', end=",      ")
        print('')
        
    # SINR values for each step
    for update_step in range(N_step+1):
        s_batch = s_stack_batch[:,update_step,:]
        sinr_db = 10*torch.log10(worst_sinr_function(constants, s_batch, G_M_batch, H_M_batch))
        print(f'Step {update_step:02d}, SINR = {sinr_db:.4f} dB')
    
    # # validate the values of rho
    # rho_stack_batch = model_outputs['rho_stack_batch']
    # rho_stack_avg = torch.sum(rho_stack_batch, dim=0)/batch_size
    # print('rho values = ')
    # for n in range(model_sred.N_step):
    #     print(f'{rho_stack_avg[n].item()}')
    # print(f'finished time: {current_time}')
    
if __name__ == "__main__":
    # batch_size = 10
    # # lambda_var_rho = 1e1
    # print(f'batch_size = {batch_size}')
    # main(batch_size)
    
    batch_size = 10
    # lambda_var_rho = 1e1
    print(f'batch_size = {batch_size}')
    main(batch_size)
    
    # batch_size = 50
    # # lambda_var_rho = 1e1
    # print(f'batch_size = {batch_size}')
    # main(batch_size)
