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
from utils.load_scalars_from_setup import load_scalars_from_setup
from utils.load_mapVector import load_mapVector
from model.sred_rho import SRED_rho

from utils.custom_loss_inter import custom_loss_function
from utils.worst_sinr import worst_sinr_function

# from utils.worst_sinr import worst_sinr_function

from torch.utils.tensorboard import SummaryWriter #tensorboard
# tensorboard --logdir=runs/SREL --reload_interval 5
# tensorboard --logdir=runs/SREL_intra
from visualization.plotting import plot_losses # result plot

import datetime
import time
import os

import torch.nn as nn

def main():
    # Load dataset
    constants = load_scalars_from_setup('data/data_setup.mat')
    y_M, Ly = load_mapVector('data/data_mapV.mat')
    data_num = '1e2'
    dataset = ComplexValuedDataset(f'data/data_trd_{data_num}.mat')
    
    
    # Split dataset into training and validation
    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,  # 20% for validation
        random_state=42
    )
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    batch_size = 10
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # loading constant
    constants['Ly'] = Ly
    Nt = constants['Nt']
    N = constants['N']
    constants['modulus'] = 1 / torch.sqrt(torch.tensor(Nt * N, dtype=torch.float))

    ###############################################################
    ## Control Panel
    ###############################################################
    # Initialize model
    N_step = 10
    constants['N_step'] = N_step
    model_sred_rho = SRED_rho(constants)
    #model_intra.apply(init_weights)
    num_epochs = 10
    # Initialize the optimizer
    learning_rate=1e-5
    print(f'learning_rate=1e{int(np.log10(learning_rate)):01d}')
    optimizer = optim.Adam(model_sred_rho.parameters(), lr=learning_rate)
    
    # loss setting
    lambda_eta = 1e-6
    lambda_sinr = 1e-2
    hyperparameters = {
        'lambda_eta': lambda_eta,
        'lambda_sinr': lambda_sinr,
    }    
    ###############################################################
    # for results
    # Get the current time
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')    
    
    # Create a unique directory name using the current time and the N_step value
    log_dir = f'runs/SRED_rho/data{data_num}/Nstep{N_step:02d}_batch{batch_size:02d}_lr_1e{-int(np.log10(learning_rate)):01d}_sinr_1e{-int(np.log10(lambda_sinr)):01d}_{current_time}'
    writer = SummaryWriter(log_dir)
    
    dir_weight_save = f'weights/SRED_rho/data{data_num}/Nstep{N_step:02d}_{current_time}'
    os.makedirs(dir_weight_save, exist_ok=True)
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print(f"Using device: {device}")
    model_sred_rho.to(device)
    model_sred_rho.device = device
    
    # List to store average loss per epoch
    training_losses = []
    validation_losses = []
    
    
    start_time = time.time()
    # Training loop
    for epoch in range(num_epochs):
        model_sred_rho.train()  # Set model to training mode
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
            
            
            model_outputs = model_sred_rho(phi_batch, w_M_batch, y_M, G_M_batch, H_M_batch)
            
            s_stack_batch = model_outputs['s_stack_batch']
            loss = custom_loss_function(constants, G_M_batch, H_M_batch, hyperparameters, s_stack_batch)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            
        # Compute average loss for the epoch
        average_train_loss = total_train_loss / len(train_loader) / model_sred_rho.M
        
        # Log the loss
        writer.add_scalar('Loss/Training', average_train_loss, epoch)
        # writer.flush()
        
        training_losses.append(average_train_loss)
        
        
        
        
        # Validation phase
        model_sred_rho.eval()  # Set model to evaluation mode
        
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
                
                # Perform training steps
                optimizer.zero_grad()
                
                model_outputs = model_sred_rho(phi_batch, w_M_batch, y_M, G_M_batch, H_M_batch)
                
                s_stack_batch = model_outputs['s_stack_batch']
                
                val_loss = custom_loss_function(constants, G_M_batch, H_M_batch, hyperparameters, s_stack_batch)
                total_val_loss += val_loss.item()
                
                s_optimal_batch = s_stack_batch[:,-1,:].squeeze()
                
                sum_of_worst_sinr_avg += worst_sinr_function(constants, s_optimal_batch, G_M_batch, H_M_batch)
                    
        average_val_loss = total_val_loss / len(test_loader) / model_sred_rho.M
        validation_losses.append(average_val_loss)
    
        # Log the loss
        writer.add_scalar('Loss/Testing', average_val_loss, epoch)
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
        
    # After completing all epochs, plot the training loss
    
    plot_losses(training_losses, validation_losses)
    
    
    # save model's information
    save_dict = {
        'state_dict': model_sred_rho.state_dict(),
        'N_step': model_sred_rho.N_step,
        # Include any other attributes here
    }
    
    torch.save(save_dict, os.path.join(dir_weight_save, 'model_with_attrs.pth'))
    
    rho_avg_stack_batch = model_outputs['rho_avg_stack_batch']
    rho_avg_stack_avg = torch.sum(rho_avg_stack_batch, dim=0)/batch_size
    
    print('rho values = ')
    for n in range(model_sred_rho.N_step):
        print(f'{rho_avg_stack_avg[n].item():.4f}')
    
if __name__ == "__main__":
    main()