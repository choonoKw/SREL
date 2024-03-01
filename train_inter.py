# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 14:06:24 2024

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
from model.srel_intra import SREL_intra
from model.srel_inter import SREL_inter

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

def main():
    torch.autograd.set_detect_anomaly(True)
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
    
    batch_size = 50
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    
    # loading constant
    constants['Ly'] = Ly
    Nt = constants['Nt']
    N = constants['N']
    constants['modulus'] = 1 / torch.sqrt(torch.tensor(Nt * N, dtype=torch.float))
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the bundled dictionary
    dir_dict = 'weights/SREL_intra/Nstep05_data1e2_20240229-012749'
    loaded_dict = torch.load(os.path.join(dir_dict,'model_with_attrs.pth'), map_location=device)
    N_step = loaded_dict['N_step']
    constants['N_step'] = N_step
    
    # Step 1: Instantiate model1
    model_intra = SREL_intra(constants)
    model_intra.load_state_dict(loaded_dict['state_dict'])                         
    
    # freeze model_intra
    for param in model_intra.parameters():
        param.requires_grad = False
    
    # Initialize model
    model_inter = SREL_inter(constants, model_intra)
    
    
    ###############################################################
    ## Control Panel
    ###############################################################
    num_epochs = 10
    # Initialize the optimizer
    learning_rate=1e-2
    print(f'learning_rate=1e{int(np.log10(learning_rate)):01d}')
    optimizer = optim.Adam(model_inter.parameters(), lr=learning_rate)
    
    # loss setting
    lambda_sinr = 1e-3
    hyperparameters = {
        'lambda_sinr': lambda_sinr,
    }    
    ###############################################################
    # for results
    # Get the current time
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')    
    
    # Create a unique directory name using the current time and the N_step value
    # log_dir = f'runs/SREL_inter/Nstep{constants["N_step"]:02d}_data{data_num}_{current_time}'
    dir_log = f'runs/SREL_inter/batch{batch_size:02d}_lr_1e{-int(np.log10(learning_rate)):01d}_sinr_1e{-int(np.log10(lambda_sinr)):01d}_{current_time}'
    writer = SummaryWriter(dir_log)
    
    dir_weight_save = f'weights/SREL_inter/Nstep{N_step:02d}_data{data_num}_{current_time}'
    #os.makedirs(dir_weight_save, exist_ok=True)
    
    model_intra.to(device)
    model_inter.to(device)
    model_intra.device = device
    model_inter.device = device
    
    # List to store average loss per epoch
    training_losses = []
    validation_losses = []
    
    
    start_time = time.time()
    # Training loop
    for epoch in range(num_epochs):
        ###############################################################
        # Training phase
        ###############################################################
        
        model_inter.train()  # Set model to training mode
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
            
            model_outputs = model_inter(phi_batch, w_M_batch, y_M)
            
            # print gradient to see gradient flows
            # for name, param in model_inter.est_mu_modules.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: Gradient norm: {param.grad.norm().item()}")
            #     else:
            #         print(f"{name}: No grad")
            
            
            # calculate loss            
            s_stack_batch = model_outputs['s_stack_batch']
            loss = custom_loss_function(constants, G_M_batch, H_M_batch, hyperparameters, s_stack_batch)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        # Compute average loss for the epoch
        average_train_loss = total_train_loss / len(train_loader) / model_inter.M
        
        # Log the loss
        writer.add_scalar('Loss/Training', average_train_loss, epoch)
        # writer.flush()
        
        training_losses.append(average_train_loss)
            
        ###############################################################
        # Validation phase
        ###############################################################
        model_inter.eval()  # Set model to evaluation mode
        
        total_val_loss = 0.0
        sum_of_worst_sinr_avg = 0.0  # Accumulate loss over all batches
        
        with torch.no_grad():  # Disable gradient computation            
            for phi_batch, w_M_batch, G_M_batch, H_M_batch in test_loader:
                # if torch.cuda.is_available():
                phi_batch = phi_batch.to(device)
                G_M_batch = G_M_batch.to(device)
                H_M_batch = H_M_batch.to(device)
                w_M_batch = w_M_batch.to(device)
                y_M = y_M.to(device)  # If y_M is a tensor that requires to be on the GPU
                
                # Perform training steps
                optimizer.zero_grad()
                
                model_outputs = model_inter(phi_batch, w_M_batch, y_M)
                
                s_stack_batch = model_outputs['s_stack_batch']
                
                val_loss = custom_loss_function(constants, G_M_batch, H_M_batch, hyperparameters, s_stack_batch)
                total_val_loss += val_loss.item()
                
                s_optimal_batch = s_stack_batch[:,-1,:].squeeze()
                
                sum_of_worst_sinr_avg += worst_sinr_function(constants, s_optimal_batch, G_M_batch, H_M_batch)
                
        average_val_loss = total_val_loss / len(test_loader) / model_inter.M
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
    # torch.save(model_inter.state_dict(), os.path.join(dir_weight_save, 'model_with_attrs.pth'))
    
if __name__ == "__main__":
    main()
