# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 21:40:10 2024

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
from model.srel_intra_rep import SREL_intra_rep

from utils.custom_loss_intra import custom_loss_function, sinr_function

# from utils.worst_sinr import worst_sinr_function
from model.srel_intra_rep_tester import SREL_intra_rep_tester
from utils.worst_sinr import worst_sinr_function

from torch.utils.tensorboard import SummaryWriter #tensorboard
# tensorboard --logdir=runs/SREL --reload_interval 5
# tensorboard --logdir=runs/SREL_intra_rep
from visualization.plotting import plot_losses # result plot

import datetime
import time
import os

import torch.nn as nn

def main(N_step):
    # Load dataset
    constants = load_scalars_from_setup('data/data_setup.mat')
    y_M, Ly = load_mapVector('data/data_mapV.mat')
    data_num = '1e1'
    dataset = ComplexValuedDataset(f'data/data_trd_{data_num}.mat')
    
    
    # Split dataset into training and validation
    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,  # 20% for validation
        random_state=42
    )
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # loading constant
    constants['Ly'] = Ly
    Nt = constants['Nt']
    N = constants['N']
    constants['modulus'] = 1 / torch.sqrt(torch.tensor(Nt * N, dtype=torch.float))

    ###############################################################
    ## Control Panel
    ###############################################################
    # Initialize model
    # N_step = 5
    constants['N_step'] = N_step
    model_intra = SREL_intra_rep(constants)
    # model_intra.apply(init_weights)
    num_epochs = 10
    # Initialize the optimizer
    learning_rate = 1e-5
    print(f'learning_rate=1e{int(np.log10(learning_rate)):01d}')
    optimizer = optim.Adam(model_intra.parameters(), lr=learning_rate)
    
    # loss setting
    hyperparameters = {
        'lambda_eta': 1e-5,
        'lambda_sinr': 1e-2,
    }  
    ###############################################################
    # for results
    # Get the current time
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')    
    
    # Create a unique directory name using the current time and the N_step value
    log_dir = f'runs/SREL_intra_rep/data{data_num}/{current_time}_Nstep{constants["N_step"]:02d}_lr_1e{-int(np.log10(learning_rate)):01d}_batch{batch_size:02d}'
    writer = SummaryWriter(log_dir)
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print(f"Using device: {device}")
    model_intra.to(device)
    model_intra.device = device
    
    # List to store average loss per epoch
    training_losses = []
    validation_losses = []
    
    
    start_time = time.time()
    # Training loop
    for epoch in range(num_epochs):
        model_intra.train()  # Set model to training mode
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
            
            for m, (G_batch, H_batch, w_batch) in enumerate(zip(torch.unbind(G_M_batch, dim=3),
                                                    torch.unbind(H_M_batch, dim=3),
                                                    torch.unbind(w_M_batch, dim=2))):
                y = y_M[:,m]
                
                model_outputs = model_intra(phi_batch, w_batch, y)
                
                # # print gradient to see gradient flows
                # for name, param in model_intra.named_parameters():
                #     if param.grad is not None:
                #         print(f"{name}: Gradient norm: {param.grad.norm().item()}")
                #     else:
                #         print(f"{name}: No grad")
                
                # calculate loss            
                loss = custom_loss_function(constants, G_batch, H_batch, hyperparameters, model_outputs)
            
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
            
            
        # Compute average loss for the epoch
        average_train_loss = total_train_loss / len(train_loader) / model_intra.M
        
        # Log the loss
        writer.add_scalar('Loss/Training', average_train_loss, epoch)
        # writer.flush()
        
        training_losses.append(average_train_loss)
        
        
        
        
        # Validation phase
        model_intra.eval()  # Set model to evaluation mode
        model_intra_tester = SREL_intra_rep_tester(constants, model_intra).to(device)
        model_intra_tester.device = device
        
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
                
                batch_size = phi_batch.size(0)
                sinr_M_batch = torch.empty(batch_size,constants['M'])
                
                # sinr_M = torch.empty(model_intra.M)
                for m, (G_batch, H_batch, w_batch) in enumerate(zip(torch.unbind(G_M_batch, dim=3),
                                                        torch.unbind(H_M_batch, dim=3),
                                                        torch.unbind(w_M_batch, dim=2))):
                    y = y_M[:,m]
                    
                    model_outputs = model_intra(phi_batch, w_batch, y)
                    
                    # calculate loss                            
                    val_loss = custom_loss_function(constants, G_batch, H_batch, hyperparameters, model_outputs)
                    total_val_loss += val_loss.item()
                
                    #s_stack_batch = model_outputs['s_stack_batch']
                    #s_optimal_batch = s_stack_batch[:,-1,:].squeeze()
                    #sinr_M_batch[:,m] = sinr_function(constants, G_batch, H_batch, s_optimal_batch)
            
            s_stack_batch = model_intra_tester(phi_batch, w_M_batch, y_M)
            s_optimal_batch = s_stack_batch[:,-1,:]
            sum_of_worst_sinr_avg += worst_sinr_function(constants, s_optimal_batch, G_M_batch, H_M_batch)
                    
        average_val_loss = total_val_loss / len(test_loader) / model_intra.M
        validation_losses.append(average_val_loss)
    
        # Log the loss
        writer.add_scalar('Loss/Testing', average_val_loss, epoch)
        writer.flush()
    
        # batch_size = phi_batch.size(0)
        
        
        average_worst_sinr_db = 10*torch.log10(sum_of_worst_sinr_avg/ len(test_loader))  # Compute average loss for the epoch
        print(f'Epoch [{epoch+1}/{num_epochs}]', 
              f'Train Loss = {average_train_loss:.4f}, '
              f'average_worst_sinr = {average_worst_sinr_db:.2f} dB')
            
        
        
    # End time
    end_time = time.time()
    
    # Calculate the duration
    duration = end_time - start_time
    
    print(f"Training completed in: {duration:.2f} seconds")
        
    # After completing all epochs, plot the training loss
    
    plot_losses(training_losses, validation_losses)
    
    
    
    # save model's information
    if data_num=='2e3':
        # save model's information
        save_dict = {
            'state_dict': model_intra.state_dict(),
            'N_step': model_intra.N_step,
            # Include any other attributes here
        }
        # save
        dir_weight_save = f'weights/SREL_intra_rep_rho/{current_time}_Nstep{N_step:02d}_batch{batch_size:02d}'
        os.makedirs(dir_weight_save, exist_ok=True)
        torch.save(save_dict, os.path.join(dir_weight_save, 'model_with_attrs.pth'))
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    
if __name__ == "__main__":
    N_step = 5	
    main(N_step)
    
    N_step = 10	
    main(N_step)
