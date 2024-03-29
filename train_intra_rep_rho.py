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

from utils.training_dataset import TrainingDataSet
from utils.load_scalars_from_setup import load_scalars_from_setup
# from utils.load_mapVector import load_mapVector
from model.srel_intra_rep_rho import SREL_intra_rep_rho

from utils.custom_loss_intra import custom_loss_function

# from utils.worst_sinr import worst_sinr_function
from model.srel_intra_rep_rho_tester import SREL_intra_rep_rho_tester
from utils.worst_sinr import worst_sinr_function

from torch.utils.tensorboard import SummaryWriter #tensorboard
# tensorboard --logdir=runs/SREL --reload_interval 5
# tensorboard --logdir=runs/SREL_intra_rep
from visualization.plotting import plot_losses # result plot

import argparse
import datetime
import time
import os

from utils.validation import validation

from utils.format_time import format_time

# import torch.nn as nn

def main(save_weights, save_logs, lambda_eta):
    # Load dataset
    constants = load_scalars_from_setup('data/data_setup.mat')
    # y_M, Ly = load_mapVector('data/data_mapV.mat')
    data_num = 1e1
    # batch_size = 20
    
    
    # loading constant
    constants['Ly'] = 570
    Nt = constants['Nt']
    N = constants['N']
    constants['modulus'] = 1 / torch.sqrt(torch.tensor(Nt * N, dtype=torch.float))
    num_case = 1 # 

    ###############################################################
    ## Control Panel
    ###############################################################
    # Initialize model
    N_step = 10
    constants['N_step'] = N_step
    model_intra = SREL_intra_rep_rho(constants)
    batch_size = 5
    num_epochs = 10
    # Initialize the optimizer
    learning_rate = 1e-4
    
    optimizer = optim.Adam(model_intra.parameters(), lr=learning_rate)
    
    # loss setting
    hyperparameters = {
        # 'lambda_eta': 1e-6,
        'lambda_eta': lambda_eta
        'lambda_sinr': 1e-2,
    }    
    
    print(f'learning_rate={learning_rate:.0e}, '
          f"lambda_eta={hyperparameters['lambda_eta']:.0e}")
    ###############################################################
    # for results
    # Get the current time
    start_time_tag = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')    
    
    # Create a unique directory name using the current time and the N_step value
    log_dir = (
        f'runs/SREL_intra_rep_rho/data{data_num:.0e}/{start_time_tag}'
        f'_Nstep{constants["N_step"]:02d}_batch{batch_size:02d}'
        f'_lr_{learning_rate:.0e}'
    )
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
    
    
    start_time_total = time.time()
    
    # Training loop
    num_case = 24
    for epoch in range(num_epochs):
        total_train_loss = 0.0
        total_val_loss = 0.0
        
        sum_of_worst_sinr_avg = 0.0  # Accumulate loss over all batches
        
        start_time_epoch = time.time()  # Start timing the inner loop
        
        for idx_case in range(num_case):
            case_num = idx_case + 1
            dataset = TrainingDataSet(f'data/{data_num:.0e}/data_trd_{data_num:.0e}_case{case_num:02d}.mat')
            
            # Split dataset into training and validation
            train_indices, val_indices = train_test_split(
                range(len(dataset)),
                test_size=0.2,  # 20% for validation
                random_state=42
            )
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
            
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            
            model_intra.train()  # Set model to training mode
            total_train_loss = 0.0
            
            
            for phi_batch, w_M_batch, G_M_batch, H_M_batch in train_loader:
                # if torch.cuda.is_available():
                phi_batch = phi_batch.to(device)
                G_M_batch = G_M_batch.to(device)
                H_M_batch = H_M_batch.to(device)
                w_M_batch = w_M_batch.to(device)
                y_M = dataset.y_M.to(device)  # If y_M is a tensor that requires to be on the GPU
                
                # Perform training steps
                optimizer.zero_grad()
                
                for m, (G_batch, H_batch, w_batch) in enumerate(zip(torch.unbind(G_M_batch, dim=3),
                                                        torch.unbind(H_M_batch, dim=3),
                                                        torch.unbind(w_M_batch, dim=2))
                                                        ):
                    y = y_M[:,m]
                    
                    model_outputs = model_intra(phi_batch, w_batch, y)
                    
                    # # print gradient to see gradient flows
                    # for name, param in model_intra.named_parameters():
                    #     if param.grad is not None:
                    #         print(f"{name}: Gradient norm: {param.grad.norm().item()}")
                    #     else:
                    #         print(f"{name}: No grad")
                    
                    # calculate loss            
                    loss = custom_loss_function(
                        constants, G_batch, H_batch, hyperparameters, model_outputs)
                
                    loss.backward()
                    optimizer.step()
                    
                    total_train_loss += loss.item()
                    
            # Validation phase
            model_intra.eval()  # Set model to evaluation mode
            model_intra_tester = SREL_intra_rep_rho_tester(constants, model_intra).to(device)
            model_intra_tester.device = device
            
            # total_val_loss = 0.0
            # sum_of_worst_sinr_avg = 0.0  # Accumulate loss over all batches
            
            with torch.no_grad():  # Disable gradient computation
                for phi_batch, w_M_batch, G_M_batch, H_M_batch in test_loader:
                    # s_batch = modulus * torch.exp(1j * phi_batch)
                    phi_batch = phi_batch.to(device)
                    G_M_batch = G_M_batch.to(device)
                    H_M_batch = H_M_batch.to(device)
                    w_M_batch = w_M_batch.to(device)
                    y_M = y_M.to(device)  # If y_M is a tensor that requires to be on the GPU
                    
                    batch_size = phi_batch.size(0)
                    # sinr_M_batch = torch.empty(batch_size,constants['M'])
                    
                    # sinr_M = torch.empty(model_intra.M)
                    for m, (G_batch, H_batch, w_batch) in enumerate(zip(torch.unbind(G_M_batch, dim=3),
                                                            torch.unbind(H_M_batch, dim=3),
                                                            torch.unbind(w_M_batch, dim=2))
                                                            ):
                        y = y_M[:,m]
                        
                        model_outputs = model_intra(phi_batch, w_batch, y)
                        
                        # calculate loss                            
                        val_loss = custom_loss_function(
                            constants, G_batch, H_batch, hyperparameters, model_outputs)
                        total_val_loss += val_loss.item()
                    
                        #s_stack_batch = model_outputs['s_stack_batch']
                        #s_optimal_batch = s_stack_batch[:,-1,:].squeeze()
                        #sinr_M_batch[:,m] = sinr_function(constants, G_batch, H_batch, s_optimal_batch)
                    model_outputs = model_intra_tester(phi_batch, w_M_batch, y_M)
                    s_stack_batch = model_outputs['s_stack_batch']
                    s_optimal_batch = s_stack_batch[:,-1,:]
                    sum_of_worst_sinr_avg += worst_sinr_function(
                        constants, s_optimal_batch, G_M_batch, H_M_batch)
                    
                #sum_of_worst_sinr_avg += torch.sum(torch.min(sinr_M_batch, dim=1).values)/batch_size
                
                
        # Compute average loss for the epoch and Log the loss
        average_train_loss = total_train_loss / model_intra.M / len(train_loader) / num_case
        average_train_loss_db = 10*np.log10(average_train_loss)
        training_losses.append(average_train_loss_db)
        
        average_val_loss = total_val_loss / model_intra.M / len(test_loader) / num_case
        average_val_loss_db = 10*np.log10(average_val_loss)
        validation_losses.append(average_val_loss_db)
        
        if save_logs:
            writer.add_scalar('Loss/Training [dB]', average_train_loss_db, epoch)
            
            writer.add_scalar('Loss/Testing [dB]', average_val_loss_db, epoch)
            writer.flush()
        
        # # Compute average loss for the epoch
        # average_train_loss = total_train_loss / len(train_loader) / model_intra.M
        
        # # Log the loss
        # writer.add_scalar('Loss/Training', average_train_loss, epoch)
        # # writer.flush()
        
        # training_losses.append(average_train_loss)
                    
        # average_val_loss = total_val_loss / len(test_loader) / model_intra.M
        # validation_losses.append(average_val_loss)
    
        # # Log the loss
        # writer.add_scalar('Loss/Testing', average_val_loss, epoch)
        
        
    
        worst_sinr_avg_db = 10*np.log10(sum_of_worst_sinr_avg/ len(test_loader) / num_case)  # Compute average loss for the epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], '
             f'Train Loss = {average_train_loss_db:.2f} dB, '
             f'average_worst_sinr = {worst_sinr_avg_db:.4f} dB')
        
        time_spent_epoch = time.time() - start_time_epoch  # Time spent in the current inner loop iteration
        time_left = time_spent_epoch * (num_epochs - epoch - 1)  # Estimate the time left
        formatted_time_left = format_time(time_left)
        print(f"{formatted_time_left} left")
            
        
        
    # End time
    end_time = time.time()
    
    # Calculate the time_spent_total
    time_spent_total = end_time - start_time_total
    
    print(f"Training completed in: {time_spent_total:.2f} seconds")
    
    plot_losses(training_losses, validation_losses)
    
    # validation
    sinr_db_opt = validation(constants,model_intra_tester)
    
    
    # # validate the values of rho
    # rho_stack_batch = model_outputs['rho_stack_batch']
    # rho_stack_avg = torch.sum(rho_stack_batch, dim=0)/batch_size
    # print('\n','rho values = ')
    # for n in range(model_intra.N_step):
    #     print(f'{rho_stack_avg[n].item():.4f}')
    # print(f'finished time: {start_time_tag}')
    
    # # SINR values for each step
    # # sinr_list = torch.zeros(N_step)
    # for update_step in range(N_step+1):
    #     s_batch = s_stack_batch[:,update_step,:]
    #     # sinr_list[update_step]= worst_sinr_function(constants, s_batch, G_M_batch, H_M_batch)
    #     sinr_db = 10*torch.log10(worst_sinr_function(constants, s_batch, G_M_batch, H_M_batch))
    #     print(f'Step {update_step:02d}, SINR = {sinr_db:.4f} dB')
    
    # save model's information
    if save_weights:
        save_dict = {
            'state_dict': model_intra.state_dict(),
            'N_step': model_intra.N_step,
            # Include any other attributes here
        }
        # save
        dir_weight_save = (
            f'weights/SREL_intra_rep_rho/{start_time_tag}'
            f'_Nstep{N_step:02d}_batch{batch_size:02d}'
            f'_sinr_{sinr_db_opt:.2f}dB'
        )
        os.makedirs(dir_weight_save, exist_ok=True)
        torch.save(save_dict, os.path.join(dir_weight_save, 'model_with_attrs.pth'))
    
    
    
    
# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
#         if m.bias is not None:
#             torch.nn.init.constant_(m.bias, 0)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model.")
    
    parser.add_argument("--save-weights", action="store_true",
                        help="Save the model weights after training")
    parser.add_argument("--save-logs", action="store_true",
                        help="Save logs for Tensorboard after training")
    
    args = parser.parse_args()
    
    main(save_weights=args.save_weights, save_logs=args.save_logs,batch_size=10)
    
#     batch_size = 30	
#     main(batch_size)
    
#     batch_size = 50	
#     main(batch_size)