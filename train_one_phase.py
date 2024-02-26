# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 23:37:36 2024

@author: jbk5816
"""

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from utils.complex_valued_dataset import ComplexValuedDataset
from utils.load_scalars_from_setup import load_scalars_from_setup
from utils.load_mapVector import load_mapVector
from model.srel_model_reg_multiStep import SREL

from utils.custom_loss import custom_loss_function
from utils.worst_sinr import worst_sinr_function

from torch.utils.tensorboard import SummaryWriter #tensorboard
from visualization.plotting import plot_losses # result plot

import time

def main():
    # Load dataset
    constants = load_scalars_from_setup('data/data_setup.mat')
    v_M, Lv = load_mapVector('data/data_mapV.mat')
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
    constants['Lv'] = Lv
    Nt = constants['Nt']
    N = constants['N']
    constants['modulus'] = 1 / torch.sqrt(torch.tensor(Nt * N, dtype=torch.float))

    ###############################################################
    # Initialize model
    N_step = 4
    constants['N_step'] = N_step
    model = SREL(constants)
    num_epochs = 10
    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # loss setting
    hyperparameters = {
        'lambda_eta': 1e-7,
        'lambda_sinr': 1e-3,
    }
    
    # prepare to write logs for tensorboard
    writer = SummaryWriter(f'runs/SREL_multiStep{N_step:02d}_{data_num}')
    global_step = 0
    
    # tensorboard --logdir=runs --reload_interval 5
    ###############################################################
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    # List to store average loss per epoch
    training_losses = []
    validation_losses = []
    
    
    start_time = time.time()
    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_train_loss = 0.0
        
        
        for phi_batch, w_M_batch, G_M_batch, H_M_batch in train_loader:
            # Perform training steps
            # for m in range(1,constants['M']+1):
            optimizer.zero_grad()
            # s_batch = modulus * torch.exp(1j * phi_batch)
            
            model_outputs = model(phi_batch, w_M_batch, v_M)
            # s_optimal_batch = modulus * torch.exp(1j * phi_optimal_batch)
            
            # calculate loss
            # primary_loss = sum_of_reciprocal(constants, s_optimal_batch, G_M_batch, H_M_batch)
            # regularization_loss = regularizer_eta(constants, s_batch, G_M_batch, H_M_batch, eta_M_batch)
            
            loss = custom_loss_function(constants, G_M_batch, H_M_batch, hyperparameters, model_outputs)
        
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            
            global_step += 1
            
        # Compute average loss for the epoch
        average_train_loss = total_train_loss / len(train_loader)
        
        # Log the loss
        writer.add_scalar('Loss/Training', average_train_loss, epoch)
        # writer.flush()
        
        training_losses.append(average_train_loss)
        
        
        
        
        # Validation phase
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0.0
        
        sum_of_worst_sinr = 0.0  # Accumulate loss over all batches
        
        with torch.no_grad():  # Disable gradient computation
            for phi_batch, w_M_batch, G_M_batch, H_M_batch in test_loader:
                # s_batch = modulus * torch.exp(1j * phi_batch)
                
                # phi_optimal_batch, eta_M_batch = model(phi_batch, w_M_batch, v_M)
                # s_optimal_batch = modulus * torch.exp(1j * phi_optimal_batch)
                model_outputs = model(phi_batch, w_M_batch, v_M)
                
                # calculate loss
                # primary_loss = sum_of_reciprocal(constants, s_optimal_batch, G_M_batch, H_M_batch)
                # regularization_loss = regularizer_eta(constants, s_batch, G_M_batch, H_M_batch, eta_M_batch)
                # val_loss = primary_loss + lambda_eta * regularization_loss
                
                val_loss = custom_loss_function(constants, G_M_batch, H_M_batch, hyperparameters, model_outputs)
                total_val_loss += val_loss.item()
                
                s_list_batch = model_outputs['s_list_batch']
                sum_of_worst_sinr += worst_sinr_function(constants, s_list_batch[:,:,-1], G_M_batch, H_M_batch)
        # validation_losses.append(total_val_loss / len(test_loader))
        
        
        
        # Store the average loss for this epoch
        average_val_loss = total_val_loss / len(test_loader)
        validation_losses.append(average_val_loss)
        
        # Log the loss
        writer.add_scalar('Loss/Testing', average_val_loss, epoch)
        writer.flush()
        
        average_worst_sinr_db = 10*torch.log10(sum_of_worst_sinr/ len(test_loader))  # Compute average loss for the epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              # f'Train Loss = {average_train_loss:.4f}, '
              f'average_worst_sinr = {average_worst_sinr_db:.4f} dB')
        
    # End time
    end_time = time.time()
    
    # Calculate the duration
    duration = end_time - start_time
    
    print(f"Training completed in: {duration:.2f} seconds")
        
    # After completing all epochs, plot the training loss
    plot_losses(training_losses, validation_losses)
    
    torch.save(model.state_dict(), 'weights/model_weights.pth')
    
if __name__ == "__main__":
    main()

