# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:06:41 2024

@author: jbk5816
"""
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from utils.complex_valued_dataset import ComplexValuedDataset
from utils.load_scalars_from_setup import load_scalars_from_setup
from utils.load_mapVector import load_mapVector
from model.srel_model import SREL

from utils.custom_loss import sum_of_reciprocal, regularizer_eta
from utils.worst_sinr_db import worst_sinr_db_function

from visualization.plotting import plot_losses

def main():
    # Load dataset
    constants = load_scalars_from_setup('data/data_setup.mat')
    v_M, Lv = load_mapVector('data/data_mapV.mat')
    dataset = ComplexValuedDataset('data/data_trd.mat')
    
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
    
    # defining constant
    constants["Lv"] = Lv
    Nt = constants['Nt']
    N = constants['N']
    modulus = 1 / torch.sqrt(torch.tensor(Nt * N, dtype=torch.float))

    # Initialize model
    model = SREL(constants)
    num_epochs = 10
    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lambda_eta = 0.1
    
    # List to store average loss per epoch
    training_losses = []
    validation_losses = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_train_loss = 0.0
        
        
        for phi_batch, w_M_batch, G_M_batch, H_M_batch  in train_loader:
            # Perform training steps
            # for m in range(1,constants['M']+1):
            optimizer.zero_grad()
            s_batch = modulus * torch.exp(1j * phi_batch)
            
            phi_optimal_batch = model(phi_batch, w_M_batch, v_M)
            s_optimal_batch = modulus * torch.exp(1j * phi_optimal_batch)
            
            # calculate loss
            primary_loss = sum_of_reciprocal(constants, s_optimal_batch, G_M_batch, H_M_batch)
            regularization_loss = regularizer_eta(constants, s_batch, G_M_batch, H_M_batch, eta_M_batch)
            
            loss = primary_loss + lambda_eta * regularization_loss
        
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
            
        # Compute average loss for the epoch
        training_losses.append(total_train_loss / len(train_loader))
        
        
        # Validation phase
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0.0
        
        sum_of_worst_sinr = 0.0  # Accumulate loss over all batches
        
        with torch.no_grad():  # Disable gradient computation
            for phi_batch, w_M_batch, G_M_batch, H_M_batch in test_loader:
                phi_optimal_batch = model(phi_batch, w_M_batch, v_M)
                s_optimal_batch = modulus * torch.exp(1j * phi_optimal_batch)
                val_loss = sum_of_reciprocal(constants, s_optimal_batch, G_M_batch, H_M_batch)
                total_val_loss += val_loss.item()
                
                sum_of_worst_sinr += worst_sinr_db_function(constants, s_optimal_batch, G_M_batch, H_M_batch)
        validation_losses.append(total_val_loss / len(test_loader))
        
        average_worst_sinr_db = 10*torch.log10(sum_of_worst_sinr/ len(test_loader))  # Compute average loss for the epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_worst_sinr_db:.4f} dB')
        
        # Store the average loss for this epoch
        validation_losses.append(total_val_loss / len(test_loader))
        
    # After completing all epochs, plot the training loss
    plot_losses(training_losses, validation_losses)
    
if __name__ == "__main__":
    main()

