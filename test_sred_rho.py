# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:48:29 2024

@author: jbk5816

Bring all MATLAB codes for SRED..
"""

import torch
from model.sred_rho import SRED_rho
from utils.load_scalars_from_setup import load_scalars_from_setup
from utils.load_mapVector import load_mapVector
from torch.utils.data import DataLoader

from utils.complex_valued_dataset import ComplexValuedDataset
# from torch.utils.data import Subset

from utils.custom_loss_intra import sinr_function
from utils.worst_sinr import worst_sinr_function

import datetime
import os
from scipy.io import savemat


# Load constants and model architecture parameters, similar to train.py
constants = load_scalars_from_setup('data/data_setup.mat')
y_M, Ly = load_mapVector('data/data_mapV.mat')
constants['Ly'] = Ly
# constants['N_step'] = 5  # Ensure this matches the training setup
constants['modulus'] = 1 / torch.sqrt(torch.tensor(constants['Nt'] * constants['N'], dtype=torch.float))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the bundled dictionary
dir_dict_saved = 'weights/SRED_rho/20240319-022519_Nstep10_batch30_sinr_12.58dB'
loaded_dict = torch.load(os.path.join(dir_dict_saved,'model_with_attrs.pth'), 
                         map_location=device)
N_step = loaded_dict['N_step']
constants['N_step'] = N_step

# Step 1: Instantiate model1
model_sred_rho = SRED_rho(constants).to(device)
model_sred_rho.device = device
model_sred_rho.load_state_dict(loaded_dict['state_dict'])             

data_num = '1e1'

M = constants['M']
# model_intra.load_state_dict(torch.load(weight_file_location))
model_sred_rho.eval()  # Set the model to evaluation mode

dataset = ComplexValuedDataset(f'data/data_trd_{data_num}.mat')
test_loader = DataLoader(dataset, batch_size=2, shuffle=False)  # Adjust batch size as needed

with torch.no_grad(): # Disable gradient computation during testing
    for phi_batch, w_M_batch, G_M_batch, H_M_batch in test_loader:
        phi_batch = phi_batch.to(device)
        G_M_batch = G_M_batch.to(device)
        H_M_batch = H_M_batch.to(device)
        w_M_batch = w_M_batch.to(device)
        y_M = y_M.to(device)  # If y_M is a tensor that requires to be on the GPU
        
        
        model_outputs = model_sred_rho(phi_batch, w_M_batch, y_M, G_M_batch, H_M_batch)
        
        s_stack_batch = model_outputs['s_stack_batch']
        
        batch_size = len(test_loader)
        # for idx_batch in range(batch_size):
        for idx_batch, (G_M, H_M) in enumerate(zip(torch.unbind(G_M_batch, dim=0),
                                                torch.unbind(H_M_batch, dim=0))):
            G_M = G_M.unsqueeze(0)
            H_M = H_M.unsqueeze(0)
            for update_step in range(N_step+1):
                s = s_stack_batch[idx_batch,update_step,:].unsqueeze(0)
                
                
                
                sinr_db = 10*torch.log10(worst_sinr_function(constants, s, G_M, H_M))
                print(f'Step {update_step:02d}, SINR = {sinr_db:.4f} dB')
