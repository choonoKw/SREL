# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 13:38:42 2024

@author: jbk5816
"""

import torch
from model.srel_twoPhase import SREL_intra
from utils.load_scalars_from_setup import load_scalars_from_setup
from utils.load_mapVector import load_mapVector
from torch.utils.data import DataLoader

from utils.complex_valued_dataset import ComplexValuedDataset
# from torch.utils.data import Subset

from utils.custom_loss_intra import sinr_function

# Load constants and model architecture parameters, similar to train.py
constants = load_scalars_from_setup('data/data_setup.mat')
y_M, Ly = load_mapVector('data/data_mapV.mat')
constants['Ly'] = Ly
constants['N_step'] = 5  # Ensure this matches the training setup
constants['modulus'] = 1 / torch.sqrt(torch.tensor(constants['Nt'] * constants['N'], dtype=torch.float))
model_intra = SREL_intra(constants)

N_step = constants['N_step']
data_num = '1e1'
weight_file_location = f'weights/Nstep{N_step:02d}_data{data_num}/model_weights.pth'


# Load saved model weights
M = constants['M']
model_intra.load_state_dict(torch.load(weight_file_location))
model_intra.eval()  # Set the model to evaluation mode

# Assuming you have a test dataset or using the validation dataset as an example
dataset = ComplexValuedDataset('data/data_trd_1e1.mat')
test_loader = DataLoader(dataset, batch_size=2, shuffle=False)  # Adjust batch size as needed

sum_of_worst_sinr_avg = 0.0

with torch.no_grad():  # Disable gradient computation during testing
    for phi_batch, w_M_batch, G_M_batch, H_M_batch in test_loader:
        
        batch_size = phi_batch.size(0)
        sinr_M_batch = torch.empty(M, batch_size)
        
        for m, (G_batch, H_batch, w_batch) in enumerate(zip(torch.unbind(G_M_batch, dim=3),
                                                torch.unbind(H_M_batch, dim=3),
                                                torch.unbind(w_M_batch, dim=2))):
            y = y_M[:,m]
            
            model_outputs = model_intra(phi_batch, w_batch, y)
            # model_outputs = model(phi_batch, w_M_batch, y_M)
            s_stack_batch = model_outputs['s_stack_batch']
            s_optimal_batch = s_stack_batch[:,-1,:].squeeze()
        
            sinr_M_batch[m,:] = sinr_function(constants, G_batch, H_batch, s_optimal_batch)
        
        sum_of_worst_sinr_avg += torch.sum(torch.min(sinr_M_batch, dim=0).values)/batch_size
        
average_worst_sinr_db = 10*torch.log10(sum_of_worst_sinr_avg/ len(test_loader))  # Compute average loss for the epoch
print(f'average_worst_sinr = {average_worst_sinr_db:.4f} dB')