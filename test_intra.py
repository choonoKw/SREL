# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 13:38:42 2024

@author: jbk5816
"""

import torch
from model.srel_intra import SREL_intra
from utils.load_scalars_from_setup import load_scalars_from_setup
from utils.load_mapVector import load_mapVector
from torch.utils.data import DataLoader

from utils.complex_valued_dataset import ComplexValuedDataset
# from torch.utils.data import Subset

from utils.custom_loss_intra import sinr_function

import datetime
import os
from scipy.io import savemat

# Load constants and model architecture parameters, similar to train.py
constants = load_scalars_from_setup('data/data_setup.mat')
y_M, Ly = load_mapVector('data/data_mapV.mat')
constants['Ly'] = Ly
# constants['N_step'] = 5  # Ensure this matches the training setup
constants['modulus'] = 1 / torch.sqrt(torch.tensor(constants['Nt'] * constants['N'], dtype=torch.float))


dir_dict = 'weights/SREL_intra/Nstep05_data1e1_20240228-211924'
loaded_dict = torch.load(os.path.join(dir_dict,'model_with_attrs.pth'))
N_step = loaded_dict['N_step']
constants['N_step'] = N_step

model_intra = SREL_intra(constants)
model_intra.load_state_dict(loaded_dict['state_dict'])               

# N_step = constants['N_step']
data_num = '1e1'

M = constants['M']
# model_intra.load_state_dict(torch.load(weight_file_location))
model_intra.eval()  # Set the model to evaluation mode

# Assuming you have a test dataset or using the validation dataset as an example
dataset = ComplexValuedDataset('data/data_trd_1e1.mat')
test_loader = DataLoader(dataset, batch_size=2, shuffle=False)  # Adjust batch size as needed

sum_of_worst_sinr_avg = 0.0

# compute the average of the worst-sinr
with torch.no_grad():  # Disable gradient computation during testing
    for phi_batch, w_M_batch, G_M_batch, H_M_batch in test_loader:
        
        batch_size = phi_batch.size(0)
        sinr_M_batch = torch.empty(batch_size,M)
        
        for m, (G_batch, H_batch, w_batch) in enumerate(zip(torch.unbind(G_M_batch, dim=3),
                                                torch.unbind(H_M_batch, dim=3),
                                                torch.unbind(w_M_batch, dim=2))):
            y = y_M[:,m]
            
            for idx_batch in range(batch_size):
                phi0 = phi_batch[idx_batch]
                w = w_batch[idx_batch]
                
                
                # Repeat the update process N_step times
                phi = phi0
                for update_step in range(N_step):
                    s = model_intra.modulus*torch.exp(1j *phi)
                    
                    x = torch.cat((s.real, s.imag, w.real, w.imag, y), dim=0)
                    eta = model_intra.est_eta_modules[update_step](x)
                    beta = model_intra.est_rho_modules[update_step](x)
                    break
                break
            break
        break
            
#             model_outputs = model_intra(phi_batch, w_batch, y)
#             # model_outputs = model(phi_batch, w_M_batch, y_M)
#             s_stack_batch = model_outputs['s_stack_batch']
#             s_optimal_batch = s_stack_batch[:,-1,:].squeeze()
        
#             sinr_M_batch[:,m] = sinr_function(constants, G_batch, H_batch, s_optimal_batch)
        
#         worst_sinr_batch = torch.min(sinr_M_batch, dim=1).values        
#         sum_of_worst_sinr_avg += torch.sum(worst_sinr_batch)/batch_size
        
        
        
# average_worst_sinr_db = 10*torch.log10(sum_of_worst_sinr_avg/ len(test_loader))  # Compute average loss for the epoch
# print(f'average_worst_sinr = {average_worst_sinr_db:.4f} dB')

# # save the last output
# data = {'w_M_batch': w_M_batch,'G_M_batch': G_M_batch, 'H_M_batch': H_M_batch, \
#         's_stack_batch': s_stack_batch}

    
# # Save to a .mat file
# current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')  
# dir_matFile =  'mat/SREL_intra/'
# os.makedirs(dir_matFile, exist_ok=True)
# file_path = os.path.join(dir_matFile, f'Nstep{N_step:02d}_data{data_num}_{current_time}.mat')
# savemat(file_path, data)
