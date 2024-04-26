# -*- coding: utf-8 -*-
"""
Created on April 2 2024

@author: jbk5816
"""

import torch
# import torch.optim as optim
# import numpy as np
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader, Subset

# from utils.complex_valued_dataset import ComplexValuedDataset
# from utils.training_dataset import TrainingDataSet
from utils.load_scalars_from_setup import load_scalars_from_setup
# from utils.load_mapVector import load_mapVector

# from model.sred_rho import SRED_rho
# print('SRED_rho OG.')

# from model.sred_rho_DO import SRED_rho
# print('SRED_rho with Drop Out (DO)')

from model.srel_intra_phase1 import SREL_intra_phase1_rep_rho, SREL_intra_phase1_vary_rho

from utils.check_module_structure import is_single_nn

from model.srel_intra_infer import SREL_intra_phase1_infer
# print('SRED_rho with Batch Normalization (BN)')


# from utils.custom_loss_intra import custom_loss_intra_phase1
# from utils.worst_sinr import worst_sinr_function

# from utils.worst_sinr import worst_sinr_function

# from torch.utils.tensorboard import SummaryWriter #tensorboard
# tensorboard --logdir=runs/SREL --reload_interval 5
# tensorboard --logdir=runs/SREL_intra
# from visualization.plotting import plot_losses # result plot

# import datetime
# import time
import os
import argparse

from utils.test_joint_design import test
# from utils.save_result_mat import save_result_mat

# from utils.format_time import format_time

# import torch.nn as nn

def main(weightdir):
    # Load dataset
    constants = load_scalars_from_setup('data/data_setup.mat')
    # y_M, Ly = load_mapVector('data/data_mapV.mat')
    # data_num = 1e1
    
    
    # loading constant
    constants['Ly'] = 570
    Nt = constants['Nt']
    # M = constants['M']
    N = constants['N']
    
    constants['modulus'] = 1 / torch.sqrt(torch.tensor(Nt * N, dtype=torch.float))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ###############################################################
    ## Load weight
    ###############################################################
    # Load the bundled dictionary
    if weightdir:
        dir_dict_saved = weightdir
        loaded_dict = torch.load(os.path.join(dir_dict_saved,'model_with_attrs.pth'), 
                                 map_location=device)
    else:
        dir_dict_saved = (
            'weights/intra_phase1/rep_rho/'
            '20240403-132816_Nstep10_batch02_sinr_15.47dB')
        loaded_dict = torch.load(os.path.join(dir_dict_saved,'model_with_attrs.pth'), 
                                 map_location=device)
            
    state_dict = loaded_dict['state_dict']
    N_step = loaded_dict['N_step']
    
    ###############################################################
    ## Control Panel
    ###############################################################
    # Initialize model
    # N_step = 10
    constants['N_step'] = N_step
    
    if is_single_nn(state_dict,'est_rho_modules'):
        model_intra_phase1 = SREL_intra_phase1_rep_rho(constants)
          
    else:
        model_intra_phase1 = SREL_intra_phase1_vary_rho(constants)
        
    model_intra_phase1.load_state_dict(loaded_dict['state_dict']) 
        
    
    ###############################################################
    
    model_intra_phase1.to(device)
    model_intra_phase1.device = device
    # # for results

    
    # validation
    model_intra_phase1.eval()  # Set model to evaluation mode
    model_intra_tester = SREL_intra_phase1_infer(constants, model_intra_phase1)
    model_intra_tester.device = device

    
    worst_sinr_stack_list, f_stack_list = test(constants,model_intra_tester,1e-7)
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a model.")
    
    parser.add_argument("--weightdir", type=str, 
                        help="Save the model weights after training")
    # parser.add_argument("--save-mat", action="store_true",
    #                     help="Save mat file including worst-sinr values")
    
    args = parser.parse_args()
    
    main(weightdir=args.weightdir)
    
