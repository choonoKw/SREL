# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:48:29 2024

@author: jbk5816

Bring all MATLAB codes for SRED..
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

