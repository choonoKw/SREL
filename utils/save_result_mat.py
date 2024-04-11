# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:19:46 2024

@author: jbk5816
"""

# import numpy as np
from scipy.io import savemat

# import os
# import sys

def save_result_mat(filename, worst_sinr_stack_list, f_stack_list):
    """
    Saves the given arrays to a .mat file.

    Parameters:
    - worst_sinr_stack_list: A numpy array
    - f_stack_list: A numpy array
    """
    # Define the filename for the .mat file
    
    
    # Create a dictionary of the variables to save
    mat_contents = {
        'worst_sinr_stack_list': worst_sinr_stack_list,
        'f_stack_list': f_stack_list
    }
    
    # Save the dictionary to a .mat file
    
    savemat(filename, mat_contents)
    print(f"Data saved to {filename}")
    
    