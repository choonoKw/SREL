# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:27:10 2024

@author: jbk5816
"""

import matplotlib.pyplot as plt

def plot_training_loss(epoch_losses):

    # Plotting the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()
    
def plot_losses(training_losses, validation_losses):
    plt.figure('Loss Plot', figsize=(10, 6))
    plt.clf()  # Clear the current figure
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.show()