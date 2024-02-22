import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.complex_valued_dataset import ComplexValuedDataset
from utils.load_scalars_from_setup import load_scalars_from_setup
from utils.load_mapVector import load_mapVector
from model.srel_model import SREL

from utils.custom_loss import custom_loss_function
from utils.worst_sinr_db import worst_sinr_db_function

from visualization.plotting import plot_training_loss

def main():
    # Load dataset
    constants = load_scalars_from_setup('data/data_setup.mat')
    v_M, Lv = load_mapVector('data/data_mapV.mat')
    dataset = ComplexValuedDataset('data/data_trd.mat')
    
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
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
    
    epoch_losses = []  # List to store average loss per epoch

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0  # Accumulate loss over all batches
        total_worst_sinr = 0.0  # Accumulate loss over all batches
        
        for phi_batch, w_M_batch, G_M_batch, H_M_batch  in data_loader:
            # Perform training steps
            # for m in range(1,constants['M']+1):
            optimizer.zero_grad()
            phi_optimal_batch = model(phi_batch, w_M_batch, v_M)
            # s = torch.exp
            
            s_optimal_batch = modulus * torch.exp(1j * phi_optimal_batch)
            loss = custom_loss_function(constants, s_optimal_batch, G_M_batch, H_M_batch)
        
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()  # Accumulate the batch loss
            
            total_worst_sinr += worst_sinr_db_function(constants, s_optimal_batch, G_M_batch, H_M_batch)
            
        average_loss = total_loss / len(data_loader)  # Compute average loss for the epoch
        
        
        average_worst_sinr_db = 10*torch.log10(total_worst_sinr/ len(data_loader))  # Compute average loss for the epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_worst_sinr_db:.4f} dB')
        
        epoch_losses.append(average_loss)  # Store the average loss for this epoch
        
    # After completing all epochs, plot the training loss
    plot_training_loss(epoch_losses)
    
if __name__ == "__main__":
    main()
