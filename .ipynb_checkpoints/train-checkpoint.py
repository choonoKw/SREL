import torch
from complex_valued_dataset import ComplexValuedDataset
from srel_model import SREL
from torch.utils.data import DataLoader

def main():
    # Load dataset
    dataset = ComplexValuedDataset('/storage/work/jbk5816/MATLAB/SINR max/SREL/240202_mkTrainingData/data_trd1.mat')
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize model
    model = SREL()
    num_epochs = 100

    # Training loop
    for epoch in range(num_epochs):
        for phi_initial, G_M, H_M in data_loader:
            # Perform training steps
            optimizer.zero_grad()
            phi_optimal = model(phi_initial.squeeze(), G_M.squeeze(), H_M.squeeze())
            s = torch.exp
            
            loss = custom_loss(s_real, s_imag, G_real, G_imag, H_real, H_imag)
        
            loss.backward()
            optimizer.step()
            # Compute loss here, e.g., using custom_loss function
            # loss.backward()
            # optimizer.step()

if __name__ == "__main__":
    main()