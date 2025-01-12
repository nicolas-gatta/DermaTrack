import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import copy


from torch import nn
from tqdm import tqdm
from models import SRCNN
from dataloader import H5Dataset
from torch.utils.data.dataloader import DataLoader

def train(train_file, eval_file, output_dir, scale: int = 3, learning_rate: float = 1e-4, seed: int = 1, batch_size: int = 16, num_epochs: int = 100, num_workers: int = 8):

    cudnn.benchmark = True
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    
    dataset = H5Dataset(train_file)

    # Split into train/validation datasets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, optimizer
    model = SRCNN().to(device)
    criterion = nn.MSELoss()  # Mean Squared Error for SR tasks
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loss, val_loss = 0.0, 0.0
    
    # Trainign and Validation loop
    for epoch in range(num_epochs):
        
        for loop_type in ["Training","Validation"]:
                
            total_loss = 0.0
        
            dataloader = train_loader if loop_type == "training" else val_loader
            with torch.set_grad_enabled(loop_type == "training"):
                for lr, hr in tqdm(dataloader, desc = loop_type, leave=False):
                    lr, hr = lr.to(device), hr.to(device)

                    # Forward
                    output = model(lr)
                    loss = criterion(output, hr)
                    
                    if loop_type == "Training":
                        # Backward
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    total_loss += loss.item()
                        
                train_loss if loop_type == "Training" else val_loss = total_loss / len(dataloader)

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), 'srcnn_model.pth')
    print("Model saved as 'srcnn_model.pth'")

    # Close dataset
    dataset.close()
    
def eval():
    pass