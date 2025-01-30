import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import copy


from torch import nn
from tqdm import tqdm
from super_resolution.modules.SRCNN.model import SRCNN
from super_resolution.modules.utils.dataloader import H5Dataset
from super_resolution.modules.utils.running_average import RunningAverage
from torch.utils.data.dataloader import DataLoader

def train_model(train_file, valid_file, eval_file, output_dir, learning_rate: float = 1e-4, seed: int = 1, batch_size: int = 16, num_epochs: int = 100, num_workers: int = 8):
    
    cudnn.benchmark = True
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    
    train_dataset = H5Dataset(train_file)
    val_dataset = H5Dataset(valid_file)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Initialize model, loss, optimizer
    model = SRCNN().to(device)
    
    # Mean Squared Error for SR tasks
    criterion = nn.MSELoss()  
    
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': learning_rate * 0.1}
    ], lr=learning_rate)
    
    train_loss, val_loss = RunningAverage(), RunningAverage()
            
    # Trainign and Validation loop
    for epoch in range(num_epochs):
        
        train_loss.reset()
        val_loss.reset()
        
        with tqdm(total = len(train_loader) + len(val_loader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=True) as pbar:
            
            for loop_type, dataloader in [("Training", train_loader), ("Validation", val_loader)]:
                
                torch.set_grad_enabled(loop_type == "Training")
                
                for low_res, high_res in tqdm(dataloader, desc = loop_type, leave=False):
                    
                    low_res, high_res = low_res.to(device), high_res.to(device)

                    # Forward
                    output = model(low_res)
                    
                    loss = criterion(output, high_res)
                    
                    if loop_type == "Training":
                        # Backward
                        optimizer.zero_grad()
                        
                        loss.backward()
                        
                        optimizer.step()
                        
                        train_loss.update(loss.item())
                    
                    else:
                        
                        val_loss.update(loss.item())
                        
                    
                    pbar.update(1)
                    
                    pbar.set_postfix({
                        "Mode": loop_type,
                        "Train Loss": f"{train_loss:.4f}" if train_loss > 0 else "N/A",
                        "Val Loss": f"{val_loss:.4f}" if val_loss > 0 else "N/A",
                    })

    # Save the model
    torch.save(model.state_dict(), output_dir)
    print(f"Model saved as '{output_dir}'")

    # Close dataset
    train_dataset.close()
    val_dataset.close()
    
    # evaluate_model(model = model, device = device, criterion = criterion, eval_file = eval_file)
    
def evaluate_model(model, device, criterion, eval_file):
    
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    dataset = H5Dataset(eval_file)
    eval_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    num_batches = len(eval_loader)
    result = {}
    
    with torch.no_grad():  # Disable gradient calculations
        for lr, hr in tqdm(eval_loader, desc="Evaluating", leave=False):
            lr, hr = lr.to(device), hr.to(device)
            
            # Forward
            output = model(lr)
            loss = criterion(output, hr)
            
            total_loss += loss.item()
            
            # Compute PSNR if required
            mse = torch.mean((output - hr) ** 2)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))  # Assuming normalized images
            total_psnr += psnr.item()
    
    result["loss"] = (total_loss / num_batches)
    result["psnr"] = (total_psnr / num_batches)
        
    dataset.close()
    
    return result