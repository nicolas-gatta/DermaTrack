import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch import nn
from tqdm import tqdm
from super_resolution.modules.SRGAN.discriminator_model import SRGAN_Discriminator
from super_resolution.modules.SRGAN.generator_model import SRGAN_Generator
from super_resolution.modules.SRGAN.loss import VGGLoss
from super_resolution.modules.utils.running_average import RunningAverage
from torch.utils.data.dataloader import DataLoader

def train_model(train_file, eval_file, output_dir, learning_rate: float = 1e-4, seed: int = 1, batch_size: int = 16, num_epochs: int = 100, num_workers: int = 8):
    
    cudnn.benchmark = True
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    
    dataset = ""

    # Split into train/validation datasets
    train_size = int(0.9 * len(dataset))
    val_size = int(0.1 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # GAN Model
    generator = SRGAN_Generator().to(device)
    discriminator = SRGAN_Discriminator().to(device)
    
    # VGG LOSS Eror
    criterion = VGGLoss().to(device)
    
    optimizer = optim.Adam()
    
    train_loss, val_loss = RunningAverage(), RunningAverage()
    
    # Trainign and Validation loop
    for epoch in range(num_epochs):
        
        with tqdm(total=len(train_loader) + len(val_loader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=True) as pbar:
            
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
                        "Train Loss": f"{train_loss.average:.4f}" if train_loss.average > 0 else "N/A",
                        "Val Loss": f"{val_loss.average:.4f}" if val_loss.average > 0 else "N/A",
                    })

    # Save the model
    torch.save(generator.state_dict(), output_dir)
    print(f"Model saved as {output_dir}")

    # Close dataset
    dataset.close()