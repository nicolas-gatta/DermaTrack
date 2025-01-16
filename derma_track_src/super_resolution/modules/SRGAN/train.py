import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch import nn
from tqdm import tqdm
from super_resolution.modules.SRGAN.discriminator_model import SRGANDiscriminator
from super_resolution.modules.SRGAN.generator_model import SRGANGenerator
from super_resolution.modules.SRGAN.loss import VGGLoss
from super_resolution.modules.utils.running_average import RunningAverage
from torch.utils.data.dataloader import DataLoader
from super_resolution.modules.utils.dataloader import H5Dataset

def train_model(train_file, eval_file, output_dir, learning_rate: float = 1e-4, seed: int = 1, batch_size: int = 16, num_epochs: int = 100, num_workers: int = 8):
    
    cudnn.benchmark = True
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    
    dataset = H5Dataset(train_file)

    # Split into train/validation datasets
    train_size = int(0.9 * len(dataset))
    val_size = int(0.1 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # GAN Model
    generator = SRGANGenerator().to(device)
    discriminator = SRGANDiscriminator().to(device)
    
    # VGG LOSS Error
    content_loss = VGGLoss().to(device)
    
    # Adversarial loss (Binary Cross Entropy Loss)
    adversarial_loss = nn.BCELoss()
    
    generator_optimizer = optim.Adam(generator.parameters(), lr = 1e-4)
    
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr = 1e-4)
    
    train_loss, val_loss = RunningAverage(), RunningAverage()
    
    # Trainign and Validation loop
    for epoch in range(num_epochs):
        
        train_loss.reset()
        val_loss.reset()
        
        with tqdm(total=len(train_loader) + len(val_loader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=True) as pbar:
            
            for loop_type, dataloader in [("Training", train_loader), ("Validation", val_loader)]:
                
                torch.set_grad_enabled(loop_type == "Training")
                
                for low_res, high_res in tqdm(dataloader, desc = loop_type, leave=False):
                    
                    low_res, high_res = low_res.to(device), high_res.to(device)


                    #=======================Train Discriminator===================
                    
                    # Detach is use to stop the gradient to do the back propagation
                    fake_image = generator(low_res).detach()
                    
                    real_output = discriminator(high_res)
                    
                    fake_output = discriminator(fake_image)
                    
                    # 1 => True image & 0 => Fake Image
                    discriminator_loss = discriminator_loss(real_output, torch.ones_like(real_output)) + discriminator_loss(fake_output, torch.zeros_like(fake_output))
                    
                    if loop_type == "Training":
                        discriminator_optimizer.zero_grad()
                                                
                        discriminator_loss.backward()
                        
                        discriminator_optimizer.step()
                        
                
                    #=======================Train Generator===================
                    
                    # Detach is use to stop the gradient to do the back propagation
                    fake_image = generator(low_res)
                    
                    fake_output = discriminator(fake_image)
                    
                    generator_loss = content_loss(fake_image, high_res) + (1e-3 * adversarial_loss(fake_output, torch.ones_like(fake_output)))
                    
                    if loop_type == "Training":
                        generator_optimizer.zero_grad()
                        
                        generator_loss.backward()
                        
                        generator_optimizer.step()
                    
                        train_loss.update(generator_loss)
                        
                    else:
                        
                        val_loss.update(generator_loss)
                    
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