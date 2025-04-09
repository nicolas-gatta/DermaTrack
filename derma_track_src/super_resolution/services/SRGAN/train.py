import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time
import os

from torch import nn
from tqdm import tqdm

from super_resolution.services.SRGAN.discriminator_model import SRGANDiscriminator
from super_resolution.services.SRGAN.generator_model import SRGANGenerator
from super_resolution.services.SRGAN.loss import VGGLoss

from torch.utils.data.dataloader import DataLoader

from super_resolution.services.utils.running_average import RunningAverage
from super_resolution.services.utils.dataloader import H5ImagesDataset
from super_resolution.services.utils.image_evaluator import ImageEvaluator
from super_resolution.services.utils.early_stopping import EarlyStopping
from super_resolution.services.utils.batch_sampler import SizeBasedImageBatch
from super_resolution.services.utils.json_manager import JsonManager, ModelField
from super_resolution.services.utils.super_resolution import SuperResolution

def train_model(model_name: str, train_file: str, valid_file: str, eval_file: str, output_path: str, 
                mode: str, scale: int, invert_mode: str, patch_size: int, stride: int, learning_rate: float = 1e-4, 
                seed: int = 1, batch_size: int = 16, num_epochs: int = 100, num_workers: int = 8):
    
    crop_size = 96
    
    early_stopping = EarlyStopping(patience = 10, delta = 0, verbose = False)
    
    cudnn.benchmark = True
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    train_dataset = H5ImagesDataset(train_file, crop_size = crop_size, up_scale_factor = scale)
    val_dataset = H5ImagesDataset(valid_file, crop_size = crop_size, up_scale_factor = scale)
    
    train_batch = SizeBasedImageBatch(image_sizes = train_dataset.image_sizes, batch_size = batch_size)
    val_batch = SizeBasedImageBatch(image_sizes = val_dataset.image_sizes, batch_size = batch_size, shuffle = False)

    train_loader = DataLoader(train_dataset, batch_sampler = train_batch, num_workers = num_workers, pin_memory = True, persistent_workers = True)
    val_loader = DataLoader(val_dataset, batch_sampler = val_batch, num_workers = num_workers, pin_memory = True, persistent_workers = True)

    generator = SRGANGenerator(up_scale = scale).to(device)

    starting_time = time.time()
    
    pretrained_model(model=generator, learning_rate=learning_rate, num_epochs=num_epochs, train_loader=train_loader, val_loader=val_loader, device=device)
    
    train_loss, val_loss = RunningAverage(), RunningAverage()
    
    epoch_train_loss, epoch_val_loss = RunningAverage(), RunningAverage()
    
    discriminator = SRGANDiscriminator(crop_size = crop_size).to(device)
    
    content_loss = VGGLoss().to(device)
    
    adversarial_loss = nn.BCELoss()
    
    generator_optimizer = optim.Adam(generator.parameters(), lr = learning_rate)
    
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr = learning_rate)
        
    for epoch in range(num_epochs):
        
        train_loss.reset()
        val_loss.reset()
        
        with tqdm(total = len(train_loader) + len(val_loader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=True, dynamic_ncols=True) as pbar:
            
            for loop_type, dataloader in [("Training", train_loader), ("Validation", val_loader)]:
                
                generator.train() if loop_type == "Training" else generator.eval()
                
                with torch.set_grad_enabled(loop_type == "Training"):
                
                    for low_res, high_res in dataloader:

                        low_res, high_res = low_res.to(device, non_blocking=True), high_res.to(device, non_blocking=True)

                        #=======================Train Discriminator===================
                        
                        fake_image = generator(low_res).detach()
                        
                        real_output = discriminator(high_res)
                        
                        fake_output = discriminator(fake_image)
                        
                        # 1 => True image & 0 => Fake Image
                        discriminator_loss = adversarial_loss(real_output, torch.ones_like(real_output)) + adversarial_loss(fake_output, torch.zeros_like(fake_output))
                        
                        if loop_type == "Training":
                            discriminator_optimizer.zero_grad()
                                                    
                            discriminator_loss.backward()
                            
                            discriminator_optimizer.step()
                            
                    
                        #=======================Train Generator===================
                        
                        fake_image = generator(low_res)
                        
                        fake_output = discriminator(fake_image)
                        
                        generator_loss = content_loss(fake_image, high_res) + (1e-3 * adversarial_loss(fake_output, torch.ones_like(fake_output)))
                        
                        if loop_type == "Training":
                            generator_optimizer.zero_grad()
                            
                            generator_loss.backward()
                            
                            generator_optimizer.step()
                        
                            train_loss.update(generator_loss.item())
                            
                        else:
                            
                            val_loss.update(generator_loss.item())
                        
                        pbar.update(1)
                        
                        pbar.set_postfix({
                            "Mode": loop_type,
                            "Train Loss": f"{train_loss.rounded_average:.4f}" if train_loss.rounded_average > 0 else "N/A",
                            "Val Loss": f"{val_loss.rounded_average:.4f}" if val_loss.rounded_average > 0 else "N/A",
                        })

        early_stopping(val_loss = val_loss.average)
                    
        epoch_train_loss.update(train_loss.average)
        
        epoch_val_loss.update(val_loss.average)
        
        if early_stopping.early_stop:
            print(f"Early stopping triggered: No improvement observed for {early_stopping.patience} consecutive epochs.")
            JsonManager.update_model_data(model_name = model_name, updated_fields = {ModelField.NUM_EPOCHS: epoch + 1})
            break
        else:
            JsonManager.update_model_data(model_name = model_name, updated_fields = {ModelField.COMPLETION_STATUS: f"{round(((epoch + 1)/num_epochs)*100)} %"})

        
    torch.save({"architecture": "SRGAN", "scale": scale, "color_mode": mode, "invert_color_mode": invert_mode, 
                    "patch_size": patch_size, "stride": stride, "model_state_dict": generator.state_dict()}, os.path.join(output_path, model_name))    
    
    print(f"Model saved as '{output_path}'")

    train_dataset.close()
    
    val_dataset.close()
    
    JsonManager.update_model_data(model_name = model_name, updated_fields = {ModelField.COMPLETION_STATUS: "Completed", 
                                                                             ModelField.COMPLETION_TIME: int(time.time() - starting_time),
                                                                             ModelField.TRAINING_LOSSES: epoch_train_loss.all_values, 
                                                                             ModelField.VALIDATION_LOSSES: epoch_val_loss.all_values})
    
    evaluate_model(model_name = model_name, output_path = output_path, device = device, eval_file = eval_file)

def pretrained_model(model: SRGANGenerator, learning_rate: float, num_epochs: int, train_loader: DataLoader, 
                     val_loader: DataLoader, device: torch.device):
    
    early_stopping = EarlyStopping(patience = 10, delta = 0, verbose = False)
    
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    train_loss, val_loss = RunningAverage(), RunningAverage()
    
    criterion = nn.MSELoss()
    
    total_steps = num_epochs * (len(train_loader) + len(val_loader))
    
    with tqdm(total=total_steps, desc="Pretraining Generator", leave=True, dynamic_ncols=True) as pbar:

        for epoch in range(num_epochs):
            
            train_loss.reset()
            val_loss.reset()
                        
            for loop_type, dataloader in [("Training", train_loader), ("Validation", val_loader)]:
                
                model.train() if loop_type == "Training" else model.eval()
                
                with torch.set_grad_enabled(loop_type == "Training"):
                
                    for low_res, high_res in dataloader:

                        low_res, high_res = low_res.to(device, non_blocking=True), high_res.to(device, non_blocking=True)
                        
                        output = model(low_res)

                        loss = criterion(output, high_res)
                        
                        if loop_type == "Training":
                            optimizer.zero_grad()
                            
                            loss.backward()
                            
                            optimizer.step()
                        
                            train_loss.update(loss.item())
                            
                        else:
                            
                            val_loss.update(loss.item())
                        
                        pbar.set_postfix({
                            "Epoch":f"{epoch+1}/{num_epochs}",
                            "Mode": f"{loop_type}",
                            "Train Loss": f"{train_loss.rounded_average:.4f}" if train_loss.rounded_average > 0 else "N/A",
                            "Val Loss": f"{val_loss.rounded_average:.4f}" if val_loss.rounded_average > 0 else "N/A",
                        })
                        
                        pbar.update(1)

            early_stopping(val_loss = val_loss.average)
                        
            
            if early_stopping.early_stop:
                print(f"Early stopping triggered: No improvement observed for {early_stopping.patience} consecutive epochs.")
    

def evaluate_model(model_name, output_path, device, eval_file):
    
    model = SuperResolution(model_path = os.path.join(output_path, model_name))
    
    eval_dataset = H5ImagesDataset(h5_path = eval_file)
    
    eval_batch = SizeBasedImageBatch(image_sizes = eval_dataset.image_sizes, batch_size = 1, shuffle = False)

    eval_loader = DataLoader(eval_dataset, batch_sampler = eval_batch, num_workers = 1, pin_memory = True, persistent_workers = True)
    
    evaluator = ImageEvaluator()
    
    with torch.no_grad():
        with tqdm(total = len(eval_loader), desc="Evaluation", leave=True) as pbar:
            for lr, hr in eval_loader:
                
                lr, hr = lr.to(device), hr.to(device)
                
                output = model.process_image(lr)
                
                evaluator.evaluate(hr = hr, output = output)
            
                pbar.update(1)
            
    eval_dataset.close()
    
    JsonManager.update_model_data(model_name = model_name, updated_fields = {ModelField.EVAL_METRICS: evaluator.get_average_metrics()})