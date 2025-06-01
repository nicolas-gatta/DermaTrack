import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time
import os

from torch import nn
from tqdm import tqdm

from super_resolution.services.SRGAN.discriminator_model import SRGANDiscriminator
from super_resolution.services.SRResNet.model import SRResNet
from super_resolution.services.SRGAN.loss import VGGLoss
from super_resolution.services.SRResNet.train import pretrain_model

from torch.utils.data.dataloader import DataLoader

from super_resolution.services.utils.running_average import RunningAverage
from super_resolution.services.utils.dataloader import H5ImagesDataset
from super_resolution.services.utils.early_stopping import EarlyStopping
from super_resolution.services.utils.batch_sampler import SizeBasedImageBatch
from super_resolution.services.utils.json_manager import JsonManager, ModelField
from super_resolution.services.utils.string_extractor import extract_dataset_name
from super_resolution.services.utils.model_evaluation import ModelEvaluation

def train_model(model_name: str, train_file: str, valid_file: str, eval_file: str, output_path: str, 
                mode: str, scale: int, invert_mode: str, patch_size: int, stride: int, learning_rate: float = 1e-5, 
                seed: int = 1, batch_size: int = 16, num_epochs: int = 100, num_workers: int = 8, pretrain_model: str = None):
    
    if not pretrain_model:
        pretrained_model_name = f"SRResNet_x{scale}_{mode}.pth"
        
        pretrained_model_path = os.path.join(output_path, pretrained_model_name)
        
        if not os.path.exists(pretrained_model_path):
            pretrain_model(model_name = pretrained_model_name, train_dataset = extract_dataset_name(train_file), valid_dataset = extract_dataset_name(valid_file), 
                                    eval_dataset = extract_dataset_name(eval_file), output_path = output_path, mode = mode, scale = scale, invert_mode = invert_mode, 
                                    patch_size = patch_size, stride = 0, seed = seed, batch_size = 32, num_epochs = 1, num_workers = num_workers)
                
            JsonManager.update_model_data(model_name=model_name, updated_fields={ModelField.PRETRAINED_MODEL: pretrained_model_name})
    else:
        pretrained_model_path = os.path.join(output_path, pretrain_model)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    generator = SRResNet(up_scale = scale).to(device)
    
    model_info = torch.load(f = pretrained_model_path, map_location = device, weights_only = True)
    
    generator.load_state_dict(model_info['model_state_dict'])
    
    early_stopping = EarlyStopping(patience = 10, delta = 0, verbose = False)
    
    cudnn.benchmark = True
    
    scaler = torch.amp.GradScaler()

    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    train_dataset = H5ImagesDataset(train_file, crop_size = patch_size, up_scale_factor = scale)
    val_dataset = H5ImagesDataset(valid_file, crop_size = patch_size, up_scale_factor = scale)
    
    train_batch = SizeBasedImageBatch(image_sizes = train_dataset.image_sizes, batch_size = batch_size)
    val_batch = SizeBasedImageBatch(image_sizes = val_dataset.image_sizes, batch_size = batch_size, shuffle = False)

    train_loader = DataLoader(train_dataset, batch_sampler = train_batch, num_workers = num_workers, pin_memory = True, persistent_workers = True)
    val_loader = DataLoader(val_dataset, batch_sampler = val_batch, num_workers = num_workers, pin_memory = True, persistent_workers = True)
    
    train_loss, val_loss = RunningAverage(), RunningAverage()
    
    epoch_train_loss, epoch_val_loss = RunningAverage(), RunningAverage()
    
    discriminator = SRGANDiscriminator(crop_size = patch_size).to(device)
    
    content_loss = VGGLoss().to(device)
    
    adversarial_loss = nn.BCEWithLogitsLoss()
    
    generator_optimizer = optim.Adam(generator.parameters(), lr = learning_rate)
    
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr = learning_rate)
    
    starting_time = time.time()
    
    for epoch in range(num_epochs):
        
        train_loss.reset()
        val_loss.reset()
        
        with tqdm(total = len(train_loader) + len(val_loader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=True, dynamic_ncols=True) as pbar:
            
            for loop_type, dataloader in [("Training", train_loader), ("Validation", val_loader)]:
                
                generator.train() if loop_type == "Training" else generator.eval()
                
                with torch.set_grad_enabled(loop_type == "Training"):
                
                    for low_res, high_res in dataloader:

                        low_res, high_res = low_res.to(device, non_blocking=True), high_res.to(device, non_blocking=True)
                            
                        #=======================Train Generator===================
                        
                        with torch.autocast(device_type=device.type):
                            
                            fake_image = generator(low_res)
                            
                            fake_ouput = discriminator(fake_image)
                            
                            generator_loss = content_loss(fake_image, high_res) + (1e-3 * adversarial_loss(fake_ouput, torch.ones_like(fake_ouput)))

                        
                        if loop_type == "Training":
                            generator_optimizer.zero_grad(set_to_none = True)
                            
                            # loss.backward()
                            
                            # optimizer.step()
                            
                            scaler.scale(generator_loss).backward()
                            
                            scaler.step(optimizer=generator_optimizer)
                            
                            scaler.update()
                        
                            train_loss.update(generator_loss.item())
                            
                        else:
                            
                            val_loss.update(generator_loss.item())
                        
                        pbar.update(1)
                        
                        pbar.set_postfix({
                            "Mode": loop_type,
                            "Train Loss": f"{train_loss.rounded_average:.4f}" if train_loss.rounded_average > 0 else "N/A",
                            "Val Loss": f"{val_loss.rounded_average:.4f}" if val_loss.rounded_average > 0 else "N/A",
                        })
                        
                        #=======================Train Discriminator===================
                        
                        with torch.autocast(device_type=device.type):
                            real_output = discriminator(high_res)
                            
                            fake_output = discriminator(fake_image.detach())
                            
                            # 0 => True image & 1 => Fake Image
                            discriminator_loss = (adversarial_loss(real_output, torch.ones_like(real_output)) + adversarial_loss(fake_output, torch.zeros_like(fake_output)))
                        
                        if loop_type == "Training":
                            
                            discriminator_optimizer.zero_grad(set_to_none = True)
                            
                            # loss.backward()
                            
                            # optimizer.step()
                            
                            scaler.scale(discriminator_loss).backward()
                            
                            scaler.step(optimizer=discriminator_optimizer)
                            
                            scaler.update()
                        

        early_stopping(val_loss = val_loss.average)
                    
        epoch_train_loss.update(train_loss.average)
        
        epoch_val_loss.update(val_loss.average)
        
        if early_stopping.early_stop:
            print(f"Early stopping triggered: No improvement observed for {early_stopping.patience} consecutive epochs.")
            JsonManager.update_model_data(model_name = model_name, updated_fields = {ModelField.NUM_EPOCHS: epoch + 1})
            break
        else:
            JsonManager.update_model_data(model_name = model_name, updated_fields = {ModelField.COMPLETION_STATUS: f"{round(((epoch + 1)/num_epochs)*100)} %"})

        
    torch.save({"architecture": "SRGAN", "scale": scale, "color_mode": mode, "invert_color_mode": invert_mode, "need_resize": False,
                "patch_size": patch_size, "stride": stride, "multi_input": False, "model_state_dict": generator.state_dict()}, os.path.join(output_path, model_name))    
    
    print(f"Model saved as '{os.path.join(output_path, model_name)}'")

    train_dataset.close()
    
    val_dataset.close()
    
    JsonManager.update_model_data(model_name = model_name, updated_fields = {ModelField.COMPLETION_STATUS: "Completed", 
                                                                             ModelField.COMPLETION_TIME: int(time.time() - starting_time),
                                                                             ModelField.TRAINING_LOSSES: epoch_train_loss.all_values, 
                                                                             ModelField.VALIDATION_LOSSES: epoch_val_loss.all_values})
    
    ModelEvaluation.evaluate_model(model_name = model_name, path_to_model = output_path, device = device, eval_file = eval_file)
