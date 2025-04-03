import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time
import os

from torch import nn
from tqdm import tqdm
from super_resolution.services.SRCNN.model import SRCNN
from super_resolution.services.utils.dataloader import H5ImagesDataset
from super_resolution.services.utils.running_average import RunningAverage
from super_resolution.services.utils.batch_sampler import SizeBasedImageBatch
from super_resolution.services.utils.json_manager import JsonManager, ModelField
from super_resolution.services.utils.image_evaluator import ImageEvaluator
from super_resolution.services.utils.early_stopping import EarlyStopping

from torch.utils.data.dataloader import DataLoader

def train_model(model_name: str, train_file: str, valid_file: str, eval_file: str, output_path: str, 
                mode: str, scale: int, invert_mode: str, learning_rate: float = 1e-4, seed: int = 1, 
                batch_size: int = 16, num_epochs: int = 100, num_workers: int = 8):
    
    early_stopping = EarlyStopping(patience = 10, delta = 0, verbose = False)
    
    starting_time = time.time()
    
    cudnn.benchmark = True
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    train_dataset = H5ImagesDataset(train_file)
    val_dataset = H5ImagesDataset(valid_file)
    
    train_batch = SizeBasedImageBatch(image_sizes = train_dataset.image_sizes, batch_size = batch_size)
    val_batch = SizeBasedImageBatch(image_sizes = val_dataset.image_sizes, batch_size = batch_size, shuffle = False)

    train_loader = DataLoader(train_dataset, batch_sampler = train_batch, num_workers = num_workers, pin_memory=True, persistent_workers = True)
    val_loader = DataLoader(val_dataset, batch_sampler = val_batch, num_workers = num_workers, pin_memory=True, persistent_workers = True)

    model = SRCNN().to(device)
    
    criterion = nn.MSELoss()  
    
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': learning_rate * 0.1}
    ], lr=learning_rate)
    
    train_loss, val_loss = RunningAverage(), RunningAverage()
    
    epoch_train_loss, epoch_val_loss = RunningAverage(), RunningAverage()
            
    for epoch in range(num_epochs):
        
        train_loss.reset()
        
        val_loss.reset()
        
        with tqdm(total = len(train_loader) + len(val_loader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=True) as pbar:
            
            for loop_type, dataloader in [("Training", train_loader), ("Validation", val_loader)]:
                
                model.train() if loop_type == "Training" else model.eval()
                
                with torch.set_grad_enabled(loop_type == "Training"):
                
                    for low_res, high_res in dataloader:
                        
                        low_res, high_res = low_res.to(device, non_blocking=True), high_res.to(device, non_blocking=True)
                        
                        if loop_type == "Training":
                            
                            optimizer.zero_grad()
                            
                            output = model(low_res)
                        
                            loss = criterion(output, high_res)
                        
                            loss.backward()
                            
                            optimizer.step()
                            
                            train_loss.update(loss.item())
                        
                        else:
                            
                            output = model(low_res)
                        
                            loss = criterion(output, high_res)
                            
                            val_loss.update(loss.item())
                        
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

        
    torch.save({"architecture": "SRCNN", "scale": scale, "color_mode": mode, "invert_color_mode": invert_mode, "model_state_dict": model.state_dict()}, os.path.join(output_path, model_name))
    
    print(f"Model saved as '{output_path}'")

    train_dataset.close()
    
    val_dataset.close()
    
    JsonManager.update_model_data(model_name = model_name, updated_fields = {ModelField.COMPLETION_STATUS: "Completed", 
                                                                             ModelField.COMPLETION_TIME: int(time.time() - starting_time),
                                                                             ModelField.TRAINING_LOSSES: epoch_train_loss.all_values, 
                                                                             ModelField.VALIDATION_LOSSES: epoch_val_loss.all_values})
    
    evaluate_model(model_name = model_name, model = model, device = device, eval_file = eval_file)
    
def evaluate_model(model_name, model, device, eval_file):
    
    model.eval()
    
    eval_dataset = H5ImagesDataset(eval_file)
    
    eval_batch = SizeBasedImageBatch(image_sizes = eval_dataset.image_sizes, batch_size = 1, shuffle = False)

    eval_loader = DataLoader(eval_dataset, batch_sampler = eval_batch, num_workers = 1, pin_memory = True, persistent_workers = True)
    
    evaluator = ImageEvaluator()
    
    with torch.no_grad():
        with tqdm(total = len(eval_loader), desc="Evaluation", leave=True) as pbar:
            for lr, hr in eval_loader:
                
                lr, hr = lr.to(device), hr.to(device)
                
                output = model(lr)
                
                evaluator.evaluate(hr = hr, output = output)
            
                pbar.update(1)
            
    eval_dataset.close()
    
    JsonManager.update_model_data(model_name = model_name, updated_fields = {ModelField.EVAL_METRICS: evaluator.get_average_metrics()})