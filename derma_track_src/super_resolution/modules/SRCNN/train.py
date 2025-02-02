import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time

from torch import nn
from tqdm import tqdm
from super_resolution.modules.SRCNN.model import SRCNN
from super_resolution.modules.utils.dataloader import H5ImagesDataset
from super_resolution.modules.utils.running_average import RunningAverage
from super_resolution.modules.utils.batch_sampler import SizeBasedImageBatch
from super_resolution.modules.utils.json_manager import JsonManager, ModelField
from torch.utils.data.dataloader import DataLoader

def train_model(model_name, train_file, valid_file, eval_file, output_dir, mode, learning_rate: float = 1e-4, seed: int = 1, batch_size: int = 16, num_epochs: int = 100, num_workers: int = 8):
    
    starting_time = time.time()
    
    cudnn.benchmark = True
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    train_dataset = H5ImagesDataset(train_file)
    val_dataset = H5ImagesDataset(valid_file)
    
    train_batch = SizeBasedImageBatch(dataset = train_dataset, batch_size = batch_size)
    val_batch = SizeBasedImageBatch(dataset = val_dataset, batch_size = batch_size, shuffle = False)

    train_loader = DataLoader(train_dataset, batch_sampler = train_batch, num_workers = num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_sampler = val_batch, num_workers = num_workers, pin_memory=True)

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
                
                torch.set_grad_enabled(loop_type == "Training")
                
                for low_res, high_res in dataloader:
                    
                    low_res, high_res = low_res.to(device, non_blocking=True), high_res.to(device, non_blocking=True)

                    # Forward
                    output = model(low_res)
                    
                    loss = criterion(output, high_res)
                    
                    if loop_type == "Training":
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
                    
        epoch_train_loss.update(train_loss.average)
        epoch_val_loss.update(val_loss.average)
        
        JsonManager.update_model_data(model_name = model_name, updated_fields = {ModelField.COMPLETION_STATUS: f"{round(((epoch + 1)/num_epochs)*100)} %"})
                    
    torch.save({"architecture": "SRCNN", "color_mode": mode, "model_state_dict": model.state_dict()}, output_dir)
    
    print(f"Model saved as '{output_dir}'")

    train_dataset.close()
    
    val_dataset.close()
    
    JsonManager.update_model_data(model_name = model_name, updated_fields = {ModelField.COMPLETION_STATUS: "Completed", 
                                                                             ModelField.COMPLETION_TIME: int(time.time() - starting_time),
                                                                             ModelField.TRAINING_LOSSES: epoch_train_loss.all_values, 
                                                                             ModelField.VALIDATION_LOSSES: epoch_val_loss.all_values})
    
    evaluate_model(model_name = model_name, model = model, device = device, criterion = criterion, eval_file = eval_file)
    
def evaluate_model(model_name, model, device, criterion, eval_file):
    
    model.eval()
    total_loss, total_psnr = 0.0, 0.0
    
    eval_dataset = H5ImagesDataset(eval_file)
    
    eval_batch = SizeBasedImageBatch(dataset = eval_dataset, batch_size = 1, shuffle = False)

    eval_loader = DataLoader(eval_dataset, batch_sampler = eval_batch, pin_memory=True)
    
    num_batches = len(eval_loader)
    
    result = {}
    
    with torch.no_grad():
        for lr, hr in tqdm(eval_loader, desc="Evaluation", leave=False):
            lr, hr = lr.to(device), hr.to(device)
            
            output = model(lr)
            loss = criterion(output, hr)
            
            total_loss += loss.item()
            
            mse = torch.mean((output - hr) ** 2)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            total_psnr += psnr.item()
    
    result["MSE"] = (total_loss / num_batches)
    result["PSNR"] = (total_psnr / num_batches)
        
    eval_dataset.close()
    
    JsonManager.update_model_data(model_name = model_name, updated_fields = {ModelField.EVAL_METRICS: result})