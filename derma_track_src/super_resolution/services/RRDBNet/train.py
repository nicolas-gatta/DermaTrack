import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time
import os
from django.conf import settings

from torch import nn
from tqdm import tqdm

from super_resolution.services.RRDBNet.model import RRDBNet

from torch.utils.data.dataloader import DataLoader

from super_resolution.services.utils.running_average import RunningAverage
from super_resolution.services.utils.dataloader import H5ImagesDataset
from super_resolution.services.utils.image_evaluator import ImageEvaluator
from super_resolution.services.utils.early_stopping import EarlyStopping
from super_resolution.services.utils.batch_sampler import SizeBasedImageBatch
from super_resolution.services.utils.json_manager import JsonManager, ModelField
from super_resolution.services.utils.super_resolution import SuperResolution
from super_resolution.services.utils.prepare_dataset import dataset_exist_or_create
from super_resolution.services.utils.model_evaluation import ModelEvaluation

def pretrain_model(model_name: str, train_dataset: str, valid_dataset: str, eval_dataset: str, output_path: str, 
                mode: str, scale: int, invert_mode: str, patch_size: int, stride: int, learning_rate: float = 1e-4, 
                seed: int = 1, batch_size: int = 16, num_epochs: int = 100, num_workers: int = 8):
    """
    Pretrains an RRDBNet model using the specified datasets and training parameters.

    Args:
        model_name (str): Name to assign to the model.
        train_dataset (str): Path to the training dataset.
        valid_dataset (str): Path to the validation dataset.
        eval_dataset (str): Path to the evaluation dataset.
        output_path (str): Directory where the trained model and outputs will be saved.
        mode (str): Training mode or data processing mode.
        scale (int): Upscaling factor for super-resolution.
        invert_mode (str): Mode for inverting images or data augmentation.
        patch_size (int): Size of image patches for training.
        stride (int): Stride for extracting patches from images.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
        seed (int, optional): Random seed for reproducibility. Defaults to 1.
        batch_size (int, optional): Number of samples per training batch. Defaults to 16.
        num_epochs (int, optional): Number of training epochs. Defaults to 100.
        num_workers (int, optional): Number of worker threads for data loading. Defaults to 8.
    """
    
    JsonManager.training_results_to_json(architecture = "RRDB", stride = stride, patch_size = patch_size, resize_rule = None, 
                                                model_name = model_name, train_file = train_dataset, valid_file = valid_dataset, 
                                                eval_file = eval_dataset, mode = mode, scale = scale, learning_rate = learning_rate, seed = seed, 
                                                batch_size = batch_size, num_epochs = num_epochs, num_workers = num_workers)
        
    train_file, valid_file, eval_file = [dataset_exist_or_create(dataset = dataset, mode = mode, scale = scale, category = category, 
                                                                patch_size = patch_size, stride = stride, resize_rule = None, 
                                                                resize_to_output = False, base_dir = settings.BASE_DIR) 
                                        for dataset, category in [(train_dataset, "training"), 
                                                                (valid_dataset, "validation"), 
                                                                (eval_dataset, "evaluation")] 
                                        ]
    
    train_model(model_name = model_name, train_file = train_file, valid_file = valid_file, eval_file = eval_file, output_path = output_path, 
                mode = mode, scale = scale, invert_mode = invert_mode, patch_size = patch_size, stride = stride, learning_rate = learning_rate,
                seed = seed, batch_size = batch_size, num_epochs = num_epochs, num_workers = num_workers)


def train_model(model_name: str, train_file: str, valid_file: str, eval_file: str, output_path: str, 
                mode: str, scale: int, invert_mode: str, patch_size: int, stride: int, learning_rate: float = 1e-5, 
                seed: int = 1, batch_size: int = 16, num_epochs: int = 100, num_workers: int = 8, pretrain_model: str = None):
    """
    Trains an RRDBNet super-resolution model using the specified training, validation, and evaluation datasets.
    
    Args:
        model_name (str): Name for saving the trained model and updating metadata.
        train_file (str): Path to the HDF5 file containing training data.
        valid_file (str): Path to the HDF5 file containing validation data.
        eval_file (str): Path to the HDF5 file containing evaluation data for post-training evaluation.
        output_path (str): Directory where the trained model and related files will be saved.
        mode (str): Color mode used for training (e.g., 'RGB', 'L').
        scale (int): Upscaling factor for super-resolution.
        invert_mode (str): Specifies if color inversion is applied.
        patch_size (int): Size of image patches used for training.
        stride (int): Stride for patch extraction.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 4e-4.
        seed (int, optional): Random seed for reproducibility. Default is 1.
        batch_size (int, optional): Number of samples per batch. Default is 16.
        num_epochs (int, optional): Maximum number of training epochs. Default is 100.
        num_workers (int, optional): Number of worker processes for data loading. Default is 8.
        pretrain_model (str, optional): Path to a pretrained model to initialize weights. Default is None.
    """
    
    early_stopping = EarlyStopping(patience = 10, delta = 0, verbose = False)
    
    cudnn.benchmark = True
    
    scaler = torch.amp.GradScaler()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    train_dataset = H5ImagesDataset(train_file)
    val_dataset = H5ImagesDataset(valid_file)
    
    train_batch = SizeBasedImageBatch(image_sizes = train_dataset.image_sizes, batch_size = batch_size)
    val_batch = SizeBasedImageBatch(image_sizes = val_dataset.image_sizes, batch_size = batch_size, shuffle = False)

    train_loader = DataLoader(train_dataset, batch_sampler = train_batch, num_workers = num_workers, pin_memory = True, persistent_workers = True)
    val_loader = DataLoader(val_dataset, batch_sampler = val_batch, num_workers = num_workers, pin_memory = True, persistent_workers = True)

    model = RRDBNet(up_scale = scale).to(device)
    
    if pretrain_model:
        model.load_state_dict(torch.load(f = os.path.join(output_path, pretrain_model), map_location = device, weights_only = True)["model_state_dict"])
        
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    train_loss, val_loss = RunningAverage(), RunningAverage()
    
    epoch_train_loss, epoch_val_loss = RunningAverage(), RunningAverage()
    
    criterion = nn.L1Loss()
    
    starting_time = time.time()
    
    with tqdm(total=num_epochs * (len(train_loader) + len(val_loader)), desc="Training RRDB", leave=True, dynamic_ncols=True) as pbar:

        for epoch in range(num_epochs):
            
            train_loss.reset()
            val_loss.reset()
                        
            for loop_type, dataloader in [("Training", train_loader), ("Validation", val_loader)]:
                
                model.train() if loop_type == "Training" else model.eval()
                
                with torch.set_grad_enabled(loop_type == "Training"):
                
                    for low_res, high_res in dataloader:

                        low_res, high_res = low_res.to(device, non_blocking=True), high_res.to(device, non_blocking=True)
                        
                        with torch.autocast(device_type=device.type):
                            
                            output = model(low_res)

                            loss = criterion(output, high_res)
                        
                        if loop_type == "Training":
                            
                            optimizer.zero_grad(set_to_none = True)
                            
                            # loss.backward()
                            
                            # optimizer.step()
                            
                            scaler.scale(loss).backward()
                            
                            scaler.step(optimizer=optimizer)
                            
                            scaler.update()
                        
                            train_loss.update(loss.item())
                            
                        else:
                            
                            val_loss.update(loss.item())
                        
                        pbar.update(1)
                        
                        pbar.set_postfix({
                            "Epoch":f"{epoch+1}/{num_epochs}",
                            "Mode": f"{loop_type}",
                            "Train Loss": f"{train_loss.rounded_average}" if train_loss.rounded_average > 0 else "N/A",
                            "Val Loss": f"{val_loss.rounded_average}" if val_loss.rounded_average > 0 else "N/A",
                        })
                        
            early_stopping(val_loss = val_loss.average)
                        
            epoch_train_loss.update(train_loss.average)
            
            epoch_val_loss.update(val_loss.average)
            
            if early_stopping.early_stop:
                pbar.close()
                print(f"Early stopping triggered: No improvement observed for {early_stopping.patience} consecutive epochs.")
                JsonManager.update_model_data(model_name = model_name, updated_fields = {ModelField.NUM_EPOCHS: epoch + 1})
                break
            else:
                JsonManager.update_model_data(model_name = model_name, updated_fields = {ModelField.COMPLETION_STATUS: f"{round(((epoch + 1)/num_epochs)*100)} %"})
    
    torch.save({"architecture": "RRDBNet", "scale": scale, "color_mode": mode, "invert_color_mode": invert_mode, "need_resize": False,
                "patch_size": patch_size, "stride": stride, "multi_input": False, "model_state_dict": model.state_dict()}, os.path.join(output_path, model_name))    
    
    JsonManager.update_model_data(model_name = model_name, updated_fields = {ModelField.COMPLETION_STATUS: "Completed", 
                                                                            ModelField.COMPLETION_TIME: int(time.time() - starting_time),
                                                                            ModelField.TRAINING_LOSSES: epoch_train_loss.all_values, 
                                                                            ModelField.VALIDATION_LOSSES: epoch_val_loss.all_values})
    
    print(f"Model saved as '{os.path.join(output_path, model_name)}'")
    
    ModelEvaluation.evaluate_model(model_name = model_name, path_to_model = output_path, device = device, eval_file = eval_file)
    
    return model