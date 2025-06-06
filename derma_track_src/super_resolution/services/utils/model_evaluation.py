from super_resolution.services.utils.super_resolution import SuperResolution
from super_resolution.services.utils.dataloader import H5ImagesDataset
from super_resolution.services.utils.batch_sampler import SizeBasedImageBatch
from super_resolution.services.utils.image_evaluator import ImageEvaluator
from torch.utils.data.dataloader import DataLoader
from super_resolution.services.utils.json_manager import JsonManager, ModelField
from super_resolution.services.utils.running_average import RunningAverage
from tqdm import tqdm

import os
import torch
import torch.nn.functional as F
import time


class ModelEvaluation:
    """
    Class for evaluating super-resolution models on a given dataset.
    """
    
    @staticmethod
    def evaluate_model(model_name, path_to_model, device, eval_file, eval_file_name = None, use_bicubic = False, bicubic_scale = None) -> None:
        """
        Evaluates a super-resolution model or bicubic interpolation on a given evaluation dataset.
        
        Args:
            model_name (str): Name of the model to evaluate.
            path_to_model (str): Path to the directory containing the model file.
            device (torch.device): Device on which to run the evaluation (e.g., 'cpu' or 'cuda').
            eval_file (str): Path to the HDF5 file dataset.
            eval_file_name (str, optional): Name of the evaluation file to be recorded in the model's metadata. Defaults to None.
            use_bicubic (bool, optional): If True, evaluates using bicubic interpolation instead of the model. Defaults to False.
            bicubic_scale (int or float, optional): Scale factor for bicubic interpolation. Required if use_bicubic is True.
        """
        
        if not use_bicubic:
            model = SuperResolution(model_path = os.path.join(path_to_model, model_name))
        else:
            model = SuperResolution(model_path = None, use_bicubic = True, bicubic_scale = bicubic_scale)
        
        eval_dataset = H5ImagesDataset(h5_path = eval_file)
        
        eval_batch = SizeBasedImageBatch(image_sizes = eval_dataset.image_sizes, batch_size = 1, shuffle = False)

        eval_loader = DataLoader(eval_dataset, batch_sampler = eval_batch, num_workers = 1, pin_memory = True, persistent_workers = True)
        
        evaluator = ImageEvaluator()
        
        timings = RunningAverage()
        
        with torch.no_grad():
            with tqdm(total = len(eval_loader), desc="Evaluation", leave=True, dynamic_ncols=True) as pbar:
                for lr, hr in eval_loader:
                    
                    lr, hr = lr.to(device), hr.to(device)
                    
                    start_time = time.perf_counter()
                    
                    if not model.use_bicubic and model.model_info["multi_input"]:
                        output = model.process_images(lr)
                    else:
                        output = model.process_image(lr)
                    
                    end_time = time.perf_counter()
                    
                    timings.update(end_time - start_time)
                    
                    evaluator.evaluate(hr = hr, output = output)
                
                    pbar.update(1)
                
        eval_dataset.close()
        
        updated_fields = {ModelField.EVAL_METRICS: evaluator.get_average_metrics(), ModelField.EXECUTION_TIME: timings.rounded_average}
        
        if eval_file_name != None:
            updated_fields[ModelField.EVAL_FILE] = eval_file_name
            
            JsonManager.update_model_data(model_name = model_name, updated_fields = updated_fields)