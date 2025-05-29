from super_resolution.services.utils.super_resolution import SuperResolution
from super_resolution.services.utils.dataloader import H5ImagesDataset
from super_resolution.services.utils.batch_sampler import SizeBasedImageBatch
from super_resolution.services.utils.image_evaluator import ImageEvaluator
from torch.utils.data.dataloader import DataLoader
from basicsr.archs.edvr_arch import EDVR
from super_resolution.services.utils.json_manager import JsonManager, ModelField
from super_resolution.services.utils.running_average import RunningAverage
from tqdm import tqdm

import os
import torch
import time


class ModelEvaluation:
    
    @staticmethod
    def evaluate_model(model_name, path_to_model, device, eval_file, eval_file_name = None):
    
        model = SuperResolution(model_path = os.path.join(path_to_model, model_name))
        
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
                     
                    if model.model_info["multi_input"]:
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