import json
import os

from datetime import datetime
from enum import Enum
from super_resolution.services.utils.prepare_dataset import ResizeRule

from django.conf import settings

class ModelField(str, Enum):
    ARCHITECTURE = "architecture"
    TRAIN_FILE = "train_file"
    VALID_FILE = "valid_file"
    EVAL_FILE = "eval_file"
    SCALE = "scale"
    MODE = "mode"
    LEARNING_RATE = "learning_rate"
    SEED = "seed"
    BATCH_SIZE = "batch_size"
    NUM_EPOCHS = "num_epochs"
    NUM_WORKERS = "num_workers"
    VALIDATION_LOSSES = "validation_losses"
    TRAINING_LOSSES = "training_losses"
    COMPLETION_TIME = "completion_time"
    COMPLETION_STATUS = "completion_status"
    TIMESTAMP = "timestamp"
    EVAL_METRICS = "eval_metrics"
    STRIDE = "stride"
    PATCH_SIZE = "patch_size"
    RESIZE_RULE = "resize_rule"
    PRETRAINED_MODEL = "pretrained_model"
    EXECUTION_TIME = "execution_time"
    
class JsonManager:
    
    _output_file = os.path.join(settings.BASE_DIR, "super_resolution", "static", "data", "training_results.json")

    @staticmethod
    def training_results_to_json(architecture: str, stride: int, patch_size: int, resize_rule: ResizeRule, model_name: str, train_file: str, valid_file: str, eval_file: str, 
                                mode: str, scale: int, learning_rate: float, seed: int, batch_size: int, num_epochs: int, num_workers: int, pretrain_model: str = None):
        """
        Save or update the training results in a JSON file using `model_name` 
        as the key.
        """
        model_data = {
            ModelField.ARCHITECTURE: architecture,
            ModelField.PRETRAINED_MODEL: pretrain_model,
            ModelField.STRIDE: stride, 
            ModelField.PATCH_SIZE: patch_size,
            ModelField.RESIZE_RULE: resize_rule,
            ModelField.TRAIN_FILE: train_file,
            ModelField.VALID_FILE: valid_file,
            ModelField.EVAL_FILE: eval_file,
            ModelField.SCALE: scale,
            ModelField.MODE: mode, 
            ModelField.LEARNING_RATE: learning_rate,
            ModelField.SEED: seed,
            ModelField.BATCH_SIZE: batch_size,
            ModelField.NUM_EPOCHS: num_epochs,
            ModelField.NUM_WORKERS: num_workers,
            ModelField.VALIDATION_LOSSES: [],
            ModelField.TRAINING_LOSSES: [],
            ModelField.COMPLETION_TIME: 0,
            ModelField.COMPLETION_STATUS: "0 %",
            ModelField.TIMESTAMP: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ModelField.EVAL_METRICS: {},
            ModelField.EXECUTION_TIME: None
        }
        
        existing_data = {}
        
        if os.path.exists(JsonManager._output_file):
            try:
                with open(JsonManager._output_file, "r") as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = {}
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}

        existing_data[model_name] = model_data

        with open(JsonManager._output_file, "w") as f:
            json.dump(existing_data, f, indent = 4)
        
        return model_name
    
    @staticmethod
    def load_training_results():
        """
        Load all variables from a JSON file and return them as a dictionary.
        
        If the file contains multiple models, they will all be returned in the dict.
        """
        
        try:
            with open(JsonManager._output_file, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
        return data
    
    @staticmethod
    def update_model_data(model_name: str, updated_fields: dict):
        """
        Update specific fields of a model in the JSON file.

        Args:
            model_name (str): The name of the model to update.
            updated_fields (dict): A dictionary containing the fields to update and their new values.
        """
        
        if not os.path.exists(JsonManager._output_file):
            raise FileNotFoundError("The training results file does not exist.")
        
        try:
            with open(JsonManager._output_file, "r") as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            raise ValueError("The training results file is invalid or cannot be read.")
        
        if model_name not in existing_data:
            raise KeyError(f"Model '{model_name}' does not exist in the training results.")
        
        for key, value in updated_fields.items():
            if key in existing_data[model_name]:
                existing_data[model_name][key] = value
            else:
                raise KeyError(f"Field '{key}' does not exist in the model data.")
        
        with open(JsonManager._output_file, "w") as f:
            json.dump(existing_data, f, indent=4)
            
    def copy_model_data_and_update(source_model_name: str, destination_model_name: str, updated_fields: dict = None):
        
        if not os.path.exists(JsonManager._output_file):
            raise FileNotFoundError("The training results file does not exist.")
    
        try:
            with open(JsonManager._output_file, "r") as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            raise ValueError("The training results file is invalid or cannot be read.")
        
        if source_model_name not in existing_data:
            raise KeyError(f"Model '{source_model_name}' does not exist in the training results.")
        
        new_model_data = dict(existing_data[source_model_name])
        
        existing_data[destination_model_name] = new_model_data
    
        with open(JsonManager._output_file, "w") as f:
            json.dump(existing_data, f, indent=4)
            
        if updated_fields != None:
            JsonManager.update_model_data(model_name = destination_model_name, updated_fields = updated_fields)
