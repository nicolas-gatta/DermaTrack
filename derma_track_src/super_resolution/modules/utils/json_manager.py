import json
import os
from datetime import datetime
from utils.path_finder import PathFinder

class JsonManager:
    
    _output_file = PathFinder.get_complet_path(f"super_resolution/static/data/training_results.json")

    @staticmethod
    def training_results_to_json(architecture, model_name: str, train_file: str, eval_file: str, learning_rate: float, seed: int, batch_size: int, num_epochs: int, 
                                num_workers: int, validation_losses: list, training_losses: list, time_needed: float):
        """
        Save or update the training results in a JSON file using `model_name` 
        as the key.
        """
        model_data = {
            "architecture": architecture,
            "train_file": train_file,
            "eval_file": eval_file,
            "learning_rate": learning_rate,
            "seed": seed,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "num_workers": num_workers,
            "validation_losses": validation_losses,
            "training_losses": training_losses,
            "time_needed": time_needed,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        existing_data = {}
        if os.path.exists(JsonManager._output_file):
            try:
                with open(JsonManager._output_file, "r") as f:
                    existing_data = json.load(f)
                # Ensure it's a dictionary; otherwise reset.
                if not isinstance(existing_data, dict):
                    existing_data = {}
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}

        if model_name in existing_data:
            suffix = 1
            new_name = f"{model_name}_{suffix}"
            while new_name in existing_data:
                suffix += 1
                new_name = f"{model_name}_{suffix}"
            model_name = new_name
            

        # Add or update the model entry in the data
        existing_data[model_name] = model_data

        # Write the updated data to the JSON file
        with open(JsonManager._output_file, "w") as f:
            json.dump(existing_data, f, indent=4)
    
    @staticmethod
    def load_training_results():
        """
        Load all variables from a JSON file and return them as a dictionary.
        
        If the file contains multiple models, they will all be returned in the dict.
        """
        with open(JsonManager._output_file, 'r') as f:
            data = json.load(f)
        return data