import os
import cv2
import numpy as np
import requests
import torch

from super_resolution.services.SRCNN.model import SRCNN
from super_resolution.services.SRGAN.generator_model import SRGANGenerator
from super_resolution.services.ESRGAN.generator_model import ESRGANGenerator

class SuperResolution:
    def __init__(self, model_path):
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.mode = None
        self.model = self.load_model(model_path)
        
    
    def load_model(self, model_path):

        checkpoint  = torch.load(model_path, weights_only=True)
        
        architecture = checkpoint["architecture"]
        
        self.mode = checkpoint["mode"]
        
        match(architecture):
            
            case "SRCNN":
                return SRCNN.load_state_dict(checkpoint['model_state_dict'])
            
            case "SRGAN":
                pass
            
            case "SRGAN":
                pass
    
    def fetch_image(self, image_url):
        
        return None
    
    def apply_super_resolution(self, image):

        return None
    
    def save_image(self, image, output_dir="processed_images", filename="super_resolved.png"):
        
        return None
