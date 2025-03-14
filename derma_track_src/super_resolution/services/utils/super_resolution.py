import os
import cv2
import numpy as np
import torch

from super_resolution.services.utils.image_converter import ImageColorConverter, ImageConverter
from super_resolution.services.SRCNN.model import SRCNN
from super_resolution.services.SRGAN.generator_model import SRGANGenerator
from super_resolution.services.ESRGAN.generator_model import ESRGANGenerator

class SuperResolution:
    def __init__(self, model_path):
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.mode, self.invert_mode = None, None
        
        self.model = self.__load_model(model_path)
        
        self.model.eval()
        
    
    def __load_model(self, model_path):

        model_info = torch.load(model_path)
        
        self.mode = model_info["color_mode"]
        
        self.invert_mode = model_info["invert_color_mode"]
        
        model = None
        
        match(model_info["architecture"]):
            
            case "SRCNN":
                model = SRCNN()
            
            case "SRGAN":
                model = SRGANGenerator()
            
            case "ESRGAN":
                model = ESRGANGenerator()
            
            case _:
                raise ValueError(f"Unknown architecture: {model_info['architecture']}")

        model.load_state_dict(model_info['model_state_dict'])
        
        return model 

    
    def apply_super_resolution(self, image_path, output_path, filename):
        """
        Apply the super-resolution model to an image.

        Args:
            image_source (str): The source of the image
        """
        
        image = self.__fetch_image(image_source = image_path)
        
        preprocess_image = self.__preprocess_image(image)
        
        with torch.no_grad():
            sr_image = self.model(preprocess_image)

        postprocess_image = self.__postprocess_image(sr_image)
        
        return self.save_image(image = postprocess_image, output_path = output_path, filename = filename)
    
        
    def __fetch_image(self, image_source: str):
        
        """
        Fetch an image from a local file path

        Args:
            image_source (str): The source of the image
            
        Raises:
            ValueError: Failed to load image from path
            ValueError: Invalid image source

        Returns:
            MatLike: The loaded image
        """
        if os.path.exists(image_source):
            image = cv2.imread(image_source)
            if image is None:
                raise ValueError(f"Failed to load image from path: {image_source}")
        
        else:
            raise ValueError(f"Invalid image source: {image_source}")
        
        return image
    
    def __preprocess_image(self, image) -> torch.Tensor:
        convert_image = ImageConverter.convert_image(image, ImageColorConverter[self.mode])
        tensor_image = torch.from_numpy(convert_image).permute(2, 0, 1).float() / 255
        return tensor_image.unsqueeze(0)
    
    def __postprocess_image(self, sr_image: torch.Tensor) -> np.ndarray:
        tensor_image = sr_image.squeeze(0).permute(1, 2, 0) * 255
        numpy_image = tensor_image.cpu().numpy()
        convert_image = ImageConverter.convert_image(numpy_image, ImageColorConverter[self.invert_mode])
        return convert_image
    
    def save_image(self, image, output_path="processed_images", filename="super_resolved.png"):
        
        os.makedirs(output_path, exist_ok=True)
        
        output_path = os.path.join(output_path, filename)
        
        cv2.imwrite(output_path, image)
        
        print(f"Saved image to: {output_path}")
        
        return output_path
