import os
import cv2
import numpy as np
import torch

from super_resolution.services.utils.image_converter import ImageColorConverter, ImageConverter
from super_resolution.services.SRCNN.model import SRCNN
from super_resolution.services.SRGAN.generator_model import SRGANGenerator
from super_resolution.services.ESRGAN.generator_model import ESRGANGenerator

class SuperResolution:
    def __init__(self, model_path: str):
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                
        self.model, self.model_info = self.__load_model(model_path)
        
        self.model.to(self.device)
        
        self.model.eval()
        
    
    def __load_model(self, model_path):

        model_info = torch.load(model_path, weights_only=True)
        
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
        
        return model, model_info

    
    def apply_super_resolution(self, image_path, output_path, filename):
        """
        Apply the super-resolution model to an image.

        Args:
            image_source (str): The source of the image
        """
        
        image = self.__fetch_image(image_source = image_path)
        
        postprocess_image = self.process_image(image = image)
        
        return self.save_image(image = postprocess_image, output_path = output_path, filename = filename)
    
    def process_image(self, image: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        
        postprocess_required = True
        
        if isinstance(image, np.ndarray):  
            preprocess_image = self.__preprocess_image(image)
        else:
            preprocess_image = image
            postprocess_required = False
        
        if isinstance(self.model, SRCNN) and self.model_info["stride"] != None and self.model_info["patch_size"] != None:
            sr_image = self.__srcnn_special_processing(image_tensor = preprocess_image)
        else:
            with torch.no_grad():
                sr_image = self.model(preprocess_image)

        return self.__postprocess_image(sr_image) if postprocess_required else sr_image
        
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
    
    def __preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        convert_image = ImageConverter.convert_image(image, ImageColorConverter[self.model_info["color_mode"]])
        tensor_image = torch.from_numpy(convert_image).permute(2, 0, 1).float() / 255
        return tensor_image.unsqueeze(0).to(self.device)
    
    def __postprocess_image(self, sr_image: torch.Tensor) -> np.ndarray:
        clamp_sr_image = torch.clamp(sr_image, 0.0, 1.0)
        tensor_image = clamp_sr_image.squeeze(0).permute(1, 2, 0) * 255
        numpy_image = tensor_image.cpu().detach().numpy().astype(np.uint8)
        convert_image = ImageConverter.convert_image(numpy_image, ImageColorConverter[self.model_info["invert_color_mode"]])
        return convert_image
    
    def __srcnn_special_processing(self, image_tensor: torch.Tensor) -> torch.Tensor:
        
        _, c, h, w = image_tensor.shape
        output = torch.zeros((1, c, h, w), device = self.device)
        weight = torch.zeros((1, c, h, w), device = self.device)

        for y in range(0, h - self.model_info["patch_size"] + 1, self.model_info["stride"]):
            for x in range(0, w - self.model_info["patch_size"] + 1, self.model_info["stride"]):
                patch = image_tensor[:, :, y:y + self.model_info["patch_size"], x:x + self.model_info["patch_size"]]
                with torch.no_grad():
                    sr_patch = self.model(patch)

                _, _, ph, pw = sr_patch.shape
                output[:, :, y:y + ph, x:x + pw] += sr_patch
                weight[:, :, y:y + ph, x:x + pw] += 1

        # Avoid division by zero
        weight[weight == 0] = 1
        result = output / weight
        
        return result
    
    def save_image(self, image, output_path="processed_images", filename="super_resolved.png"):
        
        os.makedirs(output_path, exist_ok=True)
        
        output_path = os.path.join(output_path, filename)
        
        cv2.imwrite(output_path, image)
        
        print(f"Saved image to: {output_path}")
        
        return output_path
