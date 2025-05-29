import os
import cv2
import numpy as np
import torch

from super_resolution.services.utils.image_converter import ImageColorConverter, ImageConverter
from super_resolution.services.SRCNN.model import SRCNN
from super_resolution.services.SRResNet.model import SRResNet
from super_resolution.services.RRDBNet.model import RRDBNet
from basicsr.archs.edvr_arch import EDVR

class SuperResolution:
    def __init__(self, model_path: str):
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                
        self.model, self.model_info = self.__load_model(model_path)
        
        self.model.to(self.device)
        
        self.model.eval()
        
    
    def __load_model(self, model_path):

        model_info = torch.load(model_path, weights_only = True)
        
        model = None
        
        match(model_info["architecture"]):
            
            case "SRCNN":
                model = SRCNN()
            
            case "SRGAN" | "SRResNet":
                model = SRResNet(up_scale = model_info["scale"])
            
            case "ESRGAN" | "RRDBNet":
                model = RRDBNet(up_scale = model_info["scale"])
                
            case "EDVR":
                model = EDVR(center_frame_idx = 2)
            
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
        
        if self.model_info["multi_input"]:
            postprocess_image = self.process_images(images = image)
        else:
            postprocess_image = self.process_image(image = image)
        
        return self.save_image(image = postprocess_image, output_path = output_path, filename = filename)
    
    def process_image(self, image: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        
        postprocess_required = True
        
        if isinstance(image, np.ndarray):  
            preprocess_image = self.__preprocess_image(image)
        else:
            preprocess_image = image
            postprocess_required = False

        height, width = preprocess_image.shape[2:]
        quadrant_image_parts = []
        
        if height >= 1600 or width >= 1200:
            quadrant_image_parts = self.__split_image(height = height, width = width, img = preprocess_image)
            sr_part = []
            with torch.no_grad():
                for part in quadrant_image_parts:
                    sr_part.append(self.model(part))
                
                sr_image = self.__merge_images(*sr_part)
        else:
            with torch.no_grad():
                sr_image = self.model(preprocess_image)


        sr_image = torch.clamp(sr_image, 0.0, 1.0)
        return self.__postprocess_image(sr_image) if postprocess_required else sr_image
    
    def process_images(self, images: list[np.ndarray] | list[torch.Tensor]) ->  list[np.ndarray] | list[torch.Tensor]:
        
        postprocess_required = True
        preprocess_images = []

        if isinstance(images[0], np.ndarray):  
            preprocess_images = [self.__preprocess_image(image) for image in images]
 
        else:
            preprocess_images = images
            postprocess_required = False

        height, width = preprocess_images.shape[3:]
        quadrant_images_seq = [[] for _ in range(len(preprocess_images))]
        
        if height >= 1600 or width >= 1200:
            for index, image in enumerate(preprocess_images):
                quadrant_images_seq = self.__split_image(height = height, width = width, img = image)
                sr_part = []
                with torch.no_grad():
                    for index, seq in enumerate(quadrant_images_seq):
                        tensor = torch.cat(seq, dim=0).to(self.device)
                        sr_part.append(self.model(tensor))
                    
                    sr_image = self.__merge_images(*sr_part)
        
        else:
            with torch.no_grad():
                sr_image = self.model(preprocess_images)


        sr_image = torch.clamp(sr_image, 0.0, 1.0)
        return self.__postprocess_image(sr_image) if postprocess_required else sr_image
    
    def __split_image(self, height, width, img: torch.Tensor, overlapping = 16):
        middle_height, middle_width = height // 2, width // 2
        return [
            img[:, :, :middle_height + overlapping, :middle_width + overlapping],   # top-left
            img[:, :, :middle_height + overlapping, middle_width - overlapping:],   # top-right
            img[:, :, middle_height - overlapping:, :middle_width + overlapping],   # bottom-left
            img[:, :, middle_height - overlapping:, middle_width - overlapping:]    # bottom-right
        ]
        
    def __merge_images(self, top_left: torch.Tensor, top_right: torch.Tensor, bottom_left: torch.Tensor, bottom_right: torch.Tensor, overlapping = 16):
        
        overlapping = overlapping * self.model_info["scale"] if not self.model_info["need_resize"] else overlapping
        top_left = top_left[:, :, :-overlapping, :-overlapping]
        top_right = top_right[:, :, :-overlapping, overlapping:]
        bottom_left = bottom_left[:, :, overlapping:, :-overlapping]
        bottom_right = bottom_right[:, :, overlapping:, overlapping:]
        top = torch.cat((top_left, top_right), dim = 3)
        bottom = torch.cat((bottom_left, bottom_right), dim = 3)
        return torch.cat((top, bottom), dim = 2)

        
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
        if self.model_info["need_resize"]:
            image = cv2.resize(image, (image.shape[1] * self.model_info["scale"], image.shape[0] * self.model_info["scale"]))
        convert_image = ImageConverter.convert_image(image, ImageColorConverter[self.model_info["color_mode"]])
        tensor_image = torch.from_numpy(convert_image).permute(2, 0, 1).float() / 255
        return tensor_image.unsqueeze(0).to(self.device)
    
    def __postprocess_image(self, sr_image: torch.Tensor) -> np.ndarray:
        tensor_image = sr_image.squeeze(0).permute(1, 2, 0) * 255
        numpy_image = tensor_image.cpu().detach().numpy().astype(np.uint8)
        return ImageConverter.convert_image(numpy_image, ImageColorConverter[self.model_info["invert_color_mode"]])
    
    def save_image(self, image, output_path="processed_images", filename="super_resolved.png"):
        
        os.makedirs(output_path, exist_ok=True)
        
        output_path = os.path.join(output_path, filename)
        
        cv2.imwrite(output_path, image)
        
        print(f"Saved image to: {output_path}")
        
        return output_path
