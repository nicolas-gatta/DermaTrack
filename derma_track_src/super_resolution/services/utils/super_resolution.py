import os
import cv2
import numpy as np
import torch

from super_resolution.services.utils.image_converter import ImageColorConverter, ImageConverter
from super_resolution.services.SRCNN.model import SRCNN
from super_resolution.services.SRResNet.model import SRResNet
from super_resolution.services.RRDBNet.model import RRDBNet
from basicsr.archs.edvr_arch import EDVR
from image_encryption.services.advanced_encrypted_standard import AES
import torch.nn.functional as F

class SuperResolution:
    """
    Class to do the Super Solution + Processing
    """
    
    def __init__(self, model_path: str, use_bicubic: bool = False, bicubic_scale: int = None):
        
        self.use_bicubic = use_bicubic
            
        self.bicubic_scale = bicubic_scale
          
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
        if not use_bicubic:
                    
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            self.use_bicubic = None
            
            self.bicubic_scale = None
            
            self.path = model_path

                    
            self.model, self.model_info = self.__load_model(model_path)
            
            self.model.to(self.device)
            
            self.model.eval()
            
        else:
            self.model_info = dict()
            self.model_info["color_mode"] = ImageColorConverter.BGR2RGB.name
            self.model_info["invert_color_mode"] = ImageColorConverter.RGB2BGR.name
        
    
    def __load_model(self, model_path: str) -> tuple:
        """
        Load the correct model base on the model_path

        Args:
            model_path (str): The path to the selected model

        Raises:
            ValueError: If architecture is unknown

        Returns:
            tuple: The model and the model_information
        """
        
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

    
    def apply_super_resolution(self, image_path, output_path, filename, folder_path: str = None, is_encrypted: bool = False) -> str:
        """
        Apply the super-resolution model to an image.

        Args:
            image_source (str): The source of the image
            output_path (str): Directory where the output image will be saved.
            filename (str): Name of the output file.
            folder_path (str, optional): Path to a folder containing multiple input images. Defaults to None.
            is_encrypted (bool, optional): Indicates whether the input image(s) are encrypted. Defaults to False.
            
        Returns:
            str: Path to the saved super-resolved image.
        """
        
        if not self.use_bicubic and self.model_info["multi_input"]:
            postprocess_image = self.process_images(images = self.__fetch_image(image_source = folder_path, is_encrypted = is_encrypted))
        else:
            postprocess_image = self.process_image(image = self.__fetch_image(image_source = image_path, is_encrypted = is_encrypted))
        
        return self.save_image(image = postprocess_image, output_path = output_path, filename = filename, is_encrypted = is_encrypted)
    
    def process_image(self, image: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """
        Processes an input image using a super-resolution model.
        
        Args:
            image (np.ndarray | torch.Tensor): The input image to process. Can be a NumPy array or a PyTorch tensor.
            
        Returns:
            np.ndarray | torch.Tensor: The processed image, either as a NumPy array (if input was a NumPy array)
            or a PyTorch tensor (if input was a tensor).
        """
        
        postprocess_required = True
        
        if isinstance(image, np.ndarray):  
            preprocess_image = self.__preprocess_image(image)
        else:
            preprocess_image = image
            postprocess_required = False

        if self.use_bicubic:
            sr_image = F.interpolate(preprocess_image, scale_factor = self.bicubic_scale, mode="bicubic", align_corners=False)
        else:
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
        """
        Processes an input images using a super-resolution model.
        
        Args:
            image (list(np.ndarray) | list(torch.Tensor)): The input image to process. Can be a NumPy array or a PyTorch tensor.
            
        Returns:
            list(np.ndarray) | list(torch.Tensor: The processed image, either as a NumPy array (if input was a NumPy array)
            or a PyTorch tensor (if input was a tensor).
        """
        
        postprocess_required = True
        preprocess_images = []

        if isinstance(images[0], np.ndarray):  
            preprocess_images = torch.stack([self.__preprocess_image(image).squeeze(0) for image in images]).unsqueeze(0).to(self.device)
 
        else:
            preprocess_images = images
            postprocess_required = False
        
        height, width = preprocess_images.shape[3:]
        quadrant_images_seq = [[] for _ in range(len(preprocess_images))]
        
        if height >= 1600 or width >= 1200:
            for image in preprocess_images:
                quadrant_images_seq = self.__split_image(height = height, width = width, img = image)
                sr_part = []
                with torch.no_grad():
                    for seq in quadrant_images_seq:
                        tensor = seq.unsqueeze(0).to(self.device)
                        sr_part.append(self.model(tensor))
                    
                    sr_image = self.__merge_images(*sr_part)
        
        else:
            with torch.no_grad():
                sr_image = self.model(preprocess_images)


        sr_image = torch.clamp(sr_image, 0.0, 1.0)
        return self.__postprocess_image(sr_image) if postprocess_required else sr_image
    
    def __split_image(self, height: int, width: int, img: torch.Tensor, overlapping: int = 16) -> list:
        """
        Splits an input image tensor into four overlapping quadrants.
        
        Args:
            height (int): The height of the input image.
            width (int): The width of the input image.
            img (torch.Tensor): The input image tensor of shape (N, C, H, W).
            overlapping (int, optional): The number of pixels by which adjacent quadrants overlap. Default is 16.
            
        Returns:
            list of torch.Tensor: A list containing four tensors corresponding to the overlapping quadrants of the input image.
        """
        
        middle_height, middle_width = height // 2, width // 2
        return [
            img[:, :, :middle_height + overlapping, :middle_width + overlapping],   # top-left
            img[:, :, :middle_height + overlapping, middle_width - overlapping:],   # top-right
            img[:, :, middle_height - overlapping:, :middle_width + overlapping],   # bottom-left
            img[:, :, middle_height - overlapping:, middle_width - overlapping:]    # bottom-right
        ]
        
    def __merge_images(self, top_left: torch.Tensor, top_right: torch.Tensor, bottom_left: torch.Tensor, bottom_right: torch.Tensor, overlapping = 16) -> torch.Tensor:
        """
        Merges four image tensors (top-left, top-right, bottom-left, bottom-right) into a single image tensor.
        
        Args:
            top_left (torch.Tensor): The top-left image tensor.
            top_right (torch.Tensor): The top-right image tensor.
            bottom_left (torch.Tensor): The bottom-left image tensor.
            bottom_right (torch.Tensor): The bottom-right image tensor.
            overlapping (int, optional): The number of overlapping pixels between adjacent image tiles. Default is 16.
                
        Returns:
            torch.Tensor: The merged image tensor with overlapping regions removed.
        """
        
        overlapping = overlapping * self.model_info["scale"] if not self.model_info["need_resize"] else overlapping
        top_left = top_left[:, :, :-overlapping, :-overlapping]
        top_right = top_right[:, :, :-overlapping, overlapping:]
        bottom_left = bottom_left[:, :, overlapping:, :-overlapping]
        bottom_right = bottom_right[:, :, overlapping:, overlapping:]
        top = torch.cat((top_left, top_right), dim = 3)
        bottom = torch.cat((bottom_left, bottom_right), dim = 3)
        return torch.cat((top, bottom), dim = 2)

        
    def __fetch_image(self, image_source: str, is_encrypted: bool = False)  -> np.ndarray | list:
        """
        Fetch an image from a local file path

        Args:
            image_source (str): The source of the image or a folder containing multiple image
            
        Raises:
            ValueError: Failed to load image from path
            ValueError: Invalid image source

        Returns:
             np.ndarray | list[np.ndarray]: The loaded image(s)
        """
        
        images = []
        if os.path.exists(image_source):
            if os.path.isdir(image_source):
                for file in os.listdir(image_source):
                    images.append(self.__decrypt_or_get_image(file_path = os.path.join(image_source, file), is_encrypted = is_encrypted))
                return images
            
            else:
                return self.__decrypt_or_get_image(file_path = image_source, is_encrypted = is_encrypted)
        else:
            raise ValueError(f"Invalid image source: {image_source}")
    
    def __decrypt_or_get_image(self, file_path: str, is_encrypted: bool) -> np.ndarray:
        """
        Loads an image from the specified file path, decrypting it if necessary.
        
        Args:
            file_path (str): The path to the image file.
            is_encrypted (bool): Flag indicating whether the image file is encrypted.
            
        Returns:
            numpy.ndarray: The loaded image as a NumPy array in BGR format.
            
        Raises:
            ValueError: If the image cannot be loaded from the given path.
        """
        
        if is_encrypted:
            with open(file_path, "rb") as file:
                decrypted_data = AES.decrypt_message(file.read())
                image_array = np.frombuffer(decrypted_data, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError(f"Failed to load image from path: {file_path}")
        return image
    
    def __preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocesses an input image for super-resolution inference.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            
        Returns:
            torch.Tensor: The preprocessed image as a 4D tensor (batch, channels, height, width).
        """
        
        if not self.use_bicubic and self.model_info["need_resize"]:
            image = cv2.resize(image, (image.shape[1] * self.model_info["scale"], image.shape[0] * self.model_info["scale"]))
        convert_image = ImageConverter.convert_image(image, ImageColorConverter[self.model_info["color_mode"]])
        tensor_image = torch.from_numpy(convert_image).permute(2, 0, 1).float() / 255
        return tensor_image.unsqueeze(0).to(self.device)
    
    def __postprocess_image(self, sr_image: torch.Tensor) -> np.ndarray:
        """
        Preprocesses an input image for super-resolution inference.

        Args:
            image (torch.Tensor): The input image as a torch tensor.
            
        Returns:
            np.ndarray: The preprocessed image as an np.ndarray
        """
        
        tensor_image = sr_image.squeeze(0).permute(1, 2, 0) * 255
        numpy_image = tensor_image.cpu().detach().numpy().astype(np.uint8)
        return ImageConverter.convert_image(numpy_image, ImageColorConverter[self.model_info["invert_color_mode"]])
    
    def save_image(self, image, output_path: str="processed_images", filename: str="super_resolved.png", is_encrypted: bool = False) -> tuple:
        """
        Saves an image to the specified output path, with optional encryption.
        
        Args:
            image (numpy.ndarray): The image to be saved.
            output_path (str, optional): Directory where the image will be saved. Defaults to "processed_images".
            filename (str, optional): Name of the output image file. Defaults to "super_resolved.png".
            is_encrypted (bool, optional): If True, the image will be encrypted before saving. Defaults to False.
            
        Returns:
            tuple: A tuple containing ouput_path, height and width
        """
        
        os.makedirs(output_path, exist_ok=True)
        
        output_path = os.path.join(output_path, filename)
                
        height, width = image.shape[:2]
        
        if is_encrypted:
            with open(output_path, "wb") as im:
                im.write(AES.encrypt_message((cv2.imencode('.png', image)[1]).tobytes()))
        else:
            cv2.imwrite(output_path, image)

        return output_path, height, width 
