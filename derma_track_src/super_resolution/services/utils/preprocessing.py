import cv2
import os
import numpy as np
import h5py
from tqdm import tqdm

from super_resolution.services.utils.image_converter import ImageConverter, ImageColorConverter

def _prepare_and_add_images(image_folder: str, scale: int, mode: ImageColorConverter, hi_res_images: h5py.Group, low_res_images: h5py.Group):
    
    images = [file for file in os.listdir(image_folder) if file.endswith(('.png', '.jpg', '.jpeg'))]
    
    for count, file in enumerate(tqdm(images, desc="Creating Dataset"), start = 1):
            
        img_path = os.path.join(image_folder, file)
        
        hr = cv2.imread(img_path)  

        hr = ImageConverter.convert_image(hr, mode)
        
        lr = cv2.resize(hr, (hr.shape[1] // scale, hr.shape[0] // scale), interpolation = cv2.INTER_CUBIC)
        
        lr = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation = cv2.INTER_CUBIC)
        
        # Here we permute because the Pytorch convolutionnal layer accept only Channels X Height X Width and Open CV return it as
        # Height X Width X Channels
        lr = np.transpose(lr, (2, 0, 1)).astype(np.float32) / 255.0

        hr = np.transpose(hr, (2, 0, 1)).astype(np.float32) / 255.0
        
        low_image = low_res_images.create_dataset(f"image_{count}", data = np.array(lr), dtype = np.float32)
        
        hi_image = hi_res_images.create_dataset(f"image_{count}", data = np.array(hr), dtype = np.float32)
        
        low_image.attrs["file"], hi_image.attrs["file"] = file, file
            
            
def create_h5_image_file(input_path, scale, output_path, mode):
    
    with h5py.File(output_path, 'w') as h5_file:
        
        hi_res_images = h5_file.create_group('hi_res')
        
        low_res_images = h5_file.create_group('low_res')
        
        _prepare_and_add_images(input_path, scale, mode, hi_res_images, low_res_images)