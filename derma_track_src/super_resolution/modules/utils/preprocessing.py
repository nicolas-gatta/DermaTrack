import cv2
import os
import numpy as np
import h5py
import re

from super_resolution.modules.utils.image_converter import ImageConverter, ImageColorConverter

def _prepare_and_add_images(image_folder: str, scale: int, mode: ImageColorConverter, hi_res_images: h5py.Group, low_res_images: h5py.Group):

    invert_mode = re.sub(r"^(.*)2(.*)$", r"\2here\1", mode.name).replace("here","2")
    
    print(image_folder, scale, mode)
    for count, file in enumerate(os.listdir(image_folder), start = 1):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            
            img_path = os.path.join(image_folder, file)
            
            hr = cv2.imread(img_path)  

            hr = ImageConverter.convert_image(hr, mode)
            
            lr = cv2.resize(hr, (hr.shape[1] // scale, hr.shape[0] // scale), interpolation = cv2.INTER_CUBIC)
            lr = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation = cv2.INTER_CUBIC)
        
            low_image = low_res_images.create_dataset(f"image_{count:03}", data = np.array(lr), dtype= np.uint8)
            
            hi_image = hi_res_images.create_dataset(f"image_{count:03}", data = np.array(hr), dtype= np.uint8)
            
            low_image.attrs["file"], hi_image.attrs["file"] = file, file
            
            low_image.attrs["mode"], hi_image.attrs["mode"] = mode.name, mode.name
            
            low_image.attrs["invert_mode"], hi_image.attrs["invert_mode"] = invert_mode, invert_mode
            
            
def create_h5_image_file(input_path, scale, output_path, mode):
    
    with h5py.File(output_path, 'w') as h5_file:
        
        hi_res_images = h5_file.create_group('hi_res')
        
        low_res_images = h5_file.create_group('low_res')
        
        _prepare_and_add_images(input_path, scale, mode, hi_res_images, low_res_images)
        
    print(f"File created sucessfully at the path: {output_path}")