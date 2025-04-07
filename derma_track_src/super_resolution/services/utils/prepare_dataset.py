import cv2
import os
import numpy as np
import h5py

from tqdm import tqdm

from super_resolution.services.utils.image_converter import ImageConverter, ImageColorConverter

from enum import Enum

class ResizeRule(str, Enum):
    BIGGEST = "max"
    SMALLEST = "min"


def __prepare_and_add_images(image_folder: str, scale: int, mode: ImageColorConverter, hi_res_images: h5py.Group, low_res_images: h5py.Group, 
                             patch_size: int, stride: int, resize_rule: ResizeRule, preprocessing_required: bool = True, resize_to_output: bool = True):
    
    images_file_name = [file for file in os.listdir(image_folder) if file.endswith(('.png', '.jpg', '.jpeg'))]
    
    target_size = None
    
    count = 1
    
    if resize_rule != None:
        target_size = __get_extreme_image_size(images_file_name = images_file_name, image_folder = image_folder, resize_rule = resize_rule)
    
    for file in tqdm(images_file_name, desc="Creating Dataset"):
            
        img_path = os.path.join(image_folder, file)
        
        hr = cv2.imread(img_path)
            
        hr = ImageConverter.convert_image(image = hr, mode = mode)
        
        lr = cv2.resize(hr, (hr.shape[1] // scale, hr.shape[0] // scale), interpolation = cv2.INTER_CUBIC)

        if resize_to_output:
            lr = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation = cv2.INTER_CUBIC)
        
        if preprocessing_required:
            if patch_size != None:
                height, width, _ = hr.shape
                
                if height < patch_size or width < patch_size:
                    hr = __resize_image(image = hr, target_height = patch_size, target_width = patch_size) 
                    lr = __resize_image(image = lr, target_height = patch_size, target_width = patch_size)
                    __add_image(file = file, hr = hr, lr = lr, hi_res_images = hi_res_images, low_res_images = low_res_images, count = count)
                    count += 1
                    
                else:
                    
                    pad_height = (stride - (height - patch_size) % stride) % stride
                    pad_width = (stride - (width - patch_size) % stride) % stride
                    
                    if (pad_width > 0 or pad_height > 0):
                        hr = __resize_image(image = hr, target_height = height + pad_height, target_width = width + pad_width)
                        lr = __resize_image(image = lr, target_height = height + pad_height, target_width = width + pad_width)
                    
                    for y in range(0, height - patch_size + 1, stride):
                        for x in range(0, width - patch_size + 1, stride):
                            hr_patch_image = hr[y : y + patch_size, x : x + patch_size]
                            lr_patch_image = lr[y : y + patch_size, x : x + patch_size]
                            __add_image(file = file, hr = hr_patch_image, lr = lr_patch_image, hi_res_images = hi_res_images, low_res_images = low_res_images, count = count)
                            count += 1
                        
            elif resize_rule != None:
                hr = __resize_image(image = hr, target_height = target_size[0], target_width = target_size[1])
                lr = __resize_image(image = lr, target_height = target_size[0], target_width = target_size[1])
                __add_image(file = file, hr = hr, lr = lr, hi_res_images = hi_res_images, low_res_images = low_res_images, count = count)
                count += 1
            
        else:
            __add_image(file = file, hr = hr, lr = lr, hi_res_images = hi_res_images, low_res_images = low_res_images, count = count)
            count += 1
        
def __resize_image(image: np.ndarray, target_height, target_width):
    
    pad_left = max((target_width - image.shape[1])  // 2, 0)
    pad_right = max((target_width - image.shape[1]) - pad_left, 0)

    pad_top = max((target_height -  image.shape[0]) // 2, 0)
    pad_bottom = max((target_height -  image.shape[0]) - pad_top, 0)
    
    return np.pad(array = image, pad_width=[(pad_top, pad_bottom), (pad_left, pad_right), (0, 0)], mode = 'reflect')


def __add_image(file: str, hr: np.ndarray, lr: np.ndarray, hi_res_images: h5py.Group, low_res_images: h5py.Group, count):
    
    # Here we permute because the Pytorch convolutionnal layer accept only Channels X Height X Width and Open CV return it as
    # Height X Width X Channels
    lr = np.transpose(lr, (2, 0, 1)).astype(np.float32) / 255.0

    hr = np.transpose(hr, (2, 0, 1)).astype(np.float32) / 255.0
    
    low_image = low_res_images.create_dataset(f"image_{count}", data = np.array(lr), dtype = np.float32)
    
    hi_image = hi_res_images.create_dataset(f"image_{count}", data = np.array(hr), dtype = np.float32)
    
    low_image.attrs["file"], hi_image.attrs["file"] = file, file

def __get_extreme_image_size(images_file_name: list, image_folder: str, resize_rule: ResizeRule = ResizeRule.BIGGEST) -> tuple[float, float]:
    
    extreme_height, extreme_width = 0.0, 0.0 if resize_rule == ResizeRule.BIGGEST else float('inf'), float('inf')
    
    for image_file in images_file_name:
        image_path = os.path.join(image_folder, image_file)
        
        height, width, _ = image_path.shape
        
        if(resize_rule == ResizeRule.BIGGEST):
            if(extreme_height < height):
                extreme_height = height
                
            elif(extreme_width < width):
                extreme_width = width
        
        else:
            if(extreme_height > height):
                extreme_height = height
                
            elif(extreme_width > width):
                extreme_width = width
    
    return (extreme_height, extreme_width)

def create_h5_image_file(input_path: str, scale: int, output_path: str, mode: ImageColorConverter, patch_size: int = None, stride: int = None, resize_rule: ResizeRule = None, preprocessing_required: bool = True):
    
    with h5py.File(output_path, 'w', libver='latest') as h5_file:
        
        hi_res_images = h5_file.create_group('hi_res')
        
        low_res_images = h5_file.create_group('low_res')
        
        __prepare_and_add_images(image_folder = input_path, scale = scale, mode = mode, hi_res_images = hi_res_images, 
                                 low_res_images = low_res_images, patch_size=patch_size, stride = stride, resize_rule = resize_rule,
                                 preprocessing_required = preprocessing_required)