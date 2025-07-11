import cv2
import os
import numpy as np
import h5py
import math
from tqdm import tqdm
from super_resolution.services.utils.image_converter import ImageConverter, ImageColorConverter
from enum import Enum

class ResizeRule(str, Enum):
    """
    An enumeration representing rules for resizing operations.
    """
    
    BIGGEST = "max"
    SMALLEST = "min"

def __prepare_image(file: str, hr: np.ndarray, scale: int, hi_res_images: h5py.Group, low_res_images: h5py.Group,
                    patch_size: int, stride: int, resize_rule: ResizeRule, preprocessing_required: bool = True, 
                    resize_to_output: bool = True, multi_input: bool = False, target_size: int = None, max_translation: int = None,
                    image_info_list: list = None, count: int = None):
    
    """
    Prepares high-resolution (HR) and low-resolution (LR) image pairs for a super-resolution dataset.
    
    Args:
        file (str): Path or identifier of the image file being processed.
        hr (np.ndarray): High-resolution image as a NumPy array.
        scale (int): Downscaling factor for generating the low-resolution image.
        hi_res_images (h5py.Group): HDF5 group for storing high-resolution images.
        low_res_images (h5py.Group): HDF5 group for storing low-resolution images.
        patch_size (int): Size of the patches to extract from the images. If None, no patch extraction is performed.
        stride (int): Stride for patch extraction.
        resize_rule (ResizeRule): Rule or method for resizing images. If None, resizing is skipped.
        preprocessing_required (bool, optional): Whether to apply preprocessing steps. Defaults to True.
        resize_to_output (bool, optional): Whether to resize LR images to match HR output size. Defaults to True.
        multi_input (bool, optional): Whether to prepare images for multi-input models. Defaults to False.
        target_size (int, optional): Target size for resizing images (height, width). Used if patch_size is None.
        max_translation (int, optional): Maximum translation for multi-input preparation.
        image_info_list (list, optional): List to store image metadata and information.
        count (int, optional): Counter for the number of processed images.
        
    Returns:
        tuple: image_info_list, count
    """
    
    blur_image = cv2.GaussianBlur(hr, (5, 5), 1.5)
    
    lr = cv2.resize(blur_image, (hr.shape[1] // scale, hr.shape[0] // scale))

    if resize_to_output:
        lr = cv2.resize(lr, (hr.shape[1], hr.shape[0]))
            
    if preprocessing_required and not multi_input:
        if patch_size != None:
            height, width, _ = hr.shape
            
            if height < patch_size or width < patch_size:
                
                lr_target_height = patch_size if resize_to_output else (patch_size // scale)
                
                lr_target_width =  patch_size if resize_to_output else (patch_size // scale)
                
                hr = __crop_or_pad_image(image = hr, target_height = patch_size, target_width = patch_size) 
                
                lr = __crop_or_pad_image(image = lr, target_height = lr_target_height, target_width = lr_target_width)
                    
                image_info_list = __add_image(file = file, hr = hr, lr = lr, hi_res_images = hi_res_images, low_res_images = low_res_images, count = count, image_info_list = image_info_list)
                count += 1
                
            else:
                for y in range(0, height - patch_size + 1, stride):
                    for x in range(0, width - patch_size + 1, stride):
                        
                        lr_y = y if resize_to_output else y // scale
                        
                        lr_x = x if resize_to_output else x // scale
                        
                        lr_patch_size = patch_size if resize_to_output else patch_size // scale
                        
                        hr_patch_image = hr[y : y + patch_size, x : x + patch_size]
                        lr_patch_image = lr[lr_y : lr_y + lr_patch_size, lr_x : lr_x + lr_patch_size]
                        image_info_list = __add_image(file = file, hr = hr_patch_image, lr = lr_patch_image, hi_res_images = hi_res_images, low_res_images = low_res_images, count = count, image_info_list = image_info_list)
                        count += 1
                    
        elif resize_rule != None:
            
            lr_target_height = target_size[0] if resize_to_output else (target_size[0] // scale)  
                
            lr_target_width = target_size[1] if resize_to_output else (target_size[1] // scale) 
                
            hr = __crop_or_pad_image(image = hr, target_height = target_size[0], target_width = target_size[1])
            
            lr = __crop_or_pad_image(image = lr, target_height = lr_target_height, target_width = lr_target_width)
                
            image_info_list = __add_image(file = file, hr = hr, lr = lr, hi_res_images = hi_res_images, low_res_images = low_res_images, count = count)
            count += 1

    else:            
        
        if multi_input:
            hr = __crop_or_pad_image(image = hr, target_height = hr.shape[0] - (max_translation * scale), target_width = hr.shape[1] - (max_translation * scale))
        
        image_info_list = __add_image(file = file, hr = hr, lr = lr, hi_res_images = hi_res_images, low_res_images = low_res_images, count = count, image_info_list = image_info_list, multi_input = multi_input, max_translation = max_translation)
        
        count += 1
    
    return image_info_list, count

def __crop_rectangle(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    
    source: https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """

    rad_angle = math.radians(angle)
    
    if w <= 0 or h <= 0:
        return 0,0

    width_is_longer = w >= h
    side_long, side_short = (w,h) if width_is_longer else (h,w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(rad_angle)), abs(math.cos(rad_angle))
    if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

    return int(wr), int(hr)

def __rotate_and_crop_image(image: np.ndarray, angle: float, scale: int):
    """
    Rotates an image by a specified angle, then crops or pads it to a new size that is a multiple of the given scale.
    
    Args:
        image (numpy.ndarray): The input image to be rotated and cropped.
        angle (float): The angle (in degrees) to rotate the image.
        scale (int): The scale factor used to determine the final cropped size.
        
    Returns:
        numpy.ndarray: The rotated and cropped (or padded) image with dimensions that are multiples of the scale.
    """
    
    (h, w) = image.shape[:2]
    
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    rotated = cv2.warpAffine(image, matrix, (w, h))

    new_width, new_height = __crop_rectangle(w = w, h = h, angle = angle)
    
    new_width = ((new_width // scale) // scale) * scale * scale
    new_height = ((new_height // scale) // scale) * scale * scale
    
    return __crop_or_pad_image(image = rotated, target_width=new_width, target_height=new_height)    

def __prepare_and_add_images(image_folder: str, scale: int, mode: ImageColorConverter, hi_res_images: h5py.Group, low_res_images: h5py.Group, image_info: h5py.Group,
                             patch_size: int, stride: int, resize_rule: ResizeRule, preprocessing_required: bool = True, resize_to_output: bool = True, multi_input: bool = False,
                             max_angle_rotation: int = None, angle_rotation_step: int = None) -> None:
    """
    Prepares and adds images from a specified folder to HDF5 datasets for high-resolution and low-resolution images.
    
    Args:
        image_folder (str): Path to the folder containing input images.
        scale (int): Downscaling factor for generating low-resolution images.
        mode (ImageColorConverter): Color conversion mode to apply to images.
        hi_res_images (h5py.Group): HDF5 group to store high-resolution image patches.
        low_res_images (h5py.Group): HDF5 group to store low-resolution image patches.
        image_info (h5py.Group): HDF5 group to store image metadata (e.g., patch sizes).
        patch_size (int): Size of the patches to extract from images.
        stride (int): Stride for patch extraction.
        resize_rule (ResizeRule): Rule for resizing images before patch extraction.
        preprocessing_required (bool, optional): Whether to apply preprocessing to images. Defaults to True.
        resize_to_output (bool, optional): Whether to resize images to the output size. Defaults to True.
        multi_input (bool, optional): Whether to generate multiple input variants per image. Defaults to False.
        max_angle_rotation (int, optional): Maximum angle (in degrees) for image rotation augmentation. If None, no rotation is applied.
        angle_rotation_step (int, optional): Step size (in degrees) for rotation augmentation. Used if max_angle_rotation is set.
        
    Returns:
        None. The function modifies the provided HDF5 groups in-place by adding processed image patches and metadata.
    """
    
    images_file_name = [file for file in os.listdir(image_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    target_size = None
    
    count = 1
    
    max_translation = 20
    
    image_info_list = []
    
    if resize_rule != None:
        target_size = __get_extreme_image_size(images_file_name = images_file_name, image_folder = image_folder, resize_rule = resize_rule)
    
    for file in tqdm(images_file_name, desc="Creating Dataset", leave=True, dynamic_ncols=True):
            
        img_path = os.path.join(image_folder, file)
        
        hr = cv2.imread(img_path)
            
        hr = ImageConverter.convert_image(image = hr, mode = mode)
    
        if  max_angle_rotation != None and angle_rotation_step!= None:
            min_angle_rotation = -max_angle_rotation
            
            angles = [i for i in range(min_angle_rotation, max_angle_rotation, angle_rotation_step)]
            
            angles[len(angles)//2] = 0
            
            for angle in angles:
                
                rotated_image = __rotate_and_crop_image(image = hr, angle = angle, scale = scale)
                
                image_info_list, count = __prepare_image(file = file, hr = rotated_image, scale = scale, hi_res_images = hi_res_images, low_res_images = low_res_images, patch_size = patch_size,
                                                        stride = stride, resize_rule = resize_rule, preprocessing_required = preprocessing_required, resize_to_output = resize_to_output,
                                                        multi_input = multi_input, target_size = target_size, max_translation = max_translation, image_info_list = image_info_list, count = count)
            
        else:
            hr = hr[:(((hr.shape[0] // scale) // scale) * scale ) * scale, :(((hr.shape[1] // scale) // scale) * scale) * scale] 
            image_info_list, count = __prepare_image(file = file, hr = hr, scale = scale, hi_res_images = hi_res_images, low_res_images = low_res_images, patch_size = patch_size,
                                                    stride = stride, resize_rule = resize_rule, preprocessing_required = preprocessing_required, resize_to_output = resize_to_output,
                                                    multi_input = multi_input, target_size = target_size, max_translation = max_translation, image_info_list = image_info_list, count = count)
    
    image_info.create_dataset("image_size", data = np.array(image_info_list), dtype = np.dtype('int64'))
        
def __crop_or_pad_image(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """
    Crops or pads an image to the specified target height and width.
    
    Args:
        image (np.ndarray): Input image as a NumPy array of shape (H, W, C).
        target_height (int): Desired height for the image.
        target_width (int): Desired width for the image.
        
    Returns:
        np.ndarray: The cropped or padded image with shape (target_height, target_width, C).
    """
    
    height, width, _ = image.shape
    
    diff_width = target_width - image.shape[1]
    diff_height = target_height -  image.shape[0]
    
    pad_left = max(diff_width // 2, 0)
    pad_right = max(diff_width - pad_left, 0)
    pad_top = max(diff_height // 2, 0)
    pad_bottom = max(diff_height - pad_top, 0)
    
    crop_left = max(-diff_width // 2, 0)
    crop_right = max(-diff_width - crop_left, 0)
    crop_top = max(-diff_height // 2, 0)
    crop_bottom = max(-diff_height - crop_top,  0)
    
    image = image[crop_top:height - crop_bottom, crop_left:width - crop_right]
    return np.pad(array = image, pad_width=[(pad_top, pad_bottom), (pad_left, pad_right), (0, 0)], mode = 'reflect')


def __add_image(file: str, hr: np.ndarray, lr: np.ndarray, hi_res_images: h5py.Group, low_res_images: h5py.Group, count: int, image_info_list: list, multi_input: bool = False, max_translation: int = 20) -> list:
    """
    Adds a high-resolution (HR) and low-resolution (LR) image pair to HDF5 groups, processes them for PyTorch compatibility, and updates image metadata.
    
    Args:
        file (str): The file path or identifier for the image.
        hr (np.ndarray): High-resolution image as a NumPy array (Height x Width x Channels).
        lr (np.ndarray): Low-resolution image as a NumPy array (Height x Width x Channels).
        hi_res_images (h5py.Group): HDF5 group to store high-resolution images.
        low_res_images (h5py.Group): HDF5 group to store low-resolution images.
        count (int): Index or count for naming the dataset in the HDF5 group.
        image_info_list (list): List to append image metadata (height, width, index).
        multi_input (bool, optional): If True, creates a burst of LR images for multi-frame input. Defaults to False.
        max_translation (int, optional): Maximum translation for burst image creation. Defaults to 20.
        
    Returns:
        list: Updated image_info_list with metadata for the added image.
    """
    
    image_info_list.append((hr.shape[0], hr.shape[1], count - 1))
    
    # Here we permute because the Pytorch convolutionnal layer accept only Channels X Height X Width and Open CV return it as
    # Height X Width X Channels
    lr = np.transpose(lr, (2, 0, 1)).astype(np.float32) / 255.0

    hr = np.transpose(hr, (2, 0, 1)).astype(np.float32) / 255.0
    
    if multi_input:
        lr = __create_burst_image(lr = lr, burst_size = 5, max_translation = max_translation)
        
    low_image = low_res_images.create_dataset(f"image_{count}", data = np.array(lr), dtype = np.float32)
    
    hi_image = hi_res_images.create_dataset(f"image_{count}", data = np.array(hr), dtype = np.float32)
    
    low_image.attrs["file"], hi_image.attrs["file"] = file, file
    
    return image_info_list

def __create_burst_image(lr: np.ndarray, burst_size: int = 5, max_translation: int = 20) -> np.ndarray:
    """
    Generates a burst of low-resolution images by applying spatial translations to the input image.
    
    Args:
        lr (np.ndarray): Input low-resolution image as a NumPy array of shape (channels, height, width).
        burst_size (int, optional): Number of images to generate in the burst. Defaults to 5.
        max_translation (int, optional): Maximum number of pixels to translate the image in both height and width. Defaults to 20.
        
    Returns:
        np.ndarray: A NumPy array of shape (burst_size, channels, cropped_height, cropped_width) containing the burst of images.
    """
    
    burst_lr = []
    height, width = lr.shape[1:]

    step = max_translation // burst_size
    
    cropped_height = height - max_translation
    cropped_width = width - max_translation
    
    translations = [step * i for i in range(burst_size)]
    
    translations[(len(translations) - 1) // 2] = max_translation // 2
    
    for translation in translations:
         
        burst_lr.append(lr[:, translation: cropped_height + translation, translation: cropped_width + translation])
    
    return np.stack(burst_lr)
    
def __get_extreme_image_size(images_file_name: list, image_folder: str, resize_rule: ResizeRule = ResizeRule.BIGGEST) -> tuple:
    """
    Calculates the extreme (biggest or smallest) height and width among a list of images.
    
    Args:
        images_file_name (list): List of image file names to process.
        image_folder (str): Path to the folder containing the images.
        resize_rule (ResizeRule, optional): Rule to determine whether to find the biggest or smallest image size.
            Defaults to ResizeRule.BIGGEST.
            
    Returns:
        tuple: A tuple containing the extreme height and width found among the images.
    """
    
    
    extreme_height, extreme_width = 0.0, 0.0 if resize_rule == ResizeRule.BIGGEST else float('inf'), float('inf')
    
    for image_file in images_file_name:
        image_path = os.path.join(image_folder, image_file)
        
        height, width, _ = image_path.shape
        
        if(resize_rule == ResizeRule.BIGGEST):
            if(extreme_height < height):
                extreme_height = height
                
            if(extreme_width < width):
                extreme_width = width
        
        else:
            if(extreme_height > height):
                extreme_height = height
                
            if(extreme_width > width):
                extreme_width = width
    
    return (extreme_height, extreme_width)

def create_h5_image_file(input_path: str, scale: int, output_path: str, mode: ImageColorConverter, patch_size: int = None, stride: int = None, 
                         resize_rule: ResizeRule = None, preprocessing_required: bool = True, resize_to_output: bool = True, multi_input: bool = False,
                         max_angle_rotation: int = None, angle_rotation_step: int = None) -> None:
    """
    Creates an HDF5 (.h5) file containing high-resolution and low-resolution image datasets.
    
    Args:
        input_path (str): Path to the directory containing input images.
        scale (int): The scaling factor for generating low-resolution images from high-resolution images.
        output_path (str): Path where the output HDF5 file will be saved.
        mode (ImageColorConverter): The color conversion mode to apply to images (e.g., RGB, grayscale).
        patch_size (int, optional): Size of image patches to extract. If None, uses full images.
        stride (int, optional): Stride for patch extraction. If None, uses default stride.
        resize_rule (ResizeRule, optional): Rule for resizing images before processing.
        preprocessing_required (bool, optional): Whether to apply preprocessing steps to images. Defaults to True.
        resize_to_output (bool, optional): Whether to resize images to match output dimensions. Defaults to True.
        multi_input (bool, optional): Whether to support multiple input images per sample. Defaults to False.
        max_angle_rotation (int, optional): Maximum angle for image rotation augmentation. If None, no rotation is applied.
        angle_rotation_step (int, optional): Step size for angle rotation augmentation. If None, no rotation is applied.
    """
    
    with h5py.File(output_path, 'w', libver='latest') as h5_file:
        
        hi_res_images = h5_file.create_group('hi_res')
        
        low_res_images = h5_file.create_group('low_res')
        
        image_info = h5_file.create_group('info')
        
        __prepare_and_add_images(image_folder = input_path, scale = scale, mode = mode, hi_res_images = hi_res_images, 
                                 low_res_images = low_res_images, image_info = image_info, patch_size=patch_size,
                                 stride = stride, resize_rule = resize_rule, preprocessing_required = preprocessing_required,
                                 resize_to_output = resize_to_output, multi_input = multi_input, max_angle_rotation = max_angle_rotation, 
                                 angle_rotation_step = angle_rotation_step)

def dataset_exist_or_create(dataset: str, mode: str, scale: int, category: str, patch_size: int, stride: int, resize_rule: str, 
                            resize_to_output: bool, base_dir: str, multi_input: bool, max_angle_rotation: int, angle_rotation_step: int) -> str:
    """
    Checks if a preprocessed dataset file exists for the given parameters or creates it if it does not exist.
    
    Args:
        dataset (str): Name of the dataset.
        mode (str): Image color mode (e.g., grayscale, RGB).
        scale (int): Upscaling factor for super-resolution.
        category (str): Dataset category (e.g., 'train', 'validation', 'evaluation').
        patch_size (int): Size of image patches to extract (if applicable).
        stride (int): Stride for patch extraction (if applicable).
        resize_rule (str): Rule for resizing images (if applicable).
        resize_to_output (bool): Whether to resize images to the output size.
        base_dir (str): Base directory containing datasets.
        multi_input (bool): Whether to use multiple input images.
        max_angle_rotation (int): Maximum angle for image rotation augmentation.
        angle_rotation_step (int): Step size for rotation augmentation.
        
    Returns:
        str: Path to the (existing or newly created) HDF5 dataset file.
    """
    
    file_name = f"{dataset}_{mode}_x{scale}"
    
    c_resize_rule = None
            
    preprocessing_required = (category != "evaluation") and not multi_input
    
    if preprocessing_required:
        if patch_size != None and stride != None:
            file_name += f"_p{patch_size}_s{stride}"
            
        elif resize_rule != None:
            file_name += f"_{resize_rule}"
            c_resize_rule = ResizeRule[resize_rule]
    
    if not resize_to_output:
        file_name += "_nrto"
    
    if max_angle_rotation != None and angle_rotation_step != None:
        file_name += f"_r{max_angle_rotation}_sp{angle_rotation_step}"
        
    if multi_input:
        file_name += "_mi"
    
    output_path = os.path.join(base_dir, "super_resolution", "datasets", category, f"{file_name}.hdf5")
    
    if not os.path.exists(output_path):
        create_h5_image_file(input_path = os.path.join(base_dir, "super_resolution", "base_datasets", category, dataset),
                            scale = scale, output_path = output_path, mode = ImageColorConverter[mode], patch_size = patch_size,
                            stride = stride, resize_rule = c_resize_rule, preprocessing_required = preprocessing_required, 
                            resize_to_output = resize_to_output, multi_input = multi_input, max_angle_rotation = max_angle_rotation, 
                            angle_rotation_step = angle_rotation_step)
    return output_path