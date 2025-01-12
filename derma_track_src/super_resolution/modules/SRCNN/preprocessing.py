import cv2
import os
import numpy as np
import h5py

def prepare_images(image_folder, scale):
    lr_images = []
    hr_images = []
    
    for img_name in os.listdir(image_folder):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            
            # Get the image Path
            img_path = os.path.join(image_folder, img_name)
            
            # High resolution image
            hr = cv2.imread(img_path)  
            hr = cv2.cvtColor(hr, cv2.COLOR_BGR2YCrCb)
            
            # Downsize the high resolution image
            lr = cv2.resize(hr, (hr.shape[1] // scale, hr.shape[0] // scale), interpolation=cv2.INTER_CUBIC)
            
            # Upscale the low resolution image to get the the right size
            lr = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)
            
            lr_images.append(np.array(lr))
            hr_images.append(np.array(hr))
    
    return np.array(lr_images), np.array(hr_images)

def create_h5_image_file(image_folder, scale, output_path):
    
    with h5py.File(output_path, 'w') as h5_file:
        
        lr_patches, hr_patches = prepare_images(image_folder, scale)
        
        h5_file.create_dataset('hi_res_dataset', data=hr_patches)
        
        h5_file.create_dataset('low_res_dataset', data=lr_patches)
        
    print(f"File created sucessfully at the path: {output_path}")