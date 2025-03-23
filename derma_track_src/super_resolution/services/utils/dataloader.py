import h5py
import torch

from torch.utils.data import Dataset

class H5ImagesDataset(Dataset):
    
    def __init__(self, h5_path):    
        
        self.__h5_path = h5_path
        
        self.__h5_file, self.__lr_images, self.__hr_images = None, None, None
    
    def __compute_image_sizes(self):
        sizes = []
        for i in range(len(self.__hr_images)):
            shape = self.__hr_images[f'image_{i+1}'].shape
            sizes.append((shape[0], shape[1], i))
        sizes.sort(key=lambda x: (x[0], x[1]))
        return sizes

    @property
    def image_sizes(self):
        self.__init_h5_file()
        images_sizes = self.__compute_image_sizes()
        self.close()
        self.__reset_variables()
        return images_sizes
    
    @property
    def hr_images(self):
        return self.__hr_images
    
    @property
    def lr_images(self):
        return self.__lr_images
          
    def __len__(self):
        self.__init_h5_file()
        return len(self.__hr_images)
    
    # Here we permute because the Pytorch convolutionnal layer accept only Channels X Height X Width and Open CV return it as
    # Height X Width X Channels
    def __getitem__(self, index):
        self.__init_h5_file()
        try:
            lr = torch.from_numpy(self.__lr_images[f"image_{index + 1}"][:]).permute(2, 0, 1).float() / 255
            hr = torch.from_numpy(self.__hr_images[f"image_{index + 1}"][:]).permute(2, 0, 1).float() / 255
            return lr, hr
        
        except KeyError:
            raise IndexError(f"Image index {index} out of range.")
        
    def __del__(self):
        """ Ensures the file is closed when dataset object is deleted. """
        self.close()

        
    def close(self):
        """ Ensures the file is closed. """
        if self.__h5_file is not None:
            self.__h5_file.close()
            
    def __init_h5_file(self):
        """
            Initialize the h5 file, low resolution and high resolution image
        """
        if self.__h5_file is None:
            self.__h5_file = h5py.File(self.__h5_path, "r")
            self.__hr_images = self.__h5_file["hi_res"]
            self.__lr_images = self.__h5_file["low_res"]
    
    def __reset_variables(self):
        self.__h5_file, self.__lr_images, self.__hr_images = None, None, None

    def get_raw_data(self, index):
        try:
            lr = self.__lr_images[f"image_{index + 1}"][:]
            hr = self.__hr_images[f"image_{index + 1}"][:]
            return lr, hr
        
        except KeyError:
            raise IndexError(f"Image index {index} out of range.")

    
