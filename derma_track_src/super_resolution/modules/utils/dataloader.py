import h5py
import torch

from torch.utils.data import Dataset

class H5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.h5_file, self.lr_images, self.hr_images = None, None, None

    def __enter__(self):
        """
        Opens the HDF5 file when entering the context manager.
        """
        self.h5_file = h5py.File(self.h5_path, "r")
        self.lr_images = self.h5_file["low_res"]
        self.hr_images = self.h5_file["hi_res"]
        return self
    
    def __len__(self):
        return len(self.lr_images)

    # Here we permute because the Pytorch convolutionnal layer accept only Channels X Height X Width and Open CV return it as
    # Height X Width X Channels
    def __getitem__(self, index):
        try:
            lr = torch.from_numpy(self.lr_images[f"image_{index + 1:03}"][:]).permute(2, 0, 1).float() / 255
            hr = torch.from_numpy(self.hr_images[f"image_{index + 1:03}"][:]).permute(2, 0, 1).float() / 255
            return lr, hr
        
        except KeyError:
            raise IndexError(f"Image index {index} out of range.")
        
    def get_raw_data(self, index):
        try:
            lr = self.lr_images[f"image_{index + 1:03}"][:]
            hr = self.hr_images[f"image_{index + 1:03}"][:]
            return lr, hr
        
        except KeyError:
            raise IndexError(f"Image index {index} out of range.")
        

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Closes the HDF5 file when exiting the context manager.
        """
        if self.h5_file:
            self.h5_file.close()