import h5py
import torch

from torch.utils.data import Dataset

class H5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.h5_file, self.lr_images, self.hr_images = None, None, None
        with h5py.File(self.h5_path, "r") as f:
            self.keys = list(f["low_res"].keys())  # Store keys safely
    
    def __len__(self):
        return len(self.keys)

    def _init_h5_file(self):
        """Each worker opens its own file handle when first accessed."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")  # Read-only mode
            self.lr_images = self.h5_file["low_res"]
            self.hr_images = self.h5_file["hi_res"]

    # Here we permute because the Pytorch convolutionnal layer accept only Channels X Height X Width and Open CV return it as
    # Height X Width X Channels
    def __getitem__(self, index):
        try:
            self._init_h5_file()
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
        
    def __del__(self):
        """ Ensures the file is closed when dataset object is deleted. """
        if self.h5_file is not None:
            self.h5_file.close()