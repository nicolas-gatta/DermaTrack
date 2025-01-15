import h5py
import torch

from torch.utils.data import Dataset

class H5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.file = h5py.File(h5_path, 'r')
        self.lr_images = self.file['lr']
        self.hr_images = self.file['hr']

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, index):
        lr = torch.tensor(self.lr_images[str(index)][:], dtype=torch.float32).permute(2, 0, 1) 
        hr = torch.tensor(self.hr_images[str(index)][:], dtype=torch.float32).permute(2, 0, 1)
        return lr, hr

    def close(self):
        self.file.close()