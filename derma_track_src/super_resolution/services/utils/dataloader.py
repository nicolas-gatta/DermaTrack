import random
import h5py
import torch
from torch.utils.data import Dataset

class H5ImagesDataset(Dataset):
    """
    A PyTorch Dataset for loading low and high resolution image pairs from an HDF5 file.
    """
    
    def __init__(self, h5_path: str, crop_size: int = 0, up_scale_factor: int = 0):    
        
        self.__h5_path = h5_path
        self.__h5_file = None
        self.crop_size = crop_size
        self.crop_size_lr = crop_size // up_scale_factor if up_scale_factor != 0 else 0
        self.up_scale_factor = up_scale_factor

    @property
    def image_sizes(self) -> list:
        """
        Retrieves the sizes of all images stored in the HDF5 file.
        
        Returns:
            list: A list containing the dimensions (height, width) of each image.
        """
        
        self.__init_h5_file()
        images_sizes = self.__h5_file["info"]["image_size"][:].tolist()
        self.close()
        self.__reset_variables()
        return images_sizes
          
    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        
        Returns:
            int: The number of samples available.
        """
        
        self.__init_h5_file()
        return len(self.__h5_file["low_res"])
    
    def __getitem__(self, index) -> tuple:
        """
        Retrieves a pair of low-resolution (LR) and high-resolution (HR) images from the HDF5 file at the specified index.

        Args:
            index (int): Index of the image pair to retrieve.
            
        Returns:
            tuple: A tuple (lr, hr)
            
        Raises:
            IndexError: If the provided index is out of range.
        """
        
        self.__init_h5_file()
        try:
            lr = torch.from_numpy(self.__h5_file["low_res"][f"image_{index + 1}"][:])
            hr = torch.from_numpy(self.__h5_file["hi_res"][f"image_{index + 1}"][:])

            _, height, width = hr.shape
            
            if self.crop_size != 0 and self.up_scale_factor != 0 and height > self.crop_size and width > self.crop_size:
                _, h, w = hr.shape
                top = random.randint(0, h - self.crop_size)
                left = random.randint(0, w - self.crop_size)

                top_lr = top // self.up_scale_factor
                left_lr = left // self.up_scale_factor
                
                hr_crop = hr[:, top:top + self.crop_size, left:left + self.crop_size]
                if lr.ndim == 3:
                    lr_crop = lr[:, top_lr:top_lr + self.crop_size_lr, left_lr:left_lr + self.crop_size_lr]
                elif lr.ndim == 4:
                    lr_crop = lr[:, :, top_lr:top_lr + self.crop_size_lr, left_lr:left_lr + self.crop_size_lr]
                return lr_crop, hr_crop
            
            return lr, hr
        
        except KeyError:
            raise IndexError(f"Image index {index} out of range.")
        
    def __del__(self) -> None:
        """ 
        Ensures the file is closed when dataset object is deleted. 
        """
        
        self.close()
 
    def close(self) -> None:
        """ 
        Ensures the file is closed. 
        """
        
        if self.__h5_file is not None:
            self.__h5_file.close()
            
    def __init_h5_file(self) -> None:
        """
            Initialize the h5 file, low resolution and high resolution image
        """
        if self.__h5_file is None:
            self.__h5_file = h5py.File(self.__h5_path, "r", libver = 'latest', swmr = True)
    
    def __reset_variables(self) -> None:
        self.__h5_file = None

    def get_raw_data(self, index: int) -> tuple:
        """
        Get the brute image data

        Args:
            index (int): The index of the image

        Raises:
            IndexError: If the index is out of range

        Returns:
            tuple: The low and high resolution image into a ndarray
        """
        
        try:
            lr = self.__h5_file["low_res"][f"image_{index + 1}"][:]
            hr = self.__h5_file["hi_res"][f"image_{index + 1}"][:]
            return lr, hr
        
        except KeyError:
            raise IndexError(f"Image index {index} out of range.")





    
