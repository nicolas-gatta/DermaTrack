import h5py
import torch

from torch.utils.data import Dataset

class H5ImagesDataset(Dataset):
    
    def __init__(self, h5_path, crop_size = 96):    
        
        self.__h5_path = h5_path
        
        self.__h5_file = None

    @property
    def image_sizes(self):
        self.__init_h5_file()
        images_sizes = self.__h5_file["info"]["image_size"][:].tolist()
        self.close()
        self.__reset_variables()
        return images_sizes
          
    def __len__(self):
        self.__init_h5_file()
        return len(self.__h5_file["low_res"])
    

    def __getitem__(self, index):
        self.__init_h5_file()
        try:
            lr = torch.from_numpy(self.__h5_file["low_res"][f"image_{index + 1}"][:])
            hr = torch.from_numpy(self.__h5_file["hi_res"][f"image_{index + 1}"][:])
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
            self.__h5_file = h5py.File(self.__h5_path, "r", libver = 'latest', swmr = True)
    
    def __reset_variables(self):
        self.__h5_file = None

    def get_raw_data(self, index):
        try:
            lr = self.__h5_file["low_res"][f"image_{index + 1}"][:]
            hr = self.__h5_file["hi_res"][f"image_{index + 1}"][:]
            return lr, hr
        
        except KeyError:
            raise IndexError(f"Image index {index} out of range.")

    
