import random
import h5py
import torch


from torch.utils.data import Dataset

class H5ImagesDataset(Dataset):
    
    def __init__(self, h5_path, crop_size = 0, up_scale_factor = 0):    
        
        self.__h5_path = h5_path
        self.__h5_file = None
        self.crop_size = crop_size
        self.crop_size_lr = crop_size // up_scale_factor if up_scale_factor != 0 else 0
        self.up_scale_factor = up_scale_factor

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

            _, height, width = hr.shape
            
            if self.crop_size != 0 and self.up_scale_factor != 0 and height > self.crop_size and width > self.crop_size:
                _, h, w = hr.shape
                top = random.randint(0, h - self.crop_size)
                left = random.randint(0, w - self.crop_size)

                top_lr = top // self.up_scale_factor
                left_lr = left // self.up_scale_factor
                
                hr_crop = hr[:, top:top + self.crop_size, left:left + self.crop_size]
                lr_crop = lr[:, top_lr:top_lr + self.crop_size_lr, left_lr:left_lr + self.crop_size_lr]
                return lr_crop, hr_crop
            
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





    
