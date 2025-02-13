import h5py
import random

from torch.utils.data import Sampler

from super_resolution.services.utils.dataloader import H5ImagesDataset

class SizeBasedImageBatch(Sampler):
    def __init__(self, dataset: H5ImagesDataset, batch_size: int, shuffle: bool = True):
        self.batches, batch, image_sizes = [], [], []
        height, width = 0, 0
        
        with h5py.File(dataset.h5_path, "r") as f:
            for i in range(len(dataset)):
                shape = f["hi_res"][f"image_{i + 1}"].shape
                image_sizes.append((shape[0], shape[1], i))

        image_sizes.sort(key=lambda x: (x[0], x[1]))

        for h, w, index in image_sizes:
            
            if(batch and height != 0 and width != 0 and (height != h or width != w)):
                
                self.batches.append(batch)
                batch = []
            
            batch.append(index)
            
            if (len(batch) == batch_size):
                self.batches.append(batch)
                batch = []
            
            height = h 
            width = w

        if batch:
            self.batches.append(batch)
        
        if shuffle:
            random.shuffle(self.batches)
            
    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
