import random
from tqdm import tqdm

from torch.utils.data import Sampler

class SizeBasedImageBatch(Sampler):
    def __init__(self, image_sizes: list, batch_size: int, shuffle: bool = True):
        
        self.batch_size = batch_size
        self.batches = self.__create_batches(image_sizes = image_sizes)
        
        if shuffle:
            random.shuffle(self.batches)
                
    def __create_batches(self, image_sizes): 
        batches = []
        current_batch = []
        current_size = None
        for h, w, index in tqdm(image_sizes, desc="Creating Batch"):
            
            image_size = (h, w)
            
            if(current_batch and (current_size != image_size or len(current_batch) >= self.batch_size)):
                
                batches.append(current_batch)
                current_batch = []
            
            current_batch.append(index)
                
            current_size = image_size

        if current_batch:
            batches.append(current_batch)
        
        return batches
        
    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
