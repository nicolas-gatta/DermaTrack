import random
from tqdm import tqdm

from torch.utils.data import Sampler

class SizeBasedImageBatch(Sampler):
    """
    A PyTorch Sampler that groups image indices into batches based on their sizes.
    """
    
    def __init__(self, image_sizes: list, batch_size: int, shuffle: bool = True):
        """
        Initialize the SizeBasedImageBatch Class

        Args:
            image_sizes (list): 
            batch_size (int): The maximum size of each batches.
            shuffle (bool): Shuffle the batches or not. Default = True.
        """
        self.batch_size = batch_size
        self.batches = self.__create_batches(image_sizes = image_sizes)
        self.shuffle = shuffle
                
    def __create_batches(self, image_sizes: list) -> list: 
        """
        Create all the batches based on the image_sizes

        Args:
            image_sizes (list): list of all the images size

        Returns:
            list: a list of all the batches
        """
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
        """
        Returns an iterator over the batches with random if needed.
        """
         
        if self.shuffle:
            random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self) -> int:
        """
        Returns the size of the batches
        """
        return len(self.batches)
