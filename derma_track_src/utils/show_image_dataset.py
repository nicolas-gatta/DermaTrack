import os, sys

sys.path.insert(1, "\\".join(os.path.realpath(__file__).split("\\")[0:-2]))

from super_resolution.services.utils.dataloader import H5ImagesDataset
from super_resolution.services.utils.batch_sampler import SizeBasedImageBatch
import matplotlib.pyplot as plt



def show_first_10_images(dataset):
    """
    Show the 10 first images in the dataset

    Args:
        dataset (str): Path to the dataset
    """
    import numpy as np
    images_displayed = 0
    plt.figure(figsize=(12, 5))

    for i in range(10):
        _,image = dataset[i]
        
        image = image.detach().cpu().numpy()

        image = image.transpose(1, 2, 0)

        plt.subplot(2, 5, images_displayed + 1)
        plt.imshow(image.squeeze(), cmap='gray' if image.shape[-1] == 1 else None)
        plt.axis('off')
        images_displayed += 1

    plt.suptitle('First 10 Images from Train Dataset', fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_file = "C:\\Users\\Utilisateur\\Desktop\\Projet\\DermaTrack\\derma_track_src\\super_resolution\\datasets\\training\\BSD100_train_HR_BGR2YCrCb_x2_r5_sp1.hdf5"
    scale = 2

    train_dataset = H5ImagesDataset(train_file, up_scale_factor=scale)
    show_first_10_images(train_dataset)
