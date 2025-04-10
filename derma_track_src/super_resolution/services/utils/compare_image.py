import matplotlib.pyplot as plt
import numpy as np

def compare_image(lr_crop, hr_crop):
    # Convert from torch tensor to numpy and move channels to last dimension
    lr_np = lr_crop.permute(1, 2, 0).numpy()
    hr_np = hr_crop.permute(1, 2, 0).numpy()

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(lr_np)
    axs[0].set_title("Low-Res Crop")
    axs[0].axis('off')
    
    axs[1].imshow(hr_np)
    axs[1].set_title("High-Res Crop")
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.show()