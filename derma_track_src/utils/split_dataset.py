import os
import random
import shutil

def split_dataset(dataset_path: str, output_path: str, train_split: float = 0.70, val_split: float = 0.15, test_split: float = 0.15, seed: int = 25) -> None:
    """
    Splits images from one folder into train/val/test folders.

    Args:
        dataset_path (str): Path to the folder containing all images.
        output_path (str): Path where 'train', 'val', and 'test' folders will be created.
        train_split (float): Ratio of images to put into the train folder.
        val_split (float): Ratio of images to put into the validation folder.
        test_split (float): Ratio of images to put into the test folder.
        seed (int): Random seed for reproducibility.
    """
    
    images = [file for file in os.listdir(dataset_path) if (os.path.isfile(os.path.join(dataset_path, file)) and file.lower().endswith(('.png', '.jpg', '.jpeg')))]
    total_images = len(images)
    
    random.seed(seed)
    random.shuffle(images)
    
    train_set_until = int(total_images * train_split)
    vald_set_until = (train_set_until + int(total_images * val_split))
    
    train_images = images[:train_set_until]
    val_images = images[train_set_until : vald_set_until]
    test_images = images[vald_set_until:]
    
    split_data = [train_images, val_images, test_images]
    
    train_dir = os.path.join(output_path,"train")
    val_dir = os.path.join(output_path,"val")
    test_dir = os.path.join(output_path,"eval")
    
    new_directories = [train_dir, val_dir, test_dir]
    
    for dir in new_directories:
        os.makedirs(dir, exist_ok=True)
    
    for directory, list_images in zip(new_directories, split_data):
        for index, image in enumerate(list_images):
            image_extension = image.split(".")[1].lower()
            shutil.copy(src = os.path.join(dataset_path, image), dst = os.path.join(directory, f"image_{index + 1}.{image_extension}"))
    
    print("Dataset successfully split into train, validation and evaluation set")

if __name__ == "__main__":
    
    split_dataset(dataset_path = "C:\\Users\\Utilisateur\\Desktop\\derma_dataset", output_path = "C:\\Users\\Utilisateur\\Desktop")
    

   