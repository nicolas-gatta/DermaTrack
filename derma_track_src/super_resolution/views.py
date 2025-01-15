import numpy as np

from django.shortcuts import render
from django.http import HttpResponse
from PIL import Image
from super_resolution.modules.SRCNN import train
from derma_track_src.super_resolution.modules.utils.preprocessing import create_h5_image_file
from super_resolution.modules.utils.path_finder import PathFinder

# Create your views here.


def training_srcnn(request):
    
    base_string = "super_resolution/modules/srcnn/"
    
    scale = request.POST["scale"]
    
    train_file = PathFinder.get_complet_path(f"{base_string}input/{request.POST["training_dataset"]}_x{scale}.hdf5")
    
    eval_file = PathFinder.get_complet_path(f"{base_string}input/{request.POST["eval_dataset"]}_x{scale}.hdf5")
    
    learning_rate = request.POST["learning_rate"]
    
    batch_size = request.POST["batch_size"]
    
    num_epochs = request.POST["num_epochs"]
    
    # Create Training file
    create_h5_image_file(image_folder = PathFinder.get_complet_path(f"{base_string}dataset/{request.POST["training_dataset"]}"),
                         scale = scale,
                         output_path = train_file,
                         mode = "BGR_to_YCrCb")
    
    # Create Evaluation file
    create_h5_image_file(image_folder = PathFinder.get_complet_path(f"{base_string}dataset/{request.POST["eval_dataset"]}"),
                         scale = scale,
                         output_path = eval_file,
                         mode = "BGR_to_YCrCb")
    
    train.train_model(train_file = train_file, 
                eval_file = eval_file, 
                output_dir = PathFinder.get_complet_path(f"{base_string}output/{request.POST["output_file"]}_{learning_rate}_{batch_size}_{num_epochs}_x{scale}"),
                learning_rate = learning_rate, 
                seed = request.POST["seed"], 
                batch_size = batch_size,
                num_epochs = num_epochs,
                num_workers = request.POST["num_workers"])
    
    return HttpResponse("Hello, world. You're at the polls index.")

def training_srgan(request):
    
    return HttpResponse("Hello, world. You're at the polls index.")

def training_esrgan(request):
    
    return HttpResponse("Hello, world. You're at the polls index.")

def training_misr(request):
    
    return HttpResponse("Hello, world. You're at the polls index.")

def compare_super_resolution_algorithm():
    
    return HttpResponse("Hello, world. You're at the polls index.")

