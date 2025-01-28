import numpy as np
import cv2

from PIL import Image

from django.http import HttpResponse
from django.template.loader import render_to_string
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required

from .modules.SRCNN import train
from .modules.utils.preprocessing import create_h5_image_file
from .forms.training_form import TrainingForm
from .modules.utils.json_manager import JsonManager, ModelField
from .modules.utils.image_converter import ImageColorConverter, ImageConverter
from .modules.utils.dataloader import H5Dataset

from utils.checks import group_and_super_user_checks
from utils.path_finder import PathFinder
# Create your views here.


def training_srcnn(request):
    
    base_string = "super_resolution/modules/srcnn/"
    
    scale = request.POST["scale"]
    
    train_file = PathFinder.get_complet_path(f"{base_string}input/{request.POST['training_dataset']}_x{scale}.hdf5")
    
    eval_file = PathFinder.get_complet_path(f"{base_string}input/{request.POST['eval_dataset']}_x{scale}.hdf5")
    
    learning_rate = request.POST["learning_rate"]
    
    batch_size = request.POST["batch_size"]
    
    num_epochs = request.POST["num_epochs"]
    
    # Create Training file
    create_h5_image_file(input_path = PathFinder.get_complet_path(f"{base_string}dataset/{request.POST['training_dataset']}"),
                         scale = scale,
                         output_path = train_file,
                         mode = "BGR_to_YCrCb")
    
    # Create Evaluation file
    create_h5_image_file(input_path = PathFinder.get_complet_path(f"{base_string}dataset/{request.POST['eval_dataset']}"),
                         scale = scale,
                         output_path = eval_file,
                         mode = "BGR_to_YCrCb")
    
    train.train_model(train_file = train_file, 
                eval_file = eval_file, 
                output_dir = PathFinder.get_complet_path(f"{base_string}output/{request.POST['output_file']}_{learning_rate}_{batch_size}_{num_epochs}_x{scale}"),
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

@login_required
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def show_models(request):
    if request.headers.get('HX-Request'):
        models = JsonManager.load_training_results()
        return render(request, 'partial/show_models.html', {"models": models})

@login_required
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def model_form(request):
    if request.headers.get('HX-Request'):
        #form = TrainingForm()
        return render(request, 'partial/model_form.html', {"form": None})

@login_required
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def training_model(request):
    pass

def test_2(request):
    JsonManager.training_results_to_json(architecture="SRCNN", model_name="Hello", train_file="blablabla", eval_file="blueblueblue", learning_rate=1e-10, seed=1, batch_size=16, num_epochs=100, num_workers=8)
    JsonManager.training_results_to_json(architecture="SRGAN", model_name="Hello", train_file="blablabla", eval_file="blueblueblue", learning_rate=1e-10, seed=1, batch_size=16, num_epochs=100, num_workers=8)
    JsonManager.training_results_to_json(architecture="ESRGAN", model_name="Hello", train_file="blablabla", eval_file="blueblueblue", learning_rate=1e-10, seed=1, batch_size=16, num_epochs=100, num_workers=8)
    return redirect("/")

def test_3(request):
    updated_fields = {
        ModelField.COMPLETION_STATUS:  " 10 %",
        ModelField.TRAINING_LOSSES: [1, 0.8, 0.7, 0.6, 0.5, 0.5, 0.8, 0.7, 0.9, 0.4],
        ModelField.VALIDATION_LOSSES: [2, 1, 0.9, 0.85, 0.75, 0.4, 0.8, 0.9, 0.4, 0.5]
    }
    
    JsonManager.update_model_data("Hello", updated_fields=updated_fields)
    
    return redirect("/")

def test_4(request):
    with H5Dataset(PathFinder.get_complet_path(f"super_resolution/dataset/evaluation/Set5_{ImageColorConverter.BGR2YCrCb.name}_x2.hdf5")) as loader:
        print(f"Total images in dataset: {len(loader)}")

        # Example: Load and display the first image pair
        low_res, high_res = loader.get_raw_data(0)
        
        print(f"Low Resolution Image Shape: {low_res.shape}")
        print(f"High Resolution Image Shape: {high_res.shape}")
        
        # Channels X Height X Width -> Height X Width X Channels
        cv2.imshow("Low Resolution", low_res)
        cv2.imshow("High Resolution", high_res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return HttpResponse("Hello, world. You're at the polls index.")

def test(request):
    
    input_dataset = "super_resolution/base_dataset/"
    
    output_dataset = "super_resolution/dataset/"
    
    scale = 2 
    
    dataset = "Set5"
    
    mode = ImageColorConverter.BGR2YCrCb
    
    # Create Training file
    create_h5_image_file(input_path = PathFinder.get_complet_path(f"{input_dataset}evaluation/{dataset}"),
                         scale = scale,
                         output_path = PathFinder.get_complet_path(f"{output_dataset}evaluation/{dataset}_{mode.name}_x{scale}.hdf5"),
                         mode = mode)
    
    return HttpResponse("Hello, world. You're at the polls index.")

