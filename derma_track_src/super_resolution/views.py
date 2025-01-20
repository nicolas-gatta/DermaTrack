import numpy as np
from PIL import Image

from django.http import HttpResponse
from django.template.loader import render_to_string
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required

from .modules.SRCNN import train
from .modules.utils.preprocessing import create_h5_image_file
from .forms.training_form import TrainingForm
from .modules.utils.json_manager import JsonManager

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
    create_h5_image_file(image_folder = PathFinder.get_complet_path(f"{base_string}dataset/{request.POST['training_dataset']}"),
                         scale = scale,
                         output_path = train_file,
                         mode = "BGR_to_YCrCb")
    
    # Create Evaluation file
    create_h5_image_file(image_folder = PathFinder.get_complet_path(f"{base_string}dataset/{request.POST['eval_dataset']}"),
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
    models = JsonManager.load_training_results()
    return HttpResponse(render_to_string('partial/show_models.html', {"models": models}, request=request))

@login_required
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def model_form(request):
    form = TrainingForm()
    return HttpResponse(render_to_string('partial/model_form.html', {"form": form}, request=request))

@login_required
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def training_model(request):
    pass

def test(request):
    JsonManager.training_results_to_json("SRCNN", "Hello", "blablabla", "blueblueblue", 1e-10, 1, 16, 100, 8, [2,1,0.5,0.3,0.4], [4,5,1,0.8,0.2], 240)
    JsonManager.training_results_to_json("SRGAN", "Hello", "blablabla", "blueblueblue", 1e-10, 1, 16, 100, 8, [2,1,0.5,0.3,0.4], [4,5,1,0.8,0.2], 240)
    JsonManager.training_results_to_json("ESRGAN", "Hello", "blablabla", "blueblueblue", 1e-10, 1, 16, 100, 8, [2,1,0.5,0.3,0.4], [4,5,1,0.8,0.2], 240)
    return redirect("/")

