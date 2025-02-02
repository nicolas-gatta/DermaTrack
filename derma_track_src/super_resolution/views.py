import os
import cv2
import numpy as np
import torch

from django.http import HttpResponse
from django.template.loader import render_to_string
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.contrib.auth.decorators import login_required
from django.conf import settings


from .modules.SRCNN import train as srcnn_train
from .modules.ESRGAN import train as esrgan_train
from .modules.SRGAN import train as srgan_train
from .modules.utils.preprocessing import create_h5_image_file
from .forms.training_form import TrainingForm
from .modules.utils.json_manager import JsonManager, ModelField
from .modules.utils.image_converter import ImageColorConverter, ImageConverter
from .modules.utils.dataloader import H5ImagesDataset

from utils.checks import group_and_super_user_checks
# Create your views here.

@login_required
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def training_model(request):
    
    model_name = request.POST['name']
    
    architecture = request.POST["architecture"]
    
    scale = int(request.POST["scale"])
    
    mode = request.POST["mode"]
    
    learning_rate = float(request.POST["learning-rate"])
    
    batch_size = int(request.POST["batch-size"])
    
    num_epochs = int(request.POST["num-epochs"])
    
    seed = int(request.POST["seed"])
    
    num_workers = int(request.POST["num-workers"])
    
    train_dataset = request.POST["train-dataset"]
    
    valid_dataset = request.POST["valid-dataset"]
    
    eval_dataset = request.POST["eval-dataset"]
    
    output_dir = os.path.join(settings.BASE_DIR, "super_resolution","models", f"{model_name}_{learning_rate}_{batch_size}_{num_epochs}_x{scale}.pth")
    
    train_file, valid_file, eval_file = [_dataset_exist_or_create(dataset = dataset, mode = mode, scale = scale, category = category) 
                                         for dataset, category in [(train_dataset, "training"), 
                                                                   (valid_dataset, "validation"), 
                                                                   (eval_dataset, "evaluation")] 
                                         ]
    
    model_name = JsonManager.training_results_to_json(architecture = architecture, model_name = model_name, train_file = train_dataset, valid_file = valid_dataset, 
                                         eval_file = eval_dataset, mode = mode, scale = scale, learning_rate = learning_rate, seed = seed, 
                                         batch_size = batch_size, num_epochs = num_epochs, num_workers = num_workers)
    
    match(architecture):
        
        case "SRCNN":
            srcnn_train.train_model(
                model_name = model_name,
                train_file = train_file, 
                valid_file = valid_file,
                eval_file = eval_file, 
                output_dir = output_dir,
                learning_rate = learning_rate, 
                seed = seed, 
                batch_size = batch_size,
                num_epochs = num_epochs,
                num_workers = num_workers)
                
        case "SRGAN":
            srgan_train.train_model(
                train_file = train_file, 
                valid_file = valid_file,
                eval_file = eval_file, 
                output_dir = output_dir,
                learning_rate = learning_rate, 
                seed = seed, 
                batch_size = batch_size,
                num_epochs = num_epochs,
                num_workers = num_workers)
                        
        case "ESRGAN":
            esrgan_train.train_model(
                train_file = train_file, 
                valid_file = valid_file,
                eval_file = eval_file, 
                output_dir = output_dir,
                learning_rate = learning_rate, 
                seed = seed, 
                batch_size = batch_size,
                num_epochs = num_epochs,
                num_workers = num_workers)
        case _:
            pass
        
    return render(request, 'partial/model_form.html', {"form": None})
    

def _dataset_exist_or_create(dataset, mode, scale, category):
    
    output_path = os.path.join(settings.BASE_DIR, "super_resolution", "datasets", f"{category}/{dataset}_{mode}_x{scale}.hdf5")
    
    if not os.path.exists(output_path):
        create_h5_image_file(input_path = os.path.join(settings.BASE_DIR, "super_resolution", "base_datasets", category, dataset),
                            scale = scale,
                            output_path = output_path,
                            mode = ImageColorConverter[mode])
    return output_path
    
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
def dataset_form(request):
    if request.headers.get('HX-Request'):
        #form = TrainingForm()
        return render(request, 'partial/dataset_form.html', {"form": None})
    
@login_required
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def test_model_view(request):
    if request.headers.get('HX-Request'):
        #form = TrainingForm()
        return render(request, 'partial/test_model.html', {"form": None})
    
@login_required
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def create_dataset(request):
    
    _dataset_exist_or_create(dataset = request.POST["dataset"], 
                             mode = request.POST["mode"], 
                             scale = int(request.POST["scale"]), 
                             category = request.POST["category"])
    
    return render(request, 'partial/dataset_form.html', {"form": None})

@login_required
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def get_datasets(request, category):
    base_path = os.path.join(settings.BASE_DIR, "super_resolution", "base_datasets", category)
    try:
        datasets = os.listdir(base_path)
    except FileNotFoundError:
        datasets = []
    
    return JsonResponse({'datasets': datasets})

@login_required
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def get_test_images(request):
    
    base_path = os.path.join(settings.MEDIA_ROOT, "test")
    
    try:
        images = os.listdir(base_path)
    except FileNotFoundError:
        images = []
    
    return JsonResponse({'images': images})

@login_required
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def degrade_and_save_image(request, name, scale):
    
    base_path = os.path.join(settings.MEDIA_ROOT, "test", name)
    
    output_path = os.path.join(settings.MEDIA_ROOT, "output_test", "degraded_image.png")
    
    image = cv2.imread(base_path)
    
    degraded_image = cv2.resize(image, (image.shape[1] // scale, image.shape[0] // scale), interpolation = cv2.INTER_CUBIC)
    
    degraded_image = cv2.resize(degraded_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    cv2.imwrite(output_path, degraded_image)

    return JsonResponse({'url': os.path.join(settings.MEDIA_URL, "output_test", "degraded_image.png")})

@login_required
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def get_test_image(request, name):
    
    base_path = os.path.join(settings.MEDIA_URL, "test", name)
    
    return JsonResponse({'url': base_path})

@login_required
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def get_models(request):
    base_path = os.path.join(settings.BASE_DIR, "super_resolution", "models")
    try:
        models = os.listdir(base_path)
    except FileNotFoundError:
        models = []
    
    return JsonResponse({'models': models})


def test_2(request):
    JsonManager.training_results_to_json(architecture="SRCNN", model_name="Hello", train_file="blablabla", eval_file="blueblueblue", learning_rate=1e-10, seed=1, batch_size=16, num_epochs=100, num_workers=8)
    JsonManager.training_results_to_json(architecture="SRGAN", model_name="Hello", train_file="blablabla", eval_file="blueblueblue", learning_rate=1e-10, seed=1, batch_size=16, num_epochs=100, num_workers=8)
    JsonManager.training_results_to_json(architecture="ESRGAN", model_name="Hello", train_file="blablabla", eval_file="blueblueblue", learning_rate=1e-10, seed=1, batch_size=16, num_epochs=100, num_workers=8)
    return redirect("/")

def test_3(request):
    
    return redirect("/")

def test_4(request):
        
    return redirect("/")

def test(request):

    return redirect("/")

