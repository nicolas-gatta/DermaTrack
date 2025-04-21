import os
import cv2
import re

from django.template.loader import render_to_string
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.contrib.auth.decorators import login_required
from django.conf import settings


from super_resolution.services.SRCNN import train as srcnn_train
from super_resolution.services.ESRGAN import train as esrgan_train
from super_resolution.services.SRGAN import train as srgan_train
from super_resolution.services.utils.prepare_dataset import create_h5_image_file, ResizeRule
from super_resolution.services.utils.json_manager import JsonManager
from super_resolution.services.utils.image_converter import ImageColorConverter
from super_resolution.services.utils.super_resolution import SuperResolution
from utils.unique_filename import get_unique_filename

from utils.checks import group_and_super_user_checks

_model = None
__test_model = None

@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def training_model(request):
    
    output_path = os.path.join(settings.BASE_DIR, "super_resolution","models")
    
    model_name = get_unique_filename(model_name = f"{request.POST['name']}.pth", output_path = output_path)
    
    architecture = request.POST["architecture"]
    
    scale = int(request.POST["scale"])
    
    mode = request.POST["mode"]
    
    invert_mode = re.sub(r"^(.*)2(.*)$", r"\2here\1", mode).replace("here","2")
    
    learning_rate = float(request.POST["learning-rate"])
    
    batch_size = int(request.POST["batch-size"])
    
    num_epochs = int(request.POST["num-epochs"])
    
    seed = int(request.POST["seed"])
    
    num_workers = int(request.POST["num-workers"])
    
    train_dataset = request.POST["train-dataset"]
    
    valid_dataset = request.POST["valid-dataset"]
    
    eval_dataset = request.POST["eval-dataset"]
    
    resize_rule = None
    
    patch_size = None
    
    stride = None
    
    resize_to_output = architecture in ["SRCNN"]
    
    if ("image-option" in request.POST):
        
        if(request.POST["image-option"] == "resize"):
            resize_rule = "BIGGEST" if int(request.POST["resize-rule"]) == 1 else "SMALLEST"
            
        elif(request.POST["image-option"] == "subdivise"):
            patch_size = int(request.POST["patch-size"])
            stride = int(patch_size * (float(request.POST["overlaying"]) / 100.0))
    
    train_file, valid_file, eval_file = [_dataset_exist_or_create(dataset = dataset, mode = mode, scale = scale, category = category, 
                                                                  patch_size = patch_size, stride = stride, resize_rule = resize_rule, 
                                                                  resize_to_output = resize_to_output) 
                                         for dataset, category in [(train_dataset, "training"), 
                                                                   (valid_dataset, "validation"), 
                                                                   (eval_dataset, "evaluation")] 
                                         ]
    
    JsonManager.training_results_to_json(architecture = architecture, stride = stride, patch_size = patch_size, resize_rule = resize_rule, 
                                                    model_name = model_name, train_file = train_dataset, valid_file = valid_dataset, 
                                                    eval_file = eval_dataset, mode = mode, scale = scale, learning_rate = learning_rate, seed = seed, 
                                                    batch_size = batch_size, num_epochs = num_epochs, num_workers = num_workers)
    
    match(architecture):
        
        case "SRCNN":
            srcnn_train.train_model(
                model_name = model_name,
                train_file = train_file, 
                valid_file = valid_file,
                eval_file = eval_file, 
                output_path = output_path,
                mode = mode,
                scale = scale,
                invert_mode = invert_mode,
                patch_size= patch_size, 
                stride = stride,
                learning_rate = learning_rate,  
                seed = seed, 
                batch_size = batch_size,
                num_epochs = num_epochs,
                num_workers = num_workers)
                
        case "SRGAN":
            srgan_train.train_model(
                model_name = model_name,
                train_file = train_file, 
                valid_file = valid_file,
                eval_file = eval_file, 
                output_path = output_path,
                mode = mode,
                scale = scale,
                invert_mode = invert_mode,
                patch_size= patch_size, 
                stride = stride,
                learning_rate = learning_rate,  
                seed = seed, 
                batch_size = batch_size,
                num_epochs = num_epochs,
                num_workers = num_workers)
                        
        case "ESRGAN":
            esrgan_train.train_model(
                model_name = model_name,
                train_file = train_file, 
                valid_file = valid_file,
                eval_file = eval_file, 
                output_path = output_path,
                mode = mode,
                scale = scale,
                invert_mode = invert_mode,
                patch_size= patch_size, 
                stride = stride,
                learning_rate = learning_rate,  
                seed = seed, 
                batch_size = batch_size,
                num_epochs = num_epochs,
                num_workers = num_workers)
        case _:
            pass
        
    return render(request, 'partial/model_form.html', {"form": None})
    

def _dataset_exist_or_create(dataset: str, mode: str, scale: int, category: str, patch_size: int, stride: int, resize_rule: str, resize_to_output: bool):

    file_name = f"{dataset}_{mode}_x{scale}"
    
    c_resize_rule = None
            
    preprocessing_required = (category != "evaluation") and (patch_size != None and stride != None) or resize_rule != None
    
    if preprocessing_required:
        if patch_size != None and stride != None:
            file_name += f"_{patch_size}_s{stride}"
            
        elif resize_rule != None:
            file_name += f"_{resize_rule}"
            c_resize_rule = ResizeRule[resize_rule]
            
    if not resize_to_output:
        file_name += f"_nrto"
    
    output_path = os.path.join(settings.BASE_DIR, "super_resolution", "datasets", category, f"{file_name}.hdf5")
    
    if not os.path.exists(output_path):
        create_h5_image_file(input_path = os.path.join(settings.BASE_DIR, "super_resolution", "base_datasets", category, dataset),
                            scale = scale, output_path = output_path, mode = ImageColorConverter[mode], patch_size = patch_size,
                            stride = stride, resize_rule = c_resize_rule, preprocessing_required = preprocessing_required, 
                            resize_to_output = resize_to_output)
    return output_path
    
@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def show_models(request):
    if request.headers.get('HX-Request'):
        models = JsonManager.load_training_results()
        return render(request, 'partial/show_models.html', {"models": models})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def model_form(request):
    if request.headers.get('HX-Request'):
        #form = TrainingForm()
        return render(request, 'partial/model_form.html', {"form": None})
    
@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def load_test_model(request, model_name):
    global __test_model
    if model_name != "":
        model_path = os.path.join(settings.BASE_DIR, "super_resolution", "models", f"{model_name}")
        __test_model = SuperResolution(model_path = model_path)
        
        return HttpResponse("Model Loaded")

@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def apply_test_sr(request, image_name):
    global __test_model
    if (__test_model != None and image_name != ""):
        output_path = os.path.join(settings.MEDIA_ROOT, "output_test")
        image_path = os.path.join(output_path, "degraded_image.png")
        filename = "super_resolution_image.png"
        __test_model.apply_super_resolution(image_path = image_path, output_path = output_path, filename = filename)
    
        return JsonResponse({'url': os.path.join(settings.MEDIA_URL, "output_test", filename)})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def dataset_form(request):
    if request.headers.get('HX-Request'):
        #form = TrainingForm()
        return render(request, 'partial/dataset_form.html', {"form": None})
    
@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def test_model_view(request):
    if request.headers.get('HX-Request'):
        #form = TrainingForm()
        return render(request, 'partial/test_model.html', {"form": None})
    
@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def create_dataset(request):
    
    _dataset_exist_or_create(dataset = request.POST["dataset"], 
                             mode = request.POST["mode"], 
                             scale = int(request.POST["scale"]), 
                             category = request.POST["category"])
    
    return render(request, 'partial/dataset_form.html', {"form": None})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def get_datasets(request, category):
    base_path = os.path.join(settings.BASE_DIR, "super_resolution", "base_datasets", category)
    try:
        datasets = os.listdir(base_path)
    except FileNotFoundError:
        datasets = []
    
    return JsonResponse({'datasets': datasets})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def get_all_test_images(request):
    
    base_path = os.path.join(settings.MEDIA_ROOT, "test")
    
    try:
        images = os.listdir(base_path)
    except FileNotFoundError:
        images = []
    
    return JsonResponse({'images': images})


@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def get_test_image(request, name):
    
    base_path = os.path.join(settings.MEDIA_URL, "test", name)
    
    return JsonResponse({'url': base_path})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def degrade_and_save_image(request, name, scale):
    
    base_path = os.path.join(settings.MEDIA_ROOT, "test", name)
    
    output_path = os.path.join(settings.MEDIA_ROOT, "output_test", "degraded_image.png")
    
    image = cv2.imread(base_path)
    
    degraded_image = cv2.resize(image, (image.shape[1] // scale, image.shape[0] // scale), interpolation = cv2.INTER_CUBIC)
    
    degraded_image = cv2.resize(degraded_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    cv2.imwrite(output_path, degraded_image)

    return JsonResponse({'url': os.path.join(settings.MEDIA_URL, "output_test", "degraded_image.png")})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def get_models(request):
    base_path = os.path.join(settings.BASE_DIR, "super_resolution", "models")
    try:
        models = os.listdir(base_path)
    except FileNotFoundError:
        models = []
    
    return JsonResponse({'models': models})
