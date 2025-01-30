import os
import cv2
import numpy as np
import h5py

from django.http import HttpResponse
from django.template.loader import render_to_string
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.contrib.auth.decorators import login_required


from .modules.SRCNN import train as srcnn_train
from .modules.ESRGAN import train as esrgan_train
from .modules.SRGAN import train as srgan_train
from .modules.utils.preprocessing import create_h5_image_file
from .forms.training_form import TrainingForm
from .modules.utils.json_manager import JsonManager, ModelField
from .modules.utils.image_converter import ImageColorConverter, ImageConverter
from .modules.utils.dataloader import H5Dataset

from utils.checks import group_and_super_user_checks
from utils.path_finder import PathFinder
# Create your views here.

@login_required
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def training_model(request):
    
    architecture = request.POST["architecture"]
    
    scale = int(request.POST["scale"])
    
    mode = request.POST["mode"]
    
    validation_dataset = request.POST["valid-dataset"]
    
    training_dataset = request.POST["train-dataset"]
    
    evaluation_dataset = request.POST["eval-dataset"]
    
    learning_rate = float(request.POST["learning-rate"])
    
    batch_size = int(request.POST["batch-size"])
    
    num_epochs = int(request.POST["num-epochs"])
    
    seed = int(request.POST["seed"])
    
    num_workers = int(request.POST["num-workers"])
    
    output_dir = PathFinder.get_complet_path(f"super_resolution/modules/{architecture}/output/{request.POST['name']}_{learning_rate}_{batch_size}_{num_epochs}_x{scale}")
    
    train_file, valid_file, eval_file = [dataset_exist_or_create(dataset = dataset, mode = mode, scale = scale, category = category) 
                                         for dataset, category in [(training_dataset, "training"), (validation_dataset, "validation"), (evaluation_dataset, "evaluation")] ]
    
    match(architecture):
        
        case "SRCNN":
            srcnn_train.train_model(
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
    

def dataset_exist_or_create(dataset, mode, scale, category):
    
    output_path = PathFinder.get_complet_path(f"super_resolution/dataset/{category}/{dataset}_{mode}_x{scale}.hdf5")
    
    if not os.path.exists(output_path):
        create_h5_image_file(input_path = PathFinder.get_complet_path(f"super_resolution/base_dataset/{category}/{dataset}"),
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
def create_dataset(request):
    
    scale = int(request.POST["scale"])
    
    dataset = request.POST["dataset"]
    
    mode = request.POST["mode"]
    
    category = request.POST["category"]
    
    create_h5_image_file(input_path = PathFinder.get_complet_path(f"super_resolution/base_dataset/{category}/{dataset}"),
                         scale = scale,
                         output_path = PathFinder.get_complet_path(f"super_resolution/dataset/{category}/{dataset}_{mode}_x{scale}.hdf5"),
                         mode = ImageColorConverter[mode])
    
    return render(request, 'partial/dataset_form.html', {"form": None})

@login_required
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def get_datasets(request, category):
    base_path = PathFinder.get_complet_path(os.path.join("super_resolution", "base_dataset", category))
    try:
        datasets = os.listdir(base_path)
    except FileNotFoundError:
        datasets = []
    
    return JsonResponse({'datasets': datasets})


def test_2(request):
    JsonManager.training_results_to_json(architecture="SRCNN", model_name="Hello", train_file="blablabla", eval_file="blueblueblue", learning_rate=1e-10, seed=1, batch_size=16, num_epochs=100, num_workers=8)
    JsonManager.training_results_to_json(architecture="SRGAN", model_name="Hello", train_file="blablabla", eval_file="blueblueblue", learning_rate=1e-10, seed=1, batch_size=16, num_epochs=100, num_workers=8)
    JsonManager.training_results_to_json(architecture="ESRGAN", model_name="Hello", train_file="blablabla", eval_file="blueblueblue", learning_rate=1e-10, seed=1, batch_size=16, num_epochs=100, num_workers=8)
    return redirect("/")

def test_3(request):
    updated_fields = {
        ModelField.COMPLETION_STATUS:  "10 %",
        ModelField.TRAINING_LOSSES: [1, 0.8, 0.7, 0.6, 0.5, 0.5, 0.8, 0.7, 0.9, 0.4],
        ModelField.VALIDATION_LOSSES: [2, 1, 0.9, 0.85, 0.75, 0.4, 0.8, 0.9, 0.4, 0.5]
    }
    
    JsonManager.update_model_data("Hello", updated_fields=updated_fields)
    
    return redirect("/")

def test_4(request):
    with H5Dataset(PathFinder.get_complet_path(f"super_resolution/dataset/evaluation/Set5_{ImageColorConverter.BGR2YCrCb.name}_x2.hdf5")) as loader:
        print(f"Total images in dataset: {len(loader)}")

        # Example: Load and display the first image pair
        low_res, _ = loader.get_raw_data(0)
        _, high_res = loader[0]
        
        print(f"Low Resolution Image Shape: {low_res.shape}")
        print(f"High Resolution Image Shape: {high_res.shape}")
        
        output_image = high_res.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        output_image = (output_image * 255).astype(np.uint8)  # Rescale to [0,255]
        # output_image = ImageConverter.convert_image(output_image, ImageColorConverter.YCrCb2BGR)  # Convert back to BGR

        # Channels X Height X Width -> Height X Width X Channels
        cv2.imshow("Low Resolution", low_res)
        cv2.imshow("High Resolution", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return HttpResponse("Hello, world. You're at the polls index.")

def test(request):
    
    h5dataset = h5py.File(PathFinder.get_complet_path(f"super_resolution/dataset/evaluation/Set5_{ImageColorConverter.BGR2YCrCb.name}_x2.hdf5"))
    
    low_res = h5dataset["low_res"]
    
    h5dataset.close()
    
    print(low_res["image_001"])
    
    return HttpResponse("Hello, world. You're at the polls index.")

