import os
import cv2
import re
import torch
import json

from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.contrib.auth.decorators import login_required
from django.conf import settings
from super_resolution.services.SRCNN import train as srcnn_train
from super_resolution.services.ESRGAN import train as esrgan_train
from super_resolution.services.SRGAN import train as srgan_train
from super_resolution.services.EDVR import train as edvr_train
from super_resolution.services.utils.prepare_dataset import dataset_exist_or_create
from super_resolution.services.utils.json_manager import JsonManager, ModelField
from super_resolution.services.utils.super_resolution import SuperResolution
from super_resolution.services.utils.model_evaluation import ModelEvaluation
from core.models import VisitBodyPart
from utils.unique_filename import get_unique_filename
from utils.checks import group_and_super_user_checks

__model = None
__test_model = None

@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def training_model(request):
    """
    Trains a super-resolution model (SRCNN, SRGAN, ESRGAN, or EDVR) based on user-provided parameters.

    It supports training from scratch or fine-tuning a pre-trained model.
    It can also handle advanced options such as image subdivision, resizing, and rotation.

    Args:
        request (HttpRequest): The incoming HTTP POST request containing training parameters.

    Returns:
        HttpResponse: Renders the model form view after training starts.
    """
    
    output_path = os.path.join(settings.BASE_DIR, "super_resolution","models")
    
    model_name = get_unique_filename(model_name = f"{request.POST['name']}.pth", output_path = output_path)
    
    learning_rate = float(request.POST["learning-rate"])
    
    batch_size = int(request.POST["batch-size"])
    
    num_epochs = int(request.POST["num-epochs"])
    
    seed = int(request.POST["seed"])
    
    num_workers = int(request.POST["num-workers"])
    
    train_dataset = request.POST["train-dataset"]
    
    valid_dataset = request.POST["valid-dataset"]
    
    eval_dataset = request.POST["eval-dataset"]
    
    pretrain_model = None
    
    max_angle_rotation = None
    
    angle_rotation_step = None 
    
    resize_rule = None
    
    patch_size = None
    
    stride = None
        
    if request.POST["pretrain-model"] != "":
        pretrain_model = request.POST["pretrain-model"]
        model_info = torch.load(os.path.join(output_path, pretrain_model), weights_only=True)
        architecture = model_info["architecture"]
        scale = model_info["scale"]
        mode = model_info["color_mode"]
        invert_mode = model_info["invert_color_mode"]
        patch_size = model_info["patch_size"]
        stride = model_info["stride"]
        multi_input = model_info["multi_input"]
        resize_to_output = model_info["need_resize"]
    else:
        architecture = request.POST["architecture"]
        scale = int(request.POST["scale"])
        mode = request.POST["mode"]
        invert_mode = re.sub(r"^(.*)2(.*)$", r"\2here\1", mode).replace("here","2")
        mode = request.POST["mode"]
        resize_to_output = architecture in ["SRCNN"]
        multi_input = architecture in ["EDVR"]

        if ("image-option" in request.POST):
            
            if(request.POST["image-option"] == "resize"):
                resize_rule = "BIGGEST" if int(request.POST["resize-rule"]) == 1 else "SMALLEST"
                
            elif(request.POST["image-option"] == "subdivise"):
                overlaying = float(request.POST["overlaying"])
                patch_size = int(request.POST["patch-size"])
                stride = int(patch_size * (overlaying / 100.0)) if overlaying != 0.0 else None
        
    if ("image-option-angle" in request.POST):
        max_angle_rotation = int(request.POST["degree"])
        angle_rotation_step = int(request.POST["step-degree"])

    train_file, valid_file, eval_file = [dataset_exist_or_create(dataset = dataset, mode = mode, scale = scale, category = category, 
                                                                  patch_size = patch_size, stride = stride, resize_rule = resize_rule, 
                                                                  resize_to_output = resize_to_output, base_dir = settings.BASE_DIR, 
                                                                  multi_input = multi_input, max_angle_rotation = max_angle_rotation, 
                                                                  angle_rotation_step = angle_rotation_step) 
                                         for dataset, category in [(train_dataset, "training"), 
                                                                   (valid_dataset, "validation"), 
                                                                   (eval_dataset, "evaluation")] 
                                        ]
    
    JsonManager.training_results_to_json(architecture = architecture, stride = stride, patch_size = patch_size, resize_rule = resize_rule, 
                                        model_name = model_name, train_file = train_dataset, valid_file = valid_dataset, 
                                        eval_file = eval_dataset, mode = mode, scale = scale, learning_rate = learning_rate, seed = seed, 
                                        batch_size = batch_size, num_epochs = num_epochs, num_workers = num_workers, pretrain_model = pretrain_model)
    
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
                num_workers = num_workers,
                pretrain_model = pretrain_model)
                
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
                patch_size = patch_size, 
                stride = stride,
                learning_rate = learning_rate,  
                seed = seed, 
                batch_size = batch_size,
                num_epochs = num_epochs,
                num_workers = num_workers,
                pretrain_model = pretrain_model)
                        
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
                num_workers = num_workers,
                pretrain_model = pretrain_model)
            
        case "EDVR":
            edvr_train.train_model(
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
                num_workers = num_workers,
                pretrain_model = pretrain_model)
        case _:
            pass
        
    return render(request, 'partial/model_form.html', {"form": None})


@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def evaluate_model(request):
    """
    Evaluate a super-resolution model (SRCNN, SRGAN, ESRGAN, or EDVR) based on user-provided parameters.

    It can also handle advanced options such as image subdivision, resizing, and rotation.

    Args:
        request (HttpRequest): The incoming HTTP POST request containing evaluation parameters.

    Returns:
        HttpResponse: Renders the model form view after evaluation starts.
    """
    
    model_name = request.POST["model"]
    
    if "BICUBIC" not in model_name:
        
        output_path = os.path.join(settings.BASE_DIR, "super_resolution","models")
            
        model_info = torch.load(os.path.join(output_path, model_name), weights_only=True)
        
        use_bicubic = False
        
        bicubic_scale = None
        
    else:
        model_info = None
        
        output_path = None
        
        use_bicubic = True
        
        bicubic_scale = 2 if "2" in model_name else 4
        
    scale = model_info["scale"] if model_info else bicubic_scale
    
    mode = model_info["color_mode"] if model_info else "BGR2RGB"
    
    resize_to_output = model_info["need_resize"] if model_info else None
    
    multi_input = model_info["multi_input"] if model_info else None
        
    eval_dataset = request.POST["eval-dataset"]
    
    max_angle_rotation = None
    
    angle_rotation_step = None

    if request.POST["degree"] != "" and request.POST["step-degree"] != "":
        max_angle_rotation = int(request.POST["degree"])
        angle_rotation_step = int(request.POST["step-degree"])
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
     
    eval_file = dataset_exist_or_create(dataset = eval_dataset, mode = mode, scale = scale, category = "evaluation", 
                                        patch_size = None, stride = None, resize_rule = None, 
                                        resize_to_output = resize_to_output, base_dir = settings.BASE_DIR,
                                        multi_input = multi_input, max_angle_rotation = max_angle_rotation, angle_rotation_step = angle_rotation_step)
    if use_bicubic:
        JsonManager.training_results_to_json(architecture = None, stride = None, patch_size = None, resize_rule = None, 
                                    model_name = model_name, train_file = None, valid_file = None, 
                                    eval_file = eval_dataset, mode = mode, scale = scale, learning_rate = None, seed = None, 
                                    batch_size = None, num_epochs = None, num_workers = None, pretrain_model = None)
        JsonManager.update_model_data(model_name = model_name, updated_fields={ModelField.COMPLETION_STATUS: "Completed"})

    ModelEvaluation.evaluate_model(model_name = model_name, path_to_model = output_path, device = device, eval_file = eval_file, eval_file_name = eval_dataset, use_bicubic = use_bicubic, bicubic_scale = bicubic_scale)
        
    return render(request, 'partial/evaluate_model_form.html', {"form": None})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def show_models(request):
    """
    Rendering the show_models page.

    This view is protected by login and group/superuser access checks.
    It renders the 'partial/show_models.html' template with form in the context

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered HTML response for the show_models.
    """
    
    if request.headers.get('HX-Request'):
        models = JsonManager.load_training_results()
        return render(request, 'partial/show_models.html', {"models": models})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def model_form(request):
    """
    Rendering the model_form page.

    This view is protected by login and group/superuser access checks.
    It renders the 'partial/model_form.html' template with form in the context

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered HTML response for the model_form.
    """
    
    if request.headers.get('HX-Request'):
        #form = TrainingForm()
        return render(request, 'partial/model_form.html', {"form": None})
    
@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def load_test_model(request, model_name):
    """
    Loads a super-resolution model.
    
    Args:
        request (HttpRequest): The HTTP request object.
        model_name (str): The name of the model file to load.
        
    Returns:
        HttpResponse: A response indicating that the model has been loaded.
    """
    
    global __test_model
    if model_name != "":
        model_path = os.path.join(settings.BASE_DIR, "super_resolution", "models", f"{model_name}")
        __test_model = SuperResolution(model_path = model_path)
        
        return HttpResponse("Model Loaded")

@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def apply_test_sr(request, image_name):
    """
    Applies a super-resolution model to a degraded image and returns the URL of the enhanced image.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        JsonResponse: The url of the image.
    """
    
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
    """
    Rendering the dataset_form page.

    This view is protected by login and group/superuser access checks.
    It renders the 'partial/dataset_form.html' template with form in the context

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered HTML response for the dataset_form.
    """
    
    if request.headers.get('HX-Request'):
        #form = TrainingForm()
        return render(request, 'partial/dataset_form.html', {"form": None})
    
@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def evaluation_form(request):
    """
    Rendering the evaluate_model_form page.

    This view is protected by login and group/superuser access checks.
    It renders the 'partial/evaluate_model_form.html' template with form in the context

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered HTML response for the evaluate_model_form.
    """
    
    if request.headers.get('HX-Request'):
        #form = TrainingForm()
        return render(request, 'partial/evaluate_model_form.html', {"form": None})
    
@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def test_model_view(request):
    """
    Rendering the test_model page.

    This view is protected by login and group/superuser access checks.
    It renders the 'partial/test_model.html' template with form in the context

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered HTML response for the test_model.
    """
    
    if request.headers.get('HX-Request'):
        #form = TrainingForm()
        return render(request, 'partial/test_model.html', {"form": None})
    
@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def create_dataset(request):
    """
    Handles the creation or retrieval of a dataset.
    
    Args:
        request (HttpRequest): The HTTP request object containing POST data
        
    Returns:
        HttpResponse: The rendered HTML response for the dataset form.
    """
    
    dataset_exist_or_create(dataset = request.POST["dataset"], 
                             mode = request.POST["mode"], 
                             scale = int(request.POST["scale"]), 
                             category = request.POST["category"])
    
    return render(request, 'partial/dataset_form.html', {"form": None})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def get_datasets(request, category):
    """
    Retrieve the list of dataset directories for a given category.
    
    Args:
        request (HttpRequest): The HTTP request object.
        category (str): The category name used to locate the datasets.
        
    Returns:
        JsonResponse: A JSON response containing a list of dataset.
    """
    
    base_path = os.path.join(settings.BASE_DIR, "super_resolution", "base_datasets", category)
    try:
        datasets = os.listdir(base_path)
    except FileNotFoundError:
        datasets = []
    
    return JsonResponse({'datasets': datasets})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def get_all_test_images(request):
    """
    Retrieve all test image filenames from the 'test' directory.
    
    Args:
        request (HttpRequest): The HTTP request object.
        
    Returns:
        JsonResponse: A JSON response containing a list of image filenames.
    """
    
    base_path = os.path.join(settings.MEDIA_ROOT, "test")
    
    try:
        images = os.listdir(base_path)
    except FileNotFoundError:
        images = []
    
    return JsonResponse({'images': images})


@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def get_test_image(request, name):
    """
    Returns the URL to a test image.
    
    Args:
        request (HttpRequest): The HTTP request object.
        name (str): The name of the image file.
        
    Returns:
        JsonResponse: A JSON response with the key 'url' pointing to the constructed image URL.
    """
    
    base_path = os.path.join(settings.MEDIA_URL, "test", name)
    
    return JsonResponse({'url': base_path})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def degrade_and_save_image(request, name, scale):
    """
    Degrades an input image by applying Gaussian blur and downscaling, then saves the result.
    
    Args:
        request (HttpRequest): The HTTP request object.
        name (str): The filename of the image to be degraded.
        scale (int): The factor by which to downscale the image dimensions.
        
    Returns:
        JsonResponse: A JSON response containing the URL to the degraded image.
    """
    
    base_path = os.path.join(settings.MEDIA_ROOT, "test", name)
    
    output_path = os.path.join(settings.MEDIA_ROOT, "output_test", "degraded_image.png")
    
    image = cv2.imread(base_path)
    
    blur_image = cv2.GaussianBlur(image, (5, 5), 1.5)
    
    degraded_image = cv2.resize(blur_image, (image.shape[1] // scale, image.shape[0] // scale))

    cv2.imwrite(output_path, degraded_image)

    return JsonResponse({'url': os.path.join(settings.MEDIA_URL, "output_test", "degraded_image.png")})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def get_models(request):
    """
    Retrive all the available model.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        JsonResponse: Contain all the available models.
    """
    
    base_path = os.path.join(settings.BASE_DIR, "super_resolution", "models")
    try:
        models = [
            file for file in os.listdir(base_path)
            if os.path.isfile(os.path.join(base_path, file)) and file.endswith(".pth")
        ]
    except FileNotFoundError:
        models = []
    
    return JsonResponse({'models': models})

def load_model() -> SuperResolution:
    global __model
    
    json_path = os.path.join(settings.BASE_DIR, "super_resolution", "static", "data", "model_selection.json")
    
    with open(json_path, "r") as f:
        model_info = json.load(f)
        model_name = model_info["model_name"]
    
    if __model != None and model_name in __model.path:
        return __model
    elif "BICUBIC" in model_name:
        bicubic_scale = 2 if "2" in model_name else 4
        return SuperResolution(model_path = None, use_bicubic = True, bicubic_scale = bicubic_scale)
    
    else:
        model_path = os.path.join(settings.BASE_DIR, "super_resolution", "models", f"{model_name}")
        return SuperResolution(model_path = model_path)
    

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def apply_sr(request):
    """
    Applies the selected super-resolution model to a medical image.

    The result is saved as a enhanced image and updates the VisitBodyPart model instance.

    Args:
        request (HttpRequest): The HTTP POST request containing 'visit_body_part_id'.

    Returns:
        JsonResponse: Success message when enhancement is done.
    """
    
    if request.method == "POST" :
        
        model = load_model()
        
        data = json.loads(request.body)
        
        visit_body_part = VisitBodyPart.objects.get(pk = data.get("visit_body_part_id", None))
        
        output_path = os.path.join(settings.MEDIA_ROOT, "visits", f"visit_{visit_body_part.visit.pk}", visit_body_part.body_part.name)
        
        filename = f"enchanced_image_{visit_body_part.pk}.enc"
                    
        if model.model_info["multi_input"]:
            output_path, height, width = model.apply_super_resolution(image_path = None, output_path = output_path, filename = filename, folder_path = visit_body_part.multi_image_path, is_encrypted = True)
        else:
            image_path = visit_body_part.image_path.path
            output_path, height, width = model.apply_super_resolution(image_path = image_path, output_path = output_path, filename = filename, is_encrypted = True)
            
        visit_body_part.image_super_height = height
        visit_body_part.image_super_width = width
        visit_body_part.image_super_name = filename
        visit_body_part.image_super_path = output_path
        
        visit_body_part.save()
        
        return JsonResponse({"message": "Enhanced Sucessful"}, status=200)

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def select_model(request):
    """
    Handles the selection of a super-resolution model.
    
    Args:
        request (HttpRequest): The HTTP request object.
        
    Returns:
        HttpResponse: The rendered model selection form template.
    """
    
    if request.method == "POST" :
        
        output_file = os.path.join(settings.BASE_DIR, "super_resolution", "static", "data", "model_selection.json")
            
        selector_model = {
            "model_name": request.POST["model"]
        }

        with open(output_file, "w") as f:
            json.dump(selector_model, f, indent = 4)
        
        return render(request, 'partial/model_selection_form.html', {"form": None})
        
@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def model_selection_form(request):
    """
    Rendering the model_selection_form page.

    This view is protected by login and group/superuser access checks.
    It renders the 'partial/model_selection_form.html' template with form in the context

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered HTML response for the model_selection_form.
    """
    if request.headers.get('HX-Request'):
        #form = TrainingForm()
        return render(request, 'partial/model_selection_form.html', {"form": None})