from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from camera.services.pose_estimator import detect_body_part
from django.views.decorators.csrf import csrf_exempt
from core.models import BodyPart, VisitBodyPart, Visit
from image_encryption.services.advanced_encrypted_standard import AES

import base64
import json
import os
import cv2
import numpy as np

# Create your views here.

def video_stream(request):
    return render(request, 'partial/video_stream.html')

def detect(request):
    
    data = json.loads(request.body)
    base64_image = data.get("image", "")
    
    detected_body_part = detect_body_part(base64_image = base64_image)
    return JsonResponse({"body_part": detected_body_part})

@csrf_exempt
def save_image(request):
    if request.method == "POST":
        data = json.loads(request.body)
        
        image = data.get("image", None)
        body_part = BodyPart.objects.get(pk = data.get("bodyPartId", None))
        visit = Visit.objects.get(pk = data.get("visitId", None))
        image_height = data.get("imageHeigth", None)
        image_width = data.get("imageWidth", None)
        distance = data.get("distance", None)
        index = data.get("index", None)
        pixel_size = data.get("pixelSize", None)
        preview_height = 160
        preview_width = 160
        
        folder_path = os.path.join(settings.MEDIA_ROOT, "visits", f"visit_{visit.pk}", body_part.name)
        
        os.makedirs(folder_path, exist_ok=True)
  
        _, encoded = image.split(',', 1)
        
        img_binary = base64.b64decode(encoded)
        
        try:
            last_visit_body_part = VisitBodyPart.objects.latest('pk')
            num_image = last_visit_body_part.pk
        except VisitBodyPart.DoesNotExist:
            num_image = 1
          
        multi_input_path = os.path.join(folder_path, f"image_{num_image}")
        
        os.makedirs(multi_input_path, exist_ok=True)
        
        image_name = f"image_{num_image}"
        
        if index:
            image_name += f"_{index}"
            
        filename = f"{image_name}.enc"
        
        preview_filename = f"preview_{image_name}.png"
        
        file_path, preview_path = os.path.join(multi_input_path, filename), os.path.join(folder_path, preview_filename)
        
        image = cv2.imdecode(np.frombuffer(img_binary, dtype=np.uint8), cv2.IMREAD_COLOR)    
        
        with open(file_path, "wb") as f:
            f.write(AES.encrypt_message(img_binary))
        
        cv2.imwrite(preview_path, cv2.resize(image, (preview_height, preview_width)))
        
        visit_body_part = VisitBodyPart.objects.create(
            image_name = filename,
            image_path = file_path,
            image_height = image_height,
            image_width = image_width,
            image_preview_name = preview_filename,
            image_preview_path = preview_path,
            image_preview_height = preview_height,
            image_preview_width = preview_width,
            distance_from_subject = distance,
            pixel_size = pixel_size,
            body_part = body_part,
            visit = visit
        )

        return JsonResponse({"status": "success", "visit_body_part_id": visit_body_part.pk ,"image": preview_filename, "visitId": visit.pk, "bodyPart": body_part.name}, status = 200)

    return JsonResponse({"status": "error", "message": "Invalid request method"}, status = 400)


@csrf_exempt
def create_visit_body_part(request):
    if request.method == "POST":
        data = json.loads(request.body)
        body_part = BodyPart.objects.get(pk = data.get("bodyPartId", None))
        visit = Visit.objects.get(pk = data.get("visitId", None))
        distance = data.get("distance", None)
        pixel_size = data.get("pixelSize", None)
        
        visit_body_part = VisitBodyPart.objects.create(
            distance_from_subject = distance,
            pixel_size = pixel_size,
            body_part = body_part,
            visit = visit
        )
        
        os.makedirs(os.path.join(settings.MEDIA_ROOT, "visits", f"visit_{visit.pk}", body_part.name), exist_ok=True)
        
        return JsonResponse({"status": "success", "visit_body_part_id": visit_body_part.pk, "visitId": visit.pk, "bodyPart": body_part.name}, status = 200)

    return JsonResponse({"status": "error", "message": "Invalid request method"}, status = 400)


@csrf_exempt
def save_image_without_db_update(request):
    if request.method == "POST":
        
        data = json.loads(request.body)
        image = data.get("image", None)
        index = data.get("index", None)
        visit_body_part_id = data.get("visit_body_part_id", None)
        body_part = data.get("body_part", None)
        visit = data.get("visit", None)
        index = data.get("index", None)

        visit_body_part = VisitBodyPart.objects.get(pk = visit_body_part_id)
        
        num_image = visit_body_part.pk
                
        folder_path = os.path.join(settings.MEDIA_ROOT, "visits", f"visit_{visit}", body_part, f"image_{num_image}")

        os.makedirs(folder_path, exist_ok=True)
        
        _, encoded = image.split(',', 1)
        
        img_binary = base64.b64decode(encoded)
        
        image_name = f"image_{num_image}_{index}"
            
        filename = f"{image_name}.enc"
        
        file_path = os.path.join(folder_path, filename)
        
        image = cv2.imdecode(np.frombuffer(img_binary, dtype=np.uint8), cv2.IMREAD_COLOR)    
        
        with open(file_path, "wb") as f:
            f.write(AES.encrypt_message(img_binary))
        
        return JsonResponse({"status": "success"}, status = 200)

    return JsonResponse({"status": "error", "message": "Invalid request method"}, status = 400)

@csrf_exempt
def save_image_with_db_update(request):
    if request.method == "POST":
        data = json.loads(request.body)
        
        image = data.get("image", None)
        index = data.get("index", None)
        image_height = data.get("imageHeigth", None)
        image_width = data.get("imageWidth", None)
        visit_body_part_id = data.get("visit_body_part_id", None)
        image_height = data.get("imageHeigth", None)
        image_width = data.get("imageWidth", None)
        preview_height = 160
        preview_width = 160
        
        visit_body_part = VisitBodyPart.objects.get(pk = visit_body_part_id)
                
        folder_path = os.path.join(settings.MEDIA_ROOT, "visits", f"visit_{visit_body_part.visit.pk}", visit_body_part.body_part.name)

        os.makedirs(folder_path, exist_ok=True)
        
        _, encoded = image.split(',', 1)
        
        img_binary = base64.b64decode(encoded)
        
        num_image = visit_body_part.pk
          
        multi_input_path = os.path.join(folder_path,f"image_{num_image}")
        
        image_name = f"image_{num_image}_{index}"
            
        filename, preview_filename = f"{image_name}.enc", f"preview_{image_name}.png"
        
        file_path, preview_path = os.path.join(multi_input_path, filename), os.path.join(folder_path, preview_filename)
        
        image = cv2.imdecode(np.frombuffer(img_binary, dtype=np.uint8), cv2.IMREAD_COLOR)    
        
        with open(file_path, "wb") as f:
            f.write(AES.encrypt_message(img_binary))

        cv2.imwrite(preview_path, cv2.resize(image, (preview_height, preview_width)))
        
        visit_body_part.image_preview_name = preview_filename
        visit_body_part.image_preview_path = preview_path 
        visit_body_part.image_preview_height = preview_height
        visit_body_part.image_preview_width = preview_width
        
        visit_body_part.image_name = filename
        visit_body_part.image_path = file_path
        visit_body_part.image_height = image_height
        visit_body_part.image_width = image_width
        
        visit_body_part.multi_image_path = multi_input_path
        
        visit_body_part.save()
        
        return JsonResponse({"status": "success", "visit_body_part_id": visit_body_part.pk ,"image": preview_filename, "visitId": visit_body_part.visit.pk, "bodyPart": visit_body_part.body_part.name}, status = 200)

    return JsonResponse({"status": "error", "message": "Invalid request method"}, status = 400)   
        
def get_body_parts(request):
    if request.method == "GET":
        body_part = list(BodyPart.objects.values_list("pk", "name"))
        return JsonResponse(body_part, safe=False)