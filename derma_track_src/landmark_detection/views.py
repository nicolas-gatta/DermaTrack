from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from landmark_detection.services.pose_estimator import detect_body_part
from django.views.decorators.csrf import csrf_exempt
from core.models import BodyPart, VisitBodyPart, Visit

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
        distance = data.get("distance", None)
        index = data.get("index", None)
        image_height = data.get("imageHeigth", None)
        image_width = data.get("imageWidth", None)
        pixel_size = data.get("pixelSize", None)
        preview_height = 160
        preview_width = 160
        
        folder_path = os.path.join(settings.MEDIA_ROOT, "visits", f"visit_{visit.pk}", body_part.name)
        
        os.makedirs(folder_path, exist_ok=True)
  
        _, encoded = image.split(',', 1)
        
        img_binary = base64.b64decode(encoded)
        
        num_image = VisitBodyPart.objects.filter(visit=visit).count()
        
        filename = f"image_{num_image + 1 + index}.png"
        
        preview_filename = f"preview_image_{num_image + 1 + index}.png"
        
        file_path, preview_path = os.path.join(folder_path, filename), os.path.join(folder_path, preview_filename)
        
        image = cv2.imdecode(np.frombuffer(img_binary, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        cv2.imwrite(file_path, image)
        
        cv2.imwrite(preview_path, cv2.resize(image, (preview_height, preview_width)))
        
        visit_body_part = VisitBodyPart(
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
        
        visit_body_part.save()

        return JsonResponse({"status": "success", "image": preview_filename, "visitId": visit.pk, "bodyPart": body_part.name}, status = 200)

    return JsonResponse({"status": "error", "message": "Invalid request method"}, status = 400)

def get_body_parts(request):
    if request.method == "GET":
        body_part = list(BodyPart.objects.values_list("pk", "name"))
        return JsonResponse(body_part, safe=False)