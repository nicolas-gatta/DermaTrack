from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from landmark_detection.services.pose_estimator import detect_body_part
from django.views.decorators.csrf import csrf_exempt

import base64
import json
import os
import cv2
from pathlib import Path

# Create your views here.

def video_stream(request):
    return render(request, 'partial/video_stream.html')

def detect(request):
    
    data = json.loads(request.body)
    base64_image = data.get("image", "")
    
    detected_body_part = detect_body_part(base64_image = base64_image)
    return JsonResponse({"body_part": detected_body_part})

@csrf_exempt
def save_images(request, visit_id, body_part = "leg"):
    if request.method == "POST":
        data = json.loads(request.body)
        image = data.get("image", None)
        saved_paths = []

        folder_path = os.path.join(settings.MEDIA_ROOT, f"visit_{visit_id}", body_part)
        
        os.makedirs(folder_path, exist_ok=True)
  
        _, encoded = image.split(',', 1)
        
        img_binary = base64.b64decode(encoded)
        
        num_png = len(list(Path(folder_path).glob('*.png')))
        
        filename = f"image_{num_png + 1}.png"
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "wb") as f:
            f.write(img_binary)

        saved_paths.append(file_path)

        return JsonResponse({"status": "success"})

    return JsonResponse({"status": "error", "message": "Invalid request method"}, status=400)