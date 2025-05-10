from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from landmark_detection.services.pose_estimator import detect_body_part
from django.views.decorators.csrf import csrf_exempt
from core.models import BodyPart, VisitBodyPart, Visit

import base64
import json
import os
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
def save_images(request):
    if request.method == "POST":
        data = json.loads(request.body)
        image = data.get("image", None)
        body_part = BodyPart.objects.get(pk = data.get("bodyPartId", None))
        visit = Visit.objects.get(pk = data.get("visitId", None))
        
        folder_path = os.path.join(settings.MEDIA_ROOT, "visits", f"visit_{visit.pk}", body_part.name)
        
        os.makedirs(folder_path, exist_ok=True)
  
        _, encoded = image.split(',', 1)
        
        img_binary = base64.b64decode(encoded)
        
        num_image = VisitBodyPart.objects.filter(visit_id=visit.pk).count()
        
        filename = f"image_{num_image + 1}.png"
        
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, "wb") as f:
            f.write(img_binary)
        
        VisitBodyPart.objects.create(
            image_path=file_path,
            comment=None,
            body_part=body_part,
            visit=visit
        )

        return JsonResponse({"status": "success"})

    return JsonResponse({"status": "error", "message": "Invalid request method"}, status=400)

def get_body_parts(request):
    if request.method == "GET":
        body_part = list(BodyPart.objects.values_list("pk", "name"))
        return JsonResponse(body_part, safe=False)