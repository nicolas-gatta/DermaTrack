from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from landmark_detection.services.pose_estimator import detect_body_part
import json

import cv2
# Create your views here.

def video_stream(request):
    return render(request, 'partial/video_stream.html')

def detect(request):
    
    data = json.loads(request.body)
    base64_image = data.get("image", "")
    
    detected_body_part = detect_body_part(base64_image = base64_image)
    return JsonResponse({"body_part": detected_body_part})