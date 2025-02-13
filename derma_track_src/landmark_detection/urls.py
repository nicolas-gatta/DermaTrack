from django.urls import path

from . import views

urlpatterns = [
    path('video_stream/', views.video_stream, name='video_stream'),
    path('detect/', views.landmark_detection_view, name='landmark-detection')
]