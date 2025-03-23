from django.urls import path

from . import views

urlpatterns = [
    path('video_stream', views.video_stream, name='video_stream'),
    path('detect', views.detect, name='detect'),
    path('save_images/<int:visit_id>/', views.save_images, name = 'save_images')
]