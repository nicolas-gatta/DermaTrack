from django.urls import path

from . import views

urlpatterns = [
    path('video_stream/', views.video_stream, name='video_stream'),
    path('detect/', views.detect, name='detect'),
    path('save_image/', views.save_image, name = 'save_image'),
    path('save_image_without_db_update/', views.save_image_without_db_update, name = 'save_image_without_db_update'),
    path('save_image_with_db_update/', views.save_image_with_db_update, name = 'save_image_with_db_update'),
    path('create_visit_body_part/', views.create_visit_body_part, name = 'create_visit_body_part'),
    path('get_body_parts/', views.get_body_parts, name = 'get_body_parts'),
]