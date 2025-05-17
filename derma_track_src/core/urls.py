from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    
    path("patient_profile", views.patient_profile, name = "patient_profile"),
    path("patient_list", views.patient_list, name = "patient_list"),
    
    path('visit_list', views.visit_list, name = 'visit_list'),
    path('visit_status', views.visit_status_change, name = 'visit_status'),
    path('visit_view', views.visit_view, name = "visit_view"),
    path('visit_list/<int:visit_id>/folders/', views.list_visit_folders, name='list_visit_folders'),
    path('visit_list/<int:visit_id>/<str:body_part>/images', views.list_visit_folder_images, name='list_visit_folder_images'),
    path('visit_list/get_image/<int:id>', views.get_image, name='get_image'),
    path('visit_list/update_visit_body_part/<int:id>/', views.update_visit_body_part, name = "update_visit_body_part"),
    path('visit_list/get_annotations/<int:id>/', views.get_annotations, name = "get_annotations"),
    
    path('doctor_list', views.doctor_list, name = 'doctor_list')
]