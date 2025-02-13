from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    
    path("patient_profile", views.patient_profile, name = "patient_profile"),
    path("patient_list", views.patient_list, name = "patient_list"),
    
    path('visit_list', views.visit_list, name = 'visit_list'),
    path('visit_status', views.visit_status_change, name = 'visit_status'),
    path('visit_view', views.visit_view, name = "visit_view"),
    
    path('doctor_list', views.doctor_list, name = 'doctor_list')
]