from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="patient_list"),
    path("patient", views.patient_profile, name="patient"),
    path("patient_list", views.patient_list, name="patient_list"),
    path('visit_list', views.visit_list, name='visit_list'),
    path('doctor_list', views.doctor_list, name='doctor_list')
]