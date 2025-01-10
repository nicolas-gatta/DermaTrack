from django.urls import path

from . import views

urlpatterns = [
    path("", views.patient_list, name="index"),
    path("patient", views.patient_profile, name="patient"),
]