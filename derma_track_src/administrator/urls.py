from django.urls import path

from . import views

import core.views as cv

urlpatterns = [
    path("", views.administrator_menu, name="index"),
    path("patient", cv.patient_profile, name="patient"),
    path("patient_list", cv.patient_list, name="patient_list"),
    path('visit_list', cv.visit_list, name='visit_list'),
    path('doctor_list', cv.doctor_list, name='doctor_list')
]