from django.urls import path

from . import views

import core.views as cv
import super_resolution.views as srv

urlpatterns = [
    path("", views.index, name="index"),
    path("show_models", srv.show_models, name = "show_models"),
    path("model_form", srv.model_form, name = "model_form"),
    path("training_model", srv.training_model, name = "training_model"),
    path("patient", cv.patient_profile, name="patient"),
    path("patient_list", cv.patient_list, name="patient_list"),
    path('visit_list', cv.visit_list, name='visit_list'),
    path('doctor_list', cv.doctor_list, name='doctor_list')
]