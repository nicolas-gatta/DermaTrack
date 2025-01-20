from django.urls import path

from . import views

urlpatterns = [
    path("training", views.training_srcnn, name="index"),
    path("show_models", views.show_models, name = "show_models"),
    path("model_form", views.model_form, name = "model_form"),
    path("training_model", views.training_model, name = "training_model"),
    path("test", views.test, name = "test")
]