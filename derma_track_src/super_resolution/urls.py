from django.urls import path

from . import views

urlpatterns = [
    path("training", views.training_srcnn, name="index"),
]