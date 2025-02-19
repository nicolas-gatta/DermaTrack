from django.urls import path

from . import views

urlpatterns = [
    path("basic_encrypt", views.basic_encrypt, name="basic_encrypt")
]