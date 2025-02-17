from django.urls import path

from . import views

urlpatterns = [
    path("encrypt", views.encrypt, name="encrypt"),
    path("encrypt_dwt", views.encrypt_dwt, name="encrypt_dwt")
]