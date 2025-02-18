from django.urls import path

from . import views

urlpatterns = [
    path("basic_encrypt", views.basic_encrypt, name="basic_encrypt"),
    path("dwt_encrypt", views.dwt_encrypt, name="dwt_encrypt")
]