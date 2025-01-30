from django.urls import path

from . import views

import core.views as cv
import super_resolution.views as srv

urlpatterns = [
    path("", views.index, name="index")
]