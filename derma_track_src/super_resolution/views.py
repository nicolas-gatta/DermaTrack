from django.shortcuts import render
from django.http import HttpResponse
from PIL import Image

import numpy as np
# Create your views here.


def index(request):
    
    return HttpResponse("Hello, world. You're at the polls index.")