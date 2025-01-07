from django.shortcuts import render
from django.http import HttpResponse
from PIL import Image
from super_resolution.modules.interpolations import Interpolation

import numpy as np
# Create your views here.


def index(request):
    # load image

    im = Image.open("super_resolution/test_images/cheval_test.png")

    width_2 = im.width // 4

    height_2 = im.height // 4


    # Go to normalized float and undo gamma

    # Note : sRGB gamma is not a pure power TF, but that will do

    im2 = (np.array(im) / 255.)**2.4


    # Interpolate in float64

    out = np.zeros((height_2, width_2, 3))

    out = Interpolation.bilinear_interpolation(im2, im.width, im.height, out, width_2, height_2)


    # Redo gamma and save back in uint8

    out = (out**(1/2.4) * 255.).astype(np.uint8)
    
    Image._show(Image.fromarray(out))
    
    return HttpResponse("Hello, world. You're at the polls index.")