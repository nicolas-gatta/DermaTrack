from django.shortcuts import render, redirect, HttpResponse
from dotenv import load_dotenv

from django.conf import settings

# Create your views here.

def basic_decrypt(request):
    pass

def basic_encrypt(request):
    
    load_dotenv()
    
    print(settings.AES_SECRET_KEY)
    
    return HttpResponse("I'm doing encryption BEEP BOOP")
    
def dwt_encrypt(request):
    pass

def dwt_decrypt(request):
    pass