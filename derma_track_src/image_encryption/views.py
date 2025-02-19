from django.shortcuts import render, redirect, HttpResponse
from dotenv import load_dotenv

from django.conf import settings
from .services.advanced_encrypted_standard import AES

import cv2
import os

# Create your views here.

def basic_decrypt(request):
    
    return HttpResponse("I'm doing decryption BEEP BOOP")

def basic_encrypt(request):
    
    input_path = os.path.join(settings.MEDIA_ROOT, "test", "woman.png")
    
    encrypted_path = os.path.join(settings.MEDIA_ROOT, "output_test", "encrypted_woman.enc")
    
    decrypted_path = os.path.join(settings.MEDIA_ROOT, "output_test", "decrypted_woman.png")
    
    with open(input_path, "rb") as f:
        image_data = f.read()
        encrypted_message = AES.encrypt_message(image_data)
        
        with open(encrypted_path, "wb") as f2:
            f2.write(encrypted_message)
    
    with open(encrypted_path, "rb") as f:
        image_data = f.read()
        decrypted_message = AES.decrypt_message(image_data)
        
        with open(decrypted_path, "wb") as f2:
            f2.write(decrypted_message)
            
    cv2.imshow("Decrypted Image", cv2.imread(decrypted_path))
    
    cv2.waitKey(0)
    # and finally destroy/close all open windows
    cv2.destroyAllWindows()
    
    return HttpResponse("I'm doing encryption BEEP BOOP")