from django.shortcuts import render, redirect, HttpResponse
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

import os

# Create your views here.

def basic_encrypt(request):
    key = os.urandom(32)  # 256-bit key
    iv = os.urandom(16)   # 16-byte IV -> Add randomness to your encryption

    
    message = "Hello how are you doing, im fine"
    
    message_byte = message.encode()
    
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_message = padder.update(message_byte) + padder.finalize()
    
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_message = encryptor.update(padded_message) + encryptor.finalize()
    
    final_message = iv + encrypted_message
        
    print(message_byte)
    print(padded_message)
    print(encrypted_message)
    print(final_message)
    
    iv = final_message[:16]
    
    encrypted_text = final_message[16:]
    
    # Decrypt the message
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_padded_message = decryptor.update(encrypted_text) + decryptor.finalize()

    # Remove padding
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    decrypted_message = unpadder.update(decrypted_padded_message) + unpadder.finalize()
    
    print(decrypted_message)
    
    return HttpResponse("I'm doing encryption BEEP BOOP")
    
def dwt_encrypt(request):
    pass