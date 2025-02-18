import os

from django.conf import settings
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

# https://cryptography.io/en/latest/hazmat/primitives/padding/
 
class AES():
    
    _key = settings.AES_SECRET_KEY
    
    def __init__(self):
        pass
    
    @staticmethod
    def encrypt_message(message):
        iv = os.urandom(16)   # 128-bits IV -> Add randomness to your encryption
    
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_message = padder.update(message) + padder.finalize()
        
        cipher = Cipher(algorithms.AES(AES._key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted_message = encryptor.update(padded_message) + encryptor.finalize()
        
        return iv + encrypted_message
        
    
    @staticmethod 
    def decrypt_message(message):
        iv = message[:16]
    
        encrypted_text = message[16:]
        
        # Decrypt the message
        cipher = Cipher(algorithms.AES(AES._key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted_padded_message = decryptor.update(encrypted_text) + decryptor.finalize()

        # Remove padding
        unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
        return unpadder.update(decrypted_padded_message) + unpadder.finalize()
    
    