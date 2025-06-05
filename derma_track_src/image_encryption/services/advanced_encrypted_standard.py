import os

from django.conf import settings
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

# https://cryptography.io/en/latest/hazmat/primitives/padding/
 
class AES():
    """
    Utility class for AES encryption and decryption using CBC mode and PKCS7 padding.

    This class uses a secret key defined in Django settings as `AES_SECRET_KEY`.
    """
    
    _key = settings.AES_SECRET_KEY
    
    def __init__(self):
        pass
    
    @staticmethod
    def encrypt_message(data: bytes) -> bytes:
        """
        Encrypts a byte string using AES encryption in CBC mode with PKCS7 padding and a random 16-byte IV.

        Args:
            data (bytes): The plaintext data to encrypt.

        Returns:
            bytes: The IV followed by the encrypted ciphertext.
        """
        
        iv = os.urandom(16)
    
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(data) + padder.finalize()
        
        cipher = Cipher(algorithms.AES(AES._key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        return iv + encrypted_data
        
    
    @staticmethod 
    def decrypt_message(data: bytes) -> bytes:
        """
        Decrypts AES-encrypted data that was encrypted using `encrypt_message`.

        Args:
            data (bytes): The encrypted data (IV + ciphertext).

        Returns:
            bytes: The decrypted plaintext.
        """
        
        iv = data[:16]
    
        encrypted_text = data[16:]
        
        # Decrypt the data
        cipher = Cipher(algorithms.AES(AES._key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted_padded_data = decryptor.update(encrypted_text) + decryptor.finalize()

        # Remove padding
        unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
        return unpadder.update(decrypted_padded_data) + unpadder.finalize()
    
    