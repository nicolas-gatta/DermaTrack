from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Doctor(models.Model):
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, blank=True)
    national_number = models.CharField(max_length=11, unique=True)
    INAMI = models.CharField(max_length=255, unique=True)
    name = models.CharField(max_length=255)
    surname = models.CharField(max_length=255)
    date_of_birth = models.DateField()
    street = models.CharField(max_length=255)
    number = models.IntegerField()
    city = models.CharField(max_length=255)
    zip_code = models.IntegerField()
    phone_number = models.CharField(max_length=255)
    other_phone_number = models.CharField(max_length=255, blank=True)
    is_retired = models.BooleanField()
    
    def __str__(self):
        return str(self.INAMI) + " - " + self.name + " " + self.surname
    
    def full_name(self):
        return self.name + " " + self.surname