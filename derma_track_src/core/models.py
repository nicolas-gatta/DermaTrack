from django.db import models
from login.models import Doctor
from django.contrib.auth.models import User

# Create your models here.
# Create your models here.
class Patient(models.Model):
    
    name = models.CharField(max_length=255)
    surname = models.CharField(max_length=255)
    street = models.CharField(max_length=255)
    number = models.IntegerField()
    city = models.CharField(max_length=255)
    zip_code = models.IntegerField()
    phone_number = models.CharField(max_length=255)
    other_phone_number = models.CharField(max_length=255, blank=True, default='')

    def __str__(self):
        return str(self.pk) +" - "+ self.name + " " + self.surname
    
class Visit(models.Model):
    date = models.DateField()
    note = models.TextField()
    is_patient_present = models.BooleanField()
    patient = models.ForeignKey(Patient, verbose_name=("patient"), on_delete=models.CASCADE)
    doctor = models.ForeignKey(Doctor, verbose_name=("doctor"), on_delete=models.CASCADE)
    
class BodyPart(models.Model):
    name = models.CharField(max_length=255)
    
class VisitBodyPart(models.Model):
    body_part = models.ForeignKey(BodyPart, verbose_name=("body part"), on_delete=models.CASCADE)
    visit = models.ForeignKey(Visit, verbose_name=("visit"), on_delete=models.CASCADE)