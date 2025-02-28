from django.db import models
from login.models import Doctor

class Status(models.TextChoices):
    SCHEDULED = 'scheduled', 'Scheduled'
    STARTED   = 'started',   'Started'
    FINISHED  = 'finished',  'Finished'
    CANCELED  = 'canceled',  'Canceled'

# Create your models here.
class Patient(models.Model):
    
    national_number = models.CharField(max_length=11, unique=True)
    name = models.CharField(max_length=255) 
    surname = models.CharField(max_length=255)
    date_of_birth = models.DateField()
    street = models.CharField(max_length=255)
    number = models.IntegerField()
    city = models.CharField(max_length=255)
    zip_code = models.IntegerField()
    phone_number = models.CharField(max_length=255)
    other_phone_number = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return str(self.national_number) + " - " + self.full_name()
    
    def full_name(self):
        return self.name + " " + self.surname
    
class Visit(models.Model):
    date = models.DateTimeField()
    note = models.TextField(blank = True)
    is_patient_present = models.BooleanField(blank = True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.SCHEDULED)
    patient = models.ForeignKey(Patient, verbose_name=("patient"), on_delete=models.CASCADE)
    doctor = models.ForeignKey(Doctor, verbose_name=("doctor"), on_delete=models.CASCADE)
    
    def __str__(self):
        return str(self.pk) + " - " + self.doctor.full_name() + " / " + self.patient.full_name() + " - "  + str(self.date)
    
class BodyPart(models.Model):
    name = models.CharField(max_length=255)
    
    def __str__(self):
        return str(self.pk) + " - " + self.name
    
class VisitBodyPart(models.Model):
    image_path = models.URLField()
    body_part = models.ForeignKey(BodyPart, verbose_name=("body part"), on_delete=models.CASCADE)
    visit = models.ForeignKey(Visit, verbose_name=("visit"), on_delete=models.CASCADE)
    comment = models.TextField(blank=True)
    
    def __str__(self):
        return str(self.pk) + " - " + self.visit.date + " - " + self.body_part.name
    