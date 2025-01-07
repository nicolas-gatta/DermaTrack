from django.db import models
from connection.models import Doctor

# Create your models here.
# Create your models here.
class Patient(models.Model):

    user = models.OneToOneField(User, on_delete=models.CASCADE, blank=True)
    INAMI = models.CharField(primary_key=True, max_length=255)
    name = models.CharField(max_length=255)
    surname = models.CharField(max_length=255)
    street = models.CharField(max_length=255)
    number = models.IntegerField()
    city = models.CharField(max_length=255)
    zip_code = models.IntegerField()
    phone_number = models.CharField(max_length=255)
    other_phone_number = models.CharField(max_length=255, blank=True)
    is_retired = models.BooleanField()
    
    def __str__(self):
        return str(self.INAMI) +" - "+ self.name + " " + self.surname
    
class Visit(models.Model):
    date = models.DateField()
    note = models.TextField()
    is_patient_present = models.BooleanField()
    patient = models.ForeignKey(Patient, verbose_name=_("patient"), on_delete=models.CASCADE)
    doctor = models.ForeignKey(Doctor, verbose_name=_("doctor"), on_delete=models.CASCADE)
    
class BodyPart():
    name = models.CharField(max_length=255)
    
class VisitBodyPart():
    body_part = models.ForeignKey(BodyPart, verbose_name=_("body part"), on_delete=models.CASCADE)
    visit = models.ForeignKey(Visit, verbose_name=_("visit"), on_delete=models.CASCADE)