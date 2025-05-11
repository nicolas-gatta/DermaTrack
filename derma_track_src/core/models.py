from django.db import models
from login.models import Doctor
from multiselectfield import MultiSelectField

class Status(models.TextChoices):
    SCHEDULED = 'scheduled', 'Scheduled'
    STARTED   = 'started',   'Started'
    FINISHED  = 'finished',  'Finished'
    CANCELED  = 'canceled',  'Canceled'
    
class BloodGroup(models.TextChoices):
    A_POSITIVE    = 'A+', 'A Positive'
    A_NEGATIVE    = 'A-', 'A Negative'
    B_POSITIVE    = 'B+', 'B Positive'
    B_NEGATIVE    = 'B-', 'B Negative'
    AB_POSITIVE   = 'AB+', 'AB Positive'
    AB_NEGATIVE   = 'AB-', 'AB Negative'
    O_POSITIVE    = 'O+', 'O Positive'
    O_NEGATIVE    = 'O-', 'O Negative'
    
class Allergy(models.TextChoices):
    PEANUTS     = 'peanuts', 'Peanuts'
    DAIRY       = 'dairy', 'Dairy'
    GLUTEN      = 'gluten', 'Gluten'
    SHELLFISH   = 'shellfish', 'Shellfish'
    SOY         = 'soy', 'Soy'
    EGGS        = 'eggs', 'Eggs'
    TREE_NUTS   = 'tree_nuts', 'Tree Nuts'
    WHEAT       = 'wheat', 'Wheat'
    FISH        = 'fish', 'Fish'
    SESAME      = 'sesame', 'Sesame'

# Create your models here.
class Patient(models.Model):
    
    national_number = models.CharField(max_length=11, unique=True)
    name = models.CharField(max_length=255) 
    surname = models.CharField(max_length=255)
    date_of_birth = models.DateField()
    blood_group = models.CharField(max_length=30, choices=BloodGroup.choices, default=BloodGroup.A_POSITIVE)
    allergies = MultiSelectField(choices=Allergy.choices, max_length=200, blank=True, default = None)
    street = models.CharField(max_length=255)
    number = models.IntegerField()
    city = models.CharField(max_length=255)
    zip_code = models.IntegerField()
    phone_number = models.CharField(max_length=255)
    other_phone_number = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return str(self.pk) + " - " + self.full_name()
    
    def full_name(self):
        return self.name + " " + self.surname
    
    def full_adress(self):
        return self.street + " " + str(self.number) + ", " + self.city + " (" + str(self.zip_code) + ")"
    
class Visit(models.Model):
    date = models.DateTimeField()
    note = models.TextField(blank = True)
    is_patient_present = models.BooleanField(blank = True, default=False)
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
    image_path = models.TextField(unique=True, null=False)
    distance_from_subject = models.FloatField(null=True)
    comment = models.TextField(blank=True, null=True)
    body_part = models.ForeignKey(BodyPart, verbose_name=("body part"), on_delete=models.CASCADE)
    visit = models.ForeignKey(Visit, verbose_name=("visit"), on_delete=models.CASCADE)
    
    def __str__(self):
        return str(self.pk) + " - " + "visit_"+str(self.visit.pk) + " - " + self.body_part.name
    