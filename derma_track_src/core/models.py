from django.db import models
from login.models import Doctor
from multiselectfield import MultiSelectField

class Status(models.TextChoices):
    """
    Enumeration for the status of a visit.
    """
     
    SCHEDULED = 'scheduled', 'Scheduled'
    STARTED   = 'started',   'Started'
    FINISHED  = 'finished',  'Finished'
    CANCELED  = 'canceled',  'Canceled'
    
class BloodGroup(models.TextChoices):
    """
    Enumeration for the blood type.
    """
    A_POSITIVE    = 'A+', 'A Positive'
    A_NEGATIVE    = 'A-', 'A Negative'
    B_POSITIVE    = 'B+', 'B Positive'
    B_NEGATIVE    = 'B-', 'B Negative'
    AB_POSITIVE   = 'AB+', 'AB Positive'
    AB_NEGATIVE   = 'AB-', 'AB Negative'
    O_POSITIVE    = 'O+', 'O Positive'
    O_NEGATIVE    = 'O-', 'O Negative'
    
class Allergy(models.TextChoices):
    """
    Enumeration for the allergies.
    """
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
    """
    Model representing a patient in the system.

    Attributes:
        national_number (str): National ID number (unique).
        name (str): First name.
        surname (str): Last name.
        date_of_birth (date): Birth date.
        blood_group (str): Blood group from predefined choices.
        allergies (list): Multiple allergies from predefined choices.
        street (str): Address street name.
        number (int): Street number.
        city (str): City.
        zip_code (int): Postal code.
        phone_number (str): Primary contact number.
        other_phone_number (str): Secondary contact number (optional).
    """
    
    national_number = models.CharField(max_length=11, unique=True, blank=False)
    name = models.CharField(max_length=255, blank=False) 
    surname = models.CharField(max_length=255, blank=False)
    date_of_birth = models.DateField(blank=False)
    blood_group = models.CharField(max_length=30, choices=BloodGroup.choices, default=BloodGroup.A_POSITIVE, blank=False)
    allergies = MultiSelectField(choices=Allergy.choices, max_length=200, blank=True, default = None)
    street = models.CharField(max_length=255, blank=False)
    number = models.IntegerField(blank=False)
    city = models.CharField(max_length=255, blank=False)
    zip_code = models.IntegerField(blank=False)
    phone_number = models.CharField(max_length=255, blank=False)
    other_phone_number = models.CharField(max_length=255, blank=True)

    def __str__(self):
        """
        String representation of the doctor.

        Returns:
            str: Formatted string including id and full name.
        """
        return str(self.pk) + " - " + self.full_name()
    
    def full_name(self):
        """
        Returns the full name of the patient.

        Returns:
            str: Concatenation of name and surname.
        """
        return self.name + " " + self.surname
    
    def full_adress(self):
        """
        Returns the full adress of the patient.

        Returns:
            str: Concatenation of street, number, city and zip code.
        """
        return self.street + " " + str(self.number) + ", " + self.city + " (" + str(self.zip_code) + ")"
    
class Visit(models.Model):
    """
    Model representing a Visit in the system.

    Attributes:
        date (datetime): Date and time of the visit.
        note (str): Optional notes for the visit.
        is_patient_present (bool): Indicates if the patient attended.
        status (str): Visit status from Status choices.
        patient (Patient): Linked patient.
        doctor (Doctor): Linked doctor.
    """
    
    date = models.DateTimeField(blank = False)
    note = models.TextField(blank = True)
    is_patient_present = models.BooleanField(blank = True, default=False)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.SCHEDULED)
    patient = models.ForeignKey(Patient, verbose_name=("patient"), on_delete=models.CASCADE, blank = False)
    doctor = models.ForeignKey(Doctor, verbose_name=("doctor"), on_delete=models.CASCADE, blank = False)
    
    def __str__(self):
        """
        String representation of the doctor.

        Returns:
            str: Formatted string including id, doctor name, patient name and datae.
        """
        return str(self.pk) + " - " + self.doctor.full_name() + " / " + self.patient.full_name() + " - "  + str(self.date)
    
class BodyPart(models.Model):
    """
    Model representing a BodyPart in the system.

    Attributes:
        name (str): The name of the body part (e.g., 'Left Shoulder').
    """

    name = models.CharField(max_length=255)
    
    def __str__(self):
        """
        String representation of the body part.

        Returns:
            str: Formatted string including id and name.
        """
        return str(self.pk) + " - " + self.name
    
class VisitBodyPart(models.Model):
    """
    Model representing a VisitBodyPart in the system.

    Attributes:
        image_path (ImageField): Path to encrypted image.
        image_name (str): Filename of encrypted image.
        image_height (int): Original image height in pixels.
        image_width (int): Original image width in pixels.
        image_preview_path (ImageField): Path to preview image (PNG).
        image_preview_name (str): Filename of preview image.
        image_preview_height (int): Preview height in pixels.
        image_preview_width (int): Preview width in pixels.
        image_super_path (ImageField): Path to enhanced/super-resolved image.
        image_super_name (str): Filename of enhanced image.
        image_super_height (int): Height of enhanced image.
        image_super_width (int): Width of enhanced image.
        multi_image_path (str): Path to folder containing image patch set (if any).
        distance_from_subject (float): Distance from camera to subject.
        focal (float): Focal length used during capture.
        pixel_size (float): Physical pixel size (mm).
        annotations (JSON): Optional annotations (e.g., bounding boxes, comments).
        body_part (BodyPart): Related body part.
        visit (Visit): Related visit.
    """
    
    image_path = models.ImageField(null = True, blank = True)
    image_name = models.CharField(null = True, blank = True, max_length=255)
    image_height = models.IntegerField(default = 0)
    image_width = models.IntegerField(default = 0)
    image_preview_path = models.ImageField(null = True, blank = True)
    image_preview_name = models.CharField(null = True, blank = True, max_length=255)
    image_preview_height = models.IntegerField(default = 0)
    image_preview_width = models.IntegerField(default = 0)
    image_super_path = models.ImageField(null = True, blank = True)
    image_super_name = models.CharField(null = True, blank = True, max_length=255, default = "")
    image_super_height = models.IntegerField(default = 0)
    image_super_width = models.IntegerField(default = 0)
    multi_image_path = models.CharField(null = True, blank = True, max_length=255, default = "")
    distance_from_subject = models.FloatField(blank = True, null = True)
    focal = models.FloatField(default = 3.543)
    pixel_size = models.FloatField(default = 0.0014)
    annotations = models.JSONField(blank = True, null=True, default = None)
    body_part = models.ForeignKey(BodyPart, verbose_name=("body part"), on_delete=models.CASCADE)
    visit = models.ForeignKey(Visit, verbose_name=("visit"), on_delete=models.CASCADE)
    
    def __str__(self):
        """
        String representation of the VisitBodyPart.

        Returns:
            str: Formatted string including ID, visit ID and body part name.
        """
        return str(self.pk) + " - " + "visit_"+str(self.visit.pk) + " - " + self.body_part.name
    