from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Doctor(models.Model):
    """
    Model representing a doctor in the system.

    Attributes:
        user (User): One-to-one link to Django's built-in User model.
        national_number (str): Unique national identification number (11 digits).
        INAMI (str): Unique INAMI registration number.
        name (str): Doctor's first name.
        surname (str): Doctor's last name.
        date_of_birth (date): Date of birth.
        street (str): Street name of the address.
        number (int): Street number.
        city (str): City of residence.
        zip_code (int): Postal code.
        phone_number (str): Primary contact number.
        other_phone_number (str): Optional secondary contact number.
        is_retired (bool): Whether the doctor is retired.
    """
    
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
        """
        String representation of the doctor.

        Returns:
            str: Formatted string including name and INAMI number.
        """
        
        return str(self.pk) + " - " + self.name + " " + self.surname + " (" + str(self.INAMI) + ")"
    
    def full_name(self):
        """
        Returns the full name of the doctor.

        Returns:
            str: Concatenation of name and surname.
        """
        
        return self.name + " " + self.surname