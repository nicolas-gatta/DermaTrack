from django.contrib import admin

from .models import Visit, Patient, BodyPart, VisitBodyPart

# Register your models here.

admin.site.register(Visit)
admin.site.register(Patient)
admin.site.register(BodyPart)
admin.site.register(VisitBodyPart)
