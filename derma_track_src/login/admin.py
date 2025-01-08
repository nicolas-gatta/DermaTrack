from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User

# Register your models here.
from .models import Doctor

class DoctorInLine(admin.StackedInline):
    model = Doctor
    can_delete = False
    verbose_name_plural = "doctor"


# Define a new User admin
class UserAdmin(BaseUserAdmin):
    inlines = [DoctorInLine]


# Re-register UserAdmin
admin.site.unregister(User)
admin.site.register(User, UserAdmin)
admin.site.register(Doctor)
