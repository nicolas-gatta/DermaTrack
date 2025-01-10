from django.shortcuts import render
from .models import Patient

# Create your views here.

def patient_list(request):
    patients = Patient.objects.all() 
    return render(request, 'core/index.html', {'patients': patients})

def patient_profile(request):
    patient = Patient.objects.get(pk=request.POST["id"])
    return render(request, 'core/patient.html', {'patient': patient})