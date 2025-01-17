from django.shortcuts import render
from .models import Patient, Visit
from login.models import Doctor
from django.http import HttpResponse
from django.template.loader import render_to_string
from django.utils.safestring import SafeString


# Create your views here.

def index(request):
    return render(request, 'core/index.html')

def patient_list(request):
    patients = Patient.objects.all() 
    return HttpResponse(render_to_string('partial/patient_list.html', {'patients': patients}, request = request))

def visit_list(request):
    visits = Visit.objects.all()
    return render_to_string('partial/visit_list.html', {'visits': visits}, request = request)
    
def doctor_list(request):
    doctors = Doctor.objects.all()
    return render_to_string('partial/doctor_list.html', {'doctors': doctors}, request = request)

def patient_profile(request):
    patient = Patient.objects.get(pk=request.POST["id"])
    return render(request, 'core/patient.html', {'patient': patient})
