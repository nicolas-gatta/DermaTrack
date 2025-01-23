from django.shortcuts import render
from .models import Patient, Visit
from login.models import Doctor
from django.http import HttpResponse
from django.template.loader import render_to_string
from django.contrib.auth.decorators import login_required
from utils.checks import group_and_super_user_checks


# Create your views here.

@login_required
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def index(request):
    return render(request, 'core/index.html')

@login_required
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def patient_list(request):
    if request.headers.get('HX-Request'):
        patients = Patient.objects.all() 
        return render(request, 'partial/patient_list.html', {'patients': patients})

@login_required
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def visit_list(request):
    if request.headers.get('HX-Request'):
        visits = Visit.objects.select_related('doctor', 'patient').all()
        return render(request, 'partial/visit_list.html', {'visits': visits})

@login_required
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def doctor_list(request):
    if request.headers.get('HX-Request'):
        doctors = Doctor.objects.all()
        return render(request, 'partial/doctor_list.html', {'doctors': doctors})

@login_required
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def patient_profile(request):
    patient = Patient.objects.get(pk=request.POST["id"])
    return render(request, 'core/patient.html', {'patient': patient})
