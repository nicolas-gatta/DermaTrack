from django.shortcuts import render
from .models import Patient, Visit
from login.models import Doctor
from django.http import HttpResponse
from django.template.loader import render_to_string
from django.utils.safestring import SafeString
from django.contrib.auth.decorators import login_required
from utils.checks import group_checks


# Create your views here.
@group_checks(group_names=[""], redirect_url="/")
@login_required
def index(request):
    return render(request, 'core/index.html')

@group_checks(group_names=[""], redirect_url="/")
@login_required
def patient_list(request):
    patients = Patient.objects.all() 
    return HttpResponse(render_to_string('partial/patient_list.html', {'patients': patients}, request = request))

@group_checks(group_names=[""], redirect_url="/")
@login_required
def visit_list(request):
    visits = Visit.objects.select_related('doctor', 'patient').all()
    return HttpResponse(render_to_string('partial/visit_list.html', {'visits': visits}, request=request))

@group_checks(group_names=[""], redirect_url="/")
@login_required
def doctor_list(request):
    doctors = Doctor.objects.all()
    return HttpResponse(render_to_string('partial/doctor_list.html', {'doctors': doctors}, request = request))

@group_checks(group_names=[""], redirect_url="/")
@login_required
def patient_profile(request):
    patient = Patient.objects.get(pk=request.POST["id"])
    return render(request, 'core/patient.html', {'patient': patient})
