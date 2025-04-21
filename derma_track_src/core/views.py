from django.shortcuts import render
from login.models import Doctor
from django.http import HttpResponse, HttpRequest, JsonResponse
from django.template.loader import render_to_string
from django.contrib.auth.decorators import login_required
from django.conf import settings

from core.models import Patient, Visit, Status
from utils.checks import group_and_super_user_checks

import os


# Create your views here.

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def index(request):
    return render(request, 'core/index.html')

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def patient_list(request):
    if request.headers.get('HX-Request'):
        patients = Patient.objects.all() 
        return render(request, 'partial/patient_list.html', {'patients': patients})
    
@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def image_capture(request):
    if request.headers.get('HX-Request'):
        return render(request, 'partial/image_capture.html')

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def visit_list(request):
    if request.headers.get('HX-Request'):
        visits = None
        if request.user.is_superuser:
            visits = Visit.objects.select_related('doctor', 'patient').all()
        elif request.user.groups.filter(name__in=["Doctor"]).exists():
            visits = Visit.objects.select_related('doctor', 'patient').filter(doctor__user=request.user)

        return render(request, 'partial/visit_list.html', {'visits': visits})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def visit_status_change(request):
    if request.method == "POST":
        
        status = request.POST['status']

        visit = Visit.objects.get(pk = request.POST['id'])
        
        if status == "Started":
            visit.status = Status.STARTED
        
        elif status == "Finished":
            visit.status = Status.FINISHED
            
        else:
            visit.status = Status.CANCELED
            
        visit.save(update_fields = ['status'])
        
        visits = Visit.objects.select_related('doctor', 'patient').all()
        
        return render(request, 'partial/visit_list.html', {'visits': visits})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def visit_view(request):
    visit = Visit.objects.get(pk = request.POST['id'])
    return render(request, 'partial/visit_view.html', {'visit': visit})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def doctor_list(request):
    if request.headers.get('HX-Request'):
        doctors = Doctor.objects.all()
        return render(request, 'partial/doctor_list.html', {'doctors': doctors})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def patient_profile(request):
    patient = Patient.objects.get(pk=request.POST["id"])
    return render(request, 'core/patient.html', {'patient': patient})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def list_visit_folders(request, visit_id):
    base_path = os.path.join(settings.MEDIA_ROOT, "visits", f"visit_{visit_id}")
    folder_names = []

    if os.path.exists(base_path):
        for name in os.listdir(base_path):
            full_path = os.path.join(base_path, name)
            if os.path.isdir(full_path):
                folder_names.append(name)
    
    return JsonResponse({"folders": folder_names})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def list_visit_folder_images(request, visit_id, body_part):
    base_path = os.path.join(settings.MEDIA_ROOT, "visits", f"visit_{visit_id}", f"{body_part}")
    image_names = []

    if os.path.exists(base_path):
        for name in os.listdir(base_path):
            full_path = os.path.join(base_path, name)
            if not os.path.isdir(full_path):
                image_names.append(name)
    
    return JsonResponse({"images": image_names})


