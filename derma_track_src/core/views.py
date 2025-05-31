from django.shortcuts import render
from login.models import Doctor
from django.http import JsonResponse
from django.template.loader import render_to_string
from django.views.decorators.csrf import csrf_exempt
import os
from django.contrib.auth.decorators import login_required
from .forms import PatientForm, VisitForm, VisitFormAdmin

from core.models import Patient, Visit, Status, VisitBodyPart, BodyPart, Doctor
from utils.checks import group_and_super_user_checks
from image_encryption.services.advanced_encrypted_standard import AES
import base64
import json


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
def doctor_list(request):
    if request.headers.get('HX-Request'):
        doctors = Doctor.objects.all()
        return render(request, 'partial/doctor_list.html', {'doctors': doctors})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def patient_profile(request, id):
    if request.method == "GET":
        patient = Patient.objects.get(pk = id)

        form = PatientForm(instance=patient)

        for field in form.fields.values():
            field.disabled = True

        return render(request, "partial/patient_form.html", {"form": form, "is_form": False})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def visit_status_change(request):
    if request.method == "POST":
        
        status = request.POST['status']

        visit = Visit.objects.get(pk = request.POST['id'])
        
        if status == "Started":
            visit.is_patient_present = True
            visit.status = Status.STARTED
        
        elif status == "Finished":
            visit.is_patient_present = True
            visit.status = Status.FINISHED
            
        else:
            visit.is_patient_present = False
            visit.status = Status.CANCELED
            
        visit.save(update_fields = ['status', 'is_patient_present'])
        
        visits = None
        if request.user.is_superuser:
            visits = Visit.objects.select_related('doctor', 'patient').all()
        elif request.user.groups.filter(name__in=["Doctor"]).exists():
            visits = Visit.objects.select_related('doctor', 'patient').filter(doctor__user=request.user)
        
        return render(request, 'partial/visit_list.html', {'visits': visits})

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
def visit_view(request):
    visit = Visit.objects.get(pk = request.POST['id'])
    return render(request, 'partial/visit_view.html', {'visit': visit})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def list_visit_folders(request, visit_id):    

    folder_names = VisitBodyPart.objects.filter(visit_id=visit_id).values_list('body_part__name', flat=True).distinct()

    return JsonResponse({"folders": list(folder_names)})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def list_visit_folder_images(request, visit_id, body_part):

    try:
        visit_body_part = VisitBodyPart.objects.filter(visit_id = visit_id, body_part = BodyPart.objects.get(name = body_part).pk)
        
        images = list(visit_body_part.values('pk','image_name','image_preview_name'))    
        
        return JsonResponse({"images": images})
    
    except VisitBodyPart.DoesNotExist:
        return JsonResponse({"images": {}})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def get_image(request, id):
    if request.method == "GET" :
        visit_body_part = VisitBodyPart.objects.get(pk = id)
        
        encoded_string = base64.b64encode(AES.decrypt_message(visit_body_part.image_path.file.read())).decode('utf-8')
        
        has_super = len(visit_body_part.image_super_name) != 0
        
        return JsonResponse({"image": encoded_string, "distance": visit_body_part.distance_from_subject, "pixel_size": visit_body_part.pixel_size, "focal": visit_body_part.focal, 
                             "has_super": has_super})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def get_enchanced_image(request, id):
    if request.method == "GET" :
        visit_body_part = VisitBodyPart.objects.get(pk = id)
        
        encoded_string = base64.b64encode(AES.decrypt_message(visit_body_part.image_super_path.file.read())).decode('utf-8')
        
        has_super = len(visit_body_part.image_super_name) != 0
        
        return JsonResponse({"image": encoded_string, "distance": visit_body_part.distance_from_subject, "pixel_size": visit_body_part.pixel_size, "focal": visit_body_part.focal, 
                             "has_super": has_super})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def get_annotations(request, id):
    if request.method == "GET" :
        try:
            annotations = VisitBodyPart.objects.get(pk = id).annotations
            
            return JsonResponse({"annotations": annotations}, status=200)

        except VisitBodyPart.DoesNotExist:
            return JsonResponse({
                "status": "error",
                "message": f"No VisitBodyPart found with ID {id}"
            }, status=404)

@csrf_exempt
@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def update_visit_body_part(request, id):
    if request.method == "PUT" :
        try:
            visit_body_part = VisitBodyPart.objects.get(pk=id)

            data = json.loads(request.body)
            
            for field, value in data.items():
                if hasattr(visit_body_part, field):
                    setattr(visit_body_part, field, value)

            visit_body_part.save()

            return JsonResponse({
                "status": "success",
                "message": f"Updated sucessfull on VisitBodyPart with ID {id} ."
            }, status=200)

        except VisitBodyPart.DoesNotExist:
            return JsonResponse({
                "status": "error",
                "message": f"No VisitBodyPart found with ID {id}"
            }, status=404)

@csrf_exempt
@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def delete_image(request, id):
    if request.method == "DELETE":
        try:
            visit_body_part = VisitBodyPart.objects.get(pk = id)
            
            body_part_name = visit_body_part.body_part.name
            
            visit_id = visit_body_part.visit.pk
            
            image_path = visit_body_part.image_path.path 
            
            preview_path = visit_body_part.image_preview_path.path
            
            multi_path = visit_body_part.multi_image_path
            
            super_path = visit_body_part.image_super_path.path

            for internal_path in [image_path, preview_path, multi_path, super_path]:
                
                if os.path.exists(internal_path):
                    if os.path.isfile(internal_path):
                        os.remove(internal_path)
                    else:
                        for filename in os.listdir(internal_path):
                            os.remove(os.path.join(internal_path, filename))
                        os.rmdir(internal_path)
                    
            visit_body_part.delete()
            
            return JsonResponse({
                "status": "success",
                "body_part": body_part_name,
                "visit_id": visit_id
            })

        except VisitBodyPart.DoesNotExist:
            return JsonResponse({"status": "error", "message": "VisitBodyPart not found"}, status=404)
            
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def create_patient(request):
    if request.method == "POST":
        form = PatientForm(request.POST)
        if form.is_valid():
            form.save()
            return JsonResponse({"success": True})
        return render(request, "partial/patient_form.html", {"form": form, "is_form": True})
    else:
        form = PatientForm()
        return render(request, "partial/patient_form.html", {"form": form, "is_form": True})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def create_visit(request):
    if request.method == "POST":
        form = VisitForm(request.POST)
        if form.is_valid():
            if request.user.is_superuser:
                form.save()
            else:
                visit = form.save(commit=False)
                visit.doctor = Doctor.objects.get(user = request.user)
                visit.status = Status.SCHEDULED
                visit.save()
            return JsonResponse({"success": True})
        return render(request, "partial/visit_form.html", {"form": form})
    else:
        form =  VisitFormAdmin() if request.user.is_superuser else VisitForm()
        return render(request, "partial/visit_form.html", {"form": form})