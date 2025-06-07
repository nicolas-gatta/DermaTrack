from django.shortcuts import render
from login.models import Doctor
from django.http import JsonResponse
from django.db.models.functions import Concat
from django.db.models import Value
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from .forms import PatientForm, VisitForm, VisitFormAdmin
from core.models import Patient, Visit, Status, VisitBodyPart, BodyPart, Doctor
from utils.checks import group_and_super_user_checks
from image_encryption.services.advanced_encrypted_standard import AES

import os
import base64
import json


@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def index(request):
    """
    Rendering the index page.

    This view is protected by login and group/superuser access checks.
    It renders the 'core/index.html' template with an empty context.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered HTML response for the index.
    """
    
    return render(request, 'core/index.html')

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def patient_list(request):
    """
    Rendering the patient_list page.

    This view is protected by login and group/superuser access checks.
    It renders the 'partial/patient_list.html' template with patients list in the context.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered HTML response for the patient_list.
    """
    
    if request.headers.get('HX-Request'):
        patients = Patient.objects.all() 
        return render(request, 'partial/patient_list.html', {'patients': patients})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def doctor_list(request):
    """
    Rendering the doctor_list page.

    This view is protected by login and group/superuser access checks.
    It renders the 'partial/doctor_list.html' template with doctors list in the context.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered HTML response for the doctor_list.
    """
    
    if request.headers.get('HX-Request'):
        doctors = Doctor.objects.all()
        return render(request, 'partial/doctor_list.html', {'doctors': doctors})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def patient_profile(request, id: int):
    """
    Rendering the patient_form page to show the patient profile.

    This view is protected by login and group/superuser access checks.
    It renders the 'partial/patient_form.html' template with for in the context.

    Args:
        request (HttpRequest): The HTTP request object.
        id (int): The id of the patient

    Returns:
        HttpResponse: The rendered HTML response for the patient_form.
    """
    
    if request.method == "GET":
        patient = Patient.objects.get(pk = id)

        form = PatientForm(instance=patient)

        for field in form.fields.values():
            field.disabled = True

        return render(request, "partial/patient_form.html", {"form": form, "is_form": False})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def visit_status_change(request):
    """
    Updates the status of a visit (Started, Finished, or Canceled),
    and refreshes the visit list.

    Args:
        request (HttpRequest): POST request with 'id' and 'status'.

    Returns:
        HttpResponse: Rendered visit list.
    """
    
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
    """
    Rendering the image_capture page.

    This view is protected by login and group/superuser access checks.
    It renders the 'partial/image_capture.html' template with an empty context.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered HTML response for the image_capture.
    """
    
    if request.headers.get('HX-Request'):
        return render(request, 'partial/image_capture.html')

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def visit_list(request):
    """
    Rendering the visit_list page.

    This view is protected by login and group/superuser access checks.
    It renders the 'partial/visit_list.html' template with visits in context.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered HTML response for the visit_list.
    """
    
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
    """
    Rendering the visit_view page.

    This view is protected by login and group/superuser access checks.
    It renders the 'partial/visit_view.html' template with visit in the context

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered HTML response for the visit_view.
    """
    
    visit = Visit.objects.get(pk = request.POST['id'])
    return render(request, 'partial/visit_view.html', {'visit': visit})


@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def create_patient(request):
    """
    Create a patient using the information in the forms.

    This view is protected by login and group/superuser access checks.
    It renders the 'partial/patient_form.html' template with form in the context

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered HTML response for the patient_form or patient_list when the patient is save.
    """
    
    if request.method == "POST":
        form = PatientForm(request.POST)
        if form.is_valid():
            form.save()
            patients = Patient.objects.all() 
            return render(request, 'partial/patient_list.html', {'patients': patients})
        return render(request, "partial/patient_form.html", {"form": form, "is_form": True})
    else:
        form = PatientForm()
        return render(request, "partial/patient_form.html", {"form": form, "is_form": True})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def create_visit(request):
    """
    Create a visit using the information in the forms.

    This view is protected by login and group/superuser access checks.
    It renders the 'partial/visit_form.html' template with form in the context

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered HTML response for the visit_form or visit_list when the patient is save.
    """
    
    if request.method == "POST":
        form = VisitFormAdmin(request.POST) if request.user.is_superuser else VisitForm(request.POST)
        if form.is_valid():
            if request.user.is_superuser:
                form.save()
            else:
                visit = form.save(commit=False)
                visit.doctor = Doctor.objects.get(user = request.user)
                visit.status = Status.SCHEDULED
                visit.save()
            if request.user.is_superuser:
                visits = Visit.objects.select_related('doctor', 'patient').all()
            elif request.user.groups.filter(name__in=["Doctor"]).exists():
                visits = Visit.objects.select_related('doctor', 'patient').filter(doctor__user=request.user)
            
            return render(request, 'partial/visit_list.html', {'visits': visits})
        return render(request, "partial/visit_form.html", {"form": form})
    else:
        form = VisitFormAdmin() if request.user.is_superuser else VisitForm()
        return render(request, "partial/visit_form.html", {"form": form})
    
@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def get_visit_by_patient_name(request):
    """
    Rendering the filter visit_view page.

    This view is protected by login and group/superuser access checks.
    It renders the 'partial/visit_view.html' template with visit in the context

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered HTML response for the visit_view.
    """
    
    if request.method == "GET":
        
        patient_name = request.GET['name']
        
        visits = Visit.objects.select_related('doctor', 'patient').all()
        
        visits = visits.annotate(full_name = Concat('patient__name', Value(' '), 'patient__surname'))
        
        visits = visits.filter(full_name__icontains = patient_name)
        
        if not request.user.is_superuser and request.user.groups.filter(name__in=["Doctor"]).exists():
            visits = Visit.objects.filter(doctor__user=request.user)

        return render(request, 'partial/visit_list.html', {'visits': visits})
    
@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def get_patient_by_name(request):
    """
    Rendering the filter patient_list page.

    This view is protected by login and group/superuser access checks.
    It renders the 'partial/patient_list.html' template with patients in the context

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered HTML response for the patient_list.
    """
    
    if request.method == "GET":
        
        patient_name = request.GET['name']
        
        patients = Patient.objects.all()
        
        patients = patients.annotate(full_name = Concat('name', Value(' '), 'surname'))
        
        patients = patients.filter(full_name__icontains = patient_name)

        return render(request, 'partial/patient_list.html', {'patients': patients})
    
@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def get_doctor_by_name(request):
    """
    Rendering the filter doctor_list page.

    This view is protected by login and group/superuser access checks.
    It renders the 'partial/doctor_list.html' template with doctors in the context

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered HTML response for the doctor_list.
    """
    
    if request.method == "GET":
        
        doctor_name = request.GET['name']
        
        doctors = Doctor.objects.all()
        
        doctors = doctors.annotate(full_name = Concat('name', Value(' '), 'surname'))
        
        doctors = doctors.filter(full_name__icontains = doctor_name)

        return render(request, 'partial/doctor_list.html', {'doctors': doctors})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def list_visit_folders(request, visit_id):    
    """
    Returns the list of distinct body part folders associated with a given visit.

    Args:
        request (HttpRequest): The incoming HTTP request.
        visit_id (int): The ID of the visit to retrieve folders for.

    Returns:
        JsonResponse: A list of distinct body part names under the 'folders' key.
    """
    
    folder_names = VisitBodyPart.objects.filter(visit_id=visit_id).values_list('body_part__name', flat=True).distinct()

    return JsonResponse({"folders": list(folder_names)})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def list_visit_folder_images(request, visit_id, body_part):
    """
    Returns the list of images associated with a given visit.

    Args:
        request (HttpRequest): The incoming HTTP request.
        visit_id (int): The ID of the visit to retrieve folders for.
        body_part (str): The name of the body part.

    Returns:
        JsonResponse: A list of images under the 'images' key.
    """
    
    try:
        visit_body_part = VisitBodyPart.objects.filter(visit_id = visit_id, body_part = BodyPart.objects.get(name = body_part).pk)
        
        images = list(visit_body_part.values('pk','image_name','image_preview_name'))    
        
        return JsonResponse({"images": images})
    
    except VisitBodyPart.DoesNotExist:
        return JsonResponse({"images": {}})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def get_image(request, id):
    """
    Retrieves and decrypts the original image for a VisitBodyPart ID.

    Args:
        request (HttpRequest): GET request.
        id (int): The id of the visitBodyPart.

    Returns:
        JsonResponse: Base64-encoded decrypted image, distance, pixel_size, focal and has_super.
    """
    
    if request.method == "GET" :
        visit_body_part = VisitBodyPart.objects.get(pk = id)
        
        encoded_string = base64.b64encode(AES.decrypt_message(visit_body_part.image_path.file.read())).decode('utf-8')
        
        has_super = visit_body_part.image_super_name != None and len(visit_body_part.image_super_name) != 0
        
        return JsonResponse({"image": encoded_string, "distance": visit_body_part.distance_from_subject, "pixel_size": visit_body_part.pixel_size, "focal": visit_body_part.focal, 
                             "has_super": has_super})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def get_enchanced_image(request, id):
    """
    Retrieves and decrypts the enchanced image for a VisitBodyPart ID.

    Args:
        request (HttpRequest): GET request.
        id (int): The id of the visitBodyPart.

    Returns:
        JsonResponse: Base64-encoded decrypted enchanced image, distance, pixel_size, focal and has_super..
    """
    
    if request.method == "GET" :
        visit_body_part = VisitBodyPart.objects.get(pk = id)
        
        encoded_string = base64.b64encode(AES.decrypt_message(visit_body_part.image_super_path.file.read())).decode('utf-8')
        
        has_super = len(visit_body_part.image_super_name) != 0
        
        return JsonResponse({"image": encoded_string, "distance": visit_body_part.distance_from_subject, "pixel_size": visit_body_part.pixel_size, "focal": visit_body_part.focal, 
                             "has_super": has_super})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def get_annotations(request, id):
    """
    Retrieves the annotation for a VisitBodyPart ID.

    Args:
        request (HttpRequest): GET request.
        id (int): The id of the visitBodyPart.

    Returns:
        JsonResponse: annotations into Json format.
    """
    
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
    """
    Updates fields of a VisitBodyPart instance via PUT request.

    Args:
        request (HttpRequest): PUT request with JSON payload.
        id (int): The ID of the VisitBodyPart to update.

    Returns:
        JsonResponse: Success or error message.
    """
    
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
    """
    Deletes all associated files (encrypted, preview, enhanced) of a VisitBodyPart,
    and removes the database entry.

    Args:
        request (HttpRequest): DELETE request.
        id (int): THe ID of the VisitBodyPart to delete.

    Returns:
        JsonResponse: Deletion confirmation or error.
    """
    
    if request.method == "DELETE":
        try:
            
            visit_body_part = VisitBodyPart.objects.get(pk = id)
            
            body_part_name = visit_body_part.body_part.name
            
            visit_id = visit_body_part.visit.pk
            
            image_path = visit_body_part.image_path.path if visit_body_part.image_path else None
            
            preview_path = visit_body_part.image_preview_path.path if visit_body_part.image_preview_path else None
            
            multi_path = visit_body_part.multi_image_path if visit_body_part.multi_image_path else None
            
            super_path = visit_body_part.image_super_path.path if visit_body_part.image_super_path else None

            for internal_path in [image_path, preview_path, multi_path, super_path]:
                
                if internal_path != None and os.path.exists(internal_path):
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
