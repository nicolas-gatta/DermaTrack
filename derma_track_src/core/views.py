from django.shortcuts import render
from login.models import Doctor
from django.http import HttpResponse, HttpRequest, JsonResponse
from django.template.loader import render_to_string
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required

from core.models import Patient, Visit, Status, VisitBodyPart, BodyPart
from utils.checks import group_and_super_user_checks
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
def patient_profile(request):
    patient = Patient.objects.get(pk=request.POST["id"])
    return render(request, 'core/patient.html', {'patient': patient})

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
        
        visits = Visit.objects.select_related('doctor', 'patient').all()
        
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

    images = list(VisitBodyPart.objects.filter(visit_id = visit_id, body_part = BodyPart.objects.get(name = body_part).pk).values('pk','image_name','image_preview_name'))
    
    return JsonResponse({"images": images})

@login_required(login_url='/')
@group_and_super_user_checks(group_names=["Doctor"], redirect_url="/")
def get_image(request, id):
    if request.method == "GET" :
        image = VisitBodyPart.objects.get(pk = id)
        
        encoded_string = base64.b64encode(image.image_path.file.read()).decode('utf-8')
            
        return JsonResponse({"image": encoded_string, "distance": image.distance_from_subject, "pixel_size": image.pixel_size, "focal": image.focal})

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
                    print(value)
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

