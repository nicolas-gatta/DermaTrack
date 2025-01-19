from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from utils.checks import group_and_super_user_checks
from django.http import HttpResponse
from django.template.loader import render_to_string

# Create your views here.

@login_required
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def index(request):
    return render(request, 'administrator/index.html', {})

@login_required
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def show_models(request):
    return HttpResponse(render_to_string('partial/show_models.html', {}, request=request))

@login_required
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def model_form(request):
    return HttpResponse(render_to_string('partial/model_form.html', {}, request=request))

@login_required
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def training_model(request):
    pass