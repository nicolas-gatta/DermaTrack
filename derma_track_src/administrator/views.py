from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from utils.checks import group_and_super_user_checks
from django.template.loader import render_to_string

# Create your views here.

@login_required
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def index(request):
    return render(request, 'administrator/index.html', {})

