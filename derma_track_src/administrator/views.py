from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from utils.checks import group_and_super_user_checks

# Create your views here.

@login_required(login_url='/')
@group_and_super_user_checks(group_names=[""], redirect_url="/")
def index(request):
    """
    Rendering the administrator index page.
    
    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered HTML response.
    """
    return render(request, 'administrator/index.html')

