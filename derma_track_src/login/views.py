from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login,logout
from django.contrib import messages
from django.http import HttpResponse

from .forms.login_form import LoginForm
    
    
# Create your views here.
def login_view(request):
    """
    Handles user login and redirects based on role (superuser or standard user).

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: A redirect to the appropriate dashboard or the login page.
    """

    if request.method == "POST":
        username = request.POST["login"]
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            if user.is_superuser:
                return redirect("/administrator")
            return redirect("/core")
        
        else:
            messages.success(request, "The password or username are invalid. Please try again")
            return redirect("/")
            
    elif request.user.is_authenticated:
        if request.user.is_superuser:
            return redirect("/administrator")
        return redirect("/core")
    else:   
        form = LoginForm()
        return render(request, "login/index.html", {"form": form })

def logout_view(request):
    """
    Logs out the current user and redirects to the login page.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: A redirect to the root login page.
    """
    
    logout(request)
    return redirect("/")
