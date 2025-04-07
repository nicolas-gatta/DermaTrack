from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login,logout
from django.contrib import messages
from django.http import HttpResponse

from .forms.login_form import LoginForm
    
    
# Create your views here.
def login_view(request):
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
    logout(request)
    return redirect("/")
