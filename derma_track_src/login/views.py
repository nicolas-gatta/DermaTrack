from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django import forms
from django.contrib.auth import logout


class Connection(forms.Form):
    username = forms.CharField(label ="Username", max_length=50)
    password = forms.CharField(label="Password", max_length=20, widget=forms.PasswordInput())
    
    
# Create your views here.
def login_view(request):
    if request.method == "POST":
        username = request.POST["login"]
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return HttpResponse("Login sucessful")
        
        else:
            messages.success(request, "The password or username are invalid. Please try again")
            return redirect("/")
            
    elif request.user.is_authenticated:
        return HttpResponse("You are already login")
    else:   
        return render(request, "login/index.html", {})

def logout_view(request):
    logout(request)
