from django.shortcuts import render

# Create your views here.

def administrator_menu(request):
    return render(request, 'administrator/index.html', {})