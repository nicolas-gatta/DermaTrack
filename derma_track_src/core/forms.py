from django import forms
from .models import Patient, Visit
from django.utils.timezone import now


class PatientForm(forms.ModelForm):
    class Meta:
        model = Patient
        fields = ["national_number", "name", "surname", "date_of_birth", "blood_group", "allergies", "street", 
                  "number", "city", "zip_code", "phone_number", "other_phone_number"]
        widgets = {
            'date_of_birth': forms.DateInput(attrs={'type': 'date'}),
            'allergies': forms.CheckboxSelectMultiple(),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.fields["national_number"].widget.attrs.update({"class":"form-control"})
        self.fields["name"].widget.attrs.update({"class":"form-control"})
        self.fields["surname"].widget.attrs.update({"class":"form-control"})
        self.fields["date_of_birth"].widget.attrs.update({"class":"form-control"})
        self.fields["blood_group"].widget.attrs.update({"class":"form-select"})
        self.fields["street"].widget.attrs.update({"class":"form-control"})
        self.fields["number"].widget.attrs.update({"class":"form-control"})
        self.fields["city"].widget.attrs.update({"class":"form-control"})
        self.fields["zip_code"].widget.attrs.update({"class":"form-control"})
        self.fields["phone_number"].widget.attrs.update({"class":"form-control"})
        self.fields["other_phone_number"].widget.attrs.update({"class":"form-control"})

class VisitForm(forms.ModelForm):
    class Meta:
        model = Visit
        fields = ['patient', 'date']
        widgets = {
            'date': forms.DateTimeInput(attrs={'type': 'datetime-local'}),
            'patient': forms.Select()
        }
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.fields["patient"].widget.attrs.update({'class': 'form-control selectpicker', 'data-live-search': 'true'})
        self.fields["date"].widget.attrs.update({"class":"form-control", 'min': now().strftime("%Y-%m-%dT%H:%M")})
