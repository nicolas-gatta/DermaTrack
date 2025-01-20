from django import forms

class LoginForm(forms.Form):
    login = forms.CharField(
        max_length=100, 
        widget=forms.TextInput(attrs={
            'class': 'fadeIn second', 
            'placeholder': 'login'
        }),
        label = ""
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'fadeIn third', 
            'placeholder': 'password'
        }),
        label = ""
    )