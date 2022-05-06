from django import forms
from django.forms import fields
from .models import image



class imageform(forms.ModelForm):
    class Meta:
        model = image
        fields =['image']