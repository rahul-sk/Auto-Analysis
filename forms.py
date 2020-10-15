from django import forms
from django.contrib.auth.models import User
import io
import csv

class DataForm(forms.Form):
    data_file = forms.FileField()

   