from django import forms

class ImageUploadForm(forms.Form):
    xray = forms.ImageField(required=True)