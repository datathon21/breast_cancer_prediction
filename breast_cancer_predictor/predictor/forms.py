from django import forms
from django.core import validators

class Patient(forms.Form):
    radius_mean = forms.FloatField()
    texture_mean = forms.FloatField()
    perimeter_mean = forms.FloatField()
    area_mean = forms.FloatField()
    smoothness_mean = forms.FloatField()
    compactness_mean = forms.FloatField()
    concavity_mean = forms.FloatField()
    concave_points_mean = forms.FloatField()
    symmetry_mean = forms.FloatField()
    fractal_mean = forms.FloatField()

    def clean(self):
        all_clean_data = super().clean()