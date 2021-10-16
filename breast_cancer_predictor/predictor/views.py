from django.shortcuts import render, redirect
from . import forms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import plot_precision_recall_curve, plot_roc_curve
# Create your views here.
def index(request):
    return render(request,'index.html')

def predict(request):
    form = forms.Patient()
    if request.method == 'POST':
        form = forms.Patient(request.POST)
        radius_mean = float(request.POST.get('radius_mean'))
        texture_mean = float(request.POST.get('texture_mean'))
        perimeter_mean = float(request.POST.get('perimeter_mean'))
        area_mean = float(request.POST.get('area_mean'))
        smoothness_mean = float(request.POST.get('smoothness_mean'))
        compactness_mean = float(request.POST.get('compactness_mean'))
        concavity_mean = float(request.POST.get('concavity_mean'))
        concave_points_mean = float(request.POST.get('concave_points_mean'))
        symmetry_mean = float(request.POST.get('symmetry_mean'))
        fractal_mean = float(request.POST.get('fractal_mean'))

        patient_info = [radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,
                        compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_mean]


        print(patient_info)

    return render(request, 'predictForm.html', {'form':form})