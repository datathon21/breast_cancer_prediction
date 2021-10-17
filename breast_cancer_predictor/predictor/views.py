from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
from . import forms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import plot_precision_recall_curve, plot_roc_curve
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)  


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

        patient_info = [[radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,
                        compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_mean]]

        df = pd.read_csv('predictor/cleaned_data.csv')
        X = df.drop('diagnosis',axis=1)
        y = df['diagnosis']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

        log_model = LogisticRegressionCV()

        log_model.fit(X_train, y_train)
        y_pred = log_model.predict(X_test)

        result = float(log_model.predict(patient_info)[0])

        if result > 0:
            msg = "Our model predicts that this mass is cancerous"
        else:
            msg = "Our model predicts that this mass is not cancerous"
        accuracy = round(accuracy_score(y_test, y_pred) * 100,2)
        precision = round(precision_score(y_test, y_pred) * 100, 2)
        recall = round(recall_score(y_test, y_pred) * 100, 2)

        return render(request, 'results.html', {'msg':msg, 'accuracy':accuracy, 'precision':precision,'recall':recall})
        
    return render(request, 'predictForm.html', {'form':form})

def details(request):
    return render(request, 'showPage.html')