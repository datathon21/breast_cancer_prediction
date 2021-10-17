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

        # pandas dataframe
        df = pd.read_csv('predictor/cleaned_data.csv')
        X = df.drop('diagnosis',axis=1)
        y = df['diagnosis']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
        scaler = StandardScaler()
        scaled_X_train = scaler.fit_transform(X_train)
        scaled_X_test = scaler.transform(X_test)

        log_model = LogisticRegressionCV()

        log_model.get_params()

        log_model.fit(scaled_X_train, y_train)
        y_pred = log_model.predict(scaled_X_test)

        log_model.C_
        log_model.get_params()

        confusion_matrix(y_test,y_pred)
        print(classification_report(y_test,y_pred))

        print(log_model.predict(patient_info))
        print(log_model.predict_proba(patient_info))
        # return '<h1>result: {}, proba: {}</h1>'.format(str(log_model.predict(patient_info)), str(log_model.predict_proba(patient_info)))

    return render(request, 'predictForm.html', {'form':form})