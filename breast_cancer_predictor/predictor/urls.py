from django.conf.urls import url
from . import views
from django.urls import path
from django.conf import settings


urlpatterns = [    
    path('', views.index, name='index'),
]