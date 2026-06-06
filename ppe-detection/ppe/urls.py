from django.urls import path
from detection import views

urlpatterns = [
    path('', views.index, name='index'),
]