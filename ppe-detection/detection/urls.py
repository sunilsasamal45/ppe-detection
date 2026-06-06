from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('start_stream/', views.start_stream, name='start_stream'),
    path('stop_stream/', views.stop_stream, name='stop_stream'),
]