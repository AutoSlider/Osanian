"""config URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from testsumury import views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('testword/', views.summarizer, name="summarizer"),
    path('testyoutube/', views.summarizer2, name="summarizer2"),
    path('summaries/', views.summary_list, name='summary_list'),
    path('summaries/<int:summary_id>/', views.summary_detail, name='summary_detail'),
    path('summaries/create/', views.summary_create, name='summary_create'),
    path('summaries/<int:summary_id>/edit/', views.summary_edit, name='summary_edit'),
    path('summaries/<int:summary_id>/delete/', views.summary_delete, name='summary_delete'),
    path('summary/<int:pk>/', views.summary_detail, name='summary_detail'),
    path('save_summary/', views.save_summary, name='save_summary'),
    path('delete_summaries/', views.delete_summaries, name='delete_summaries'),
]