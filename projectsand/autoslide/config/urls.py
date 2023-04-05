"""config URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
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
from django.urls import path, include



urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('common.urls', 'common')), # '/' 에 해당되는 path
    path('', include('boards.urls', 'boards')),
]

# from django.contrib import admin
# from django.urls import path, include
# from django.conf import settings
# from django.conf.urls.static import static
# from django.views.generic.base import TemplateView
# from common import views

# urlpatterns = [
#     path('admin/', admin.site.urls),
#     # path('', views.IndexView.as_view(), name='index'),  # '/' 에 해당되는 path
#     # path('common/', include(('common.urls', 'common'))),
#     # path('common/', include('django.contrib.auth.urls')),
#     # path('care/', include('care.urls')), # 요양원&리뷰
#     path('', include(('common.urls', 'common'))),
#     path('common/', include('django.contrib.auth.urls')),
#     path('care/', include('care.urls')), # 요양원&리뷰
#     path('care/', include('django.contrib.auth.urls')),
# ]

