from django.urls import path
from django.contrib.auth.decorators import login_required
from django.contrib.auth import views as auth_views  # login&logout
from . import views
from boards.views import BoardListView

app_name = 'common'

urlpatterns = [
    # path('', views.IndexView.as_view(), name='main'),
    path('', views.IndexView.as_view(template_name='boards/board_list.html')),
    path('login/', auth_views.LoginView.as_view(template_name='common/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('signup/', views.SignupView.as_view(template_name='common/signup.html'), name='signup'),
    path('profile/', views.profile, name='profile'),
    path('produce/', views.produce, name='produce'),
]