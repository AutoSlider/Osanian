from django import forms
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from .models import CustomUser

class SignupForm(UserCreationForm):
    nickname = forms.CharField(max_length=20)
    class Meta: #메타 안에 있는 모델과 필드들을 커스텀 유저에 맞게 수정 작업하는 것
        model=CustomUser
        fields= ['username', 'password1', 'password2', 'nickname', 'phone', 'email']


class ChangeForm(UserChangeForm):
# 회원 정보 수정
    nickname = forms.CharField(max_length=20)
    class Meta(UserChangeForm):
        model = CustomUser
        fields = ['username', 'password', 'nickname', 'phone', 'email']
