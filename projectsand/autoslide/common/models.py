from django.db import models
from django.contrib.auth.models import AbstractUser
from django.core.validators import RegexValidator

class CustomUser(AbstractUser):
    nickname = models.CharField(max_length=20, default='')
    phone_regex = RegexValidator(regex=r'^01([0|1|6|7|8|9])-?([0-9]{3,4})-?([0-9]{4})$') # 정규식 011-123-4567 혹은 010-1234-5678 기준
    phone = models.CharField(validators=[phone_regex], max_length=13, unique=True, verbose_name='phone', null=False, blank=False) # 010-1234-5678 기준 갯수

    USERNAME_FIELD = 'username'
    REQUIRED_FIELDS = ['phone', 'nickname', 'email']
    first_name = None
    last_name = None

    def __str__(self):
        return self.username