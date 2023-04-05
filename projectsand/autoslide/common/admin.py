from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.utils.translation import gettext_lazy as _
from .forms import SignupForm, ChangeForm
from .models import CustomUser

# Register your models here.

@admin.register(CustomUser)
class MyUserAdmin(UserAdmin):
    add_form = SignupForm
    form = ChangeForm
    model = CustomUser
    # list_display = UserAdmin.list_display + ('username', 'nickname', 'email', 'phone', 'password', 'is_active', 'date_joined', 'last_login')
    list_display = ('username', 'nickname', 'email', 'phone', 'password', 'is_active', 'date_joined', 'last_login', 'is_staff')
    # print(f'UserAdmin.list_display : {UserAdmin.list_display}') # ('username', 'email', 'first_name', 'last_name', 'is_staff')
    fieldsets = (
        (None, {'fields': ('username', 'password')}),
        (_('Personal info'), {'fields': ('email', 'phone', 'nickname')}),  # 'first_name', 'last_name',
        (_('Permissions'), {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        (_('Important dates'), {'fields': ('last_login', 'date_joined')}),
    )
    add_fieldsets = UserAdmin.add_fieldsets + (
        (_('Additional Fields'), {'fields': ('phone', 'nickname',)}),
    )

    def __init__(self, *args, **kwargs):
        super(UserAdmin, self).__init__(*args, **kwargs)
        # self.fieldsets = self.fieldsets[:2] + self.fieldsets[3:]  # remove first_name and last_name fields


# @admin.register(CustomUser)
# class MyUserAdmin(UserAdmin):
#     add_form = SignupForm
#     form = ChangeForm
#     model = CustomUser
#     list_display = UserAdmin.list_display + ('username', 'nickname', 'email', 'phone', 'password', 'is_active', 'date_joined', 'last_login')
#     fieldsets = UserAdmin.fieldsets + ((_('Additional Fields'), {'fields': ('phone','nickname',)}),
#     ) #this will allow to change these fields in admin module
#     add_fieldsets = UserAdmin.add_fieldsets + (
#         (_('Additional Fields'), {'fields': ('phone','nickname',)}),
#     )

#     def __init__(self, *args, **kwargs):
#         super(UserAdmin, self).__init__(*args, **kwargs)
#         self.fieldsets = self.fieldsets[:2] + self.fieldsets[3:]
