from django.contrib import admin
from .models import Board

# Register your models here.

# admin.site.register(Board)
@admin.register(Board)
class BoardAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'user_id', 'created_at', 'updated_at')
    list_filter = ('favorite', 'input_type', 'created_at', 'updated_at')
    search_fields = ('title', 'input_type', 'note', 'user_id__username')

