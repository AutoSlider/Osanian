from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Summary(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    original_text = models.TextField()
    timeline_text = models.TextField(blank=True, null=True)
    summary_text = models.TextField()
    note = models.TextField(blank=True, null=True)  # note 필드 추가
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    file = models.FileField(blank=True, null=True, upload_to='uploadfile/')
    youtube_url = models.URLField(blank=True, null=True)  # youtube_url

    def __str__(self):
        return self.title
