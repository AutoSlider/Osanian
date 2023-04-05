from django.db import models
from django.utils import timezone
from common.models import CustomUser

TIME_ZONE = 'Asia/Seoul'

# Create your models here.

class Board(models.Model):
    input_type_choices = [
        ('input_text', '텍스트 요약'),
        ('input_youtube', 'Youtube 영상 링크 입력'),
        ('input_video', '영상 파일 업로드')
    ]
    title = models.TextField(blank=True)
    input_type = models.TextField(choices=input_type_choices, null=True)
    input_text = models.TextField(blank=True, null=True)
    input_youtube = models.TextField(blank=True, null=True)
    input_video = models.FileField(upload_to='videos/', blank=True, null=True)
    total_text = models.TextField(blank=True, null=True)
    summary_text = models.TextField(blank=True, null=True)
    timeline_text = models.TextField(blank=True, null=True)
    note = models.TextField(blank=True, null=True)
    user_id = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    favorite = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        if (not self.title.strip()) or (self.title == "") or (self.title is None) : # 제목이 없는 경우에만 생성 날짜를 저장
            now = timezone.now()
            formatted_time = now.strftime('%Y-%m-%d %H:%M')
            self.title = formatted_time # str(self.created_at)
        super().save(*args, **kwargs)

    def __str__(self):
        return f'id={self.id} / user = {self.user_id.username} : {self.title}'
