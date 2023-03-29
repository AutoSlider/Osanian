from django.db import models

# Create your models here.

class Summary(models.Model):
    title = models.CharField(max_length=200)
    original_text = models.TextField()
    timeline_text = models.TextField(blank=True, null=True)
    summary_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title