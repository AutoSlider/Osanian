from django.db import models

# Create your models here.
class Summary(models.Model):
    original_text = models.TextField()
    summarized_text = models.TextField()
    author = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)