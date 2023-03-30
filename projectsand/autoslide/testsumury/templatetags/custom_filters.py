from django import template
import re

register = template.Library()

def get_youtube_id(value):
    youtube_id_match = re.search(r'(?<=v=)[^&#]+', value)
    youtube_id_match = youtube_id_match or re.search(r'(?<=be/)[^&#]+', value)
    trailer_id = youtube_id_match.group(0) if youtube_id_match else None
    return trailer_id

register.filter('get_youtube_id', get_youtube_id)
