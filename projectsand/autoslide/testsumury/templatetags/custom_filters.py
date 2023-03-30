from django import template
import re
from django.utils.safestring import mark_safe

register = template.Library()

# def get_youtube_id(value):
#     youtube_id_match = re.search(r'(?<=v=)[^&#]+', value)
#     youtube_id_match = youtube_id_match or re.search(r'(?<=be/)[^&#]+', value)
#     trailer_id = youtube_id_match.group(0) if youtube_id_match else None
#     return trailer_id
#
# register.filter('get_youtube_id', get_youtube_id)
@register.filter
def get_youtube_id(value):
    if not value:
        return None
    youtube_id_match = re.search(r'(?<=v=)[^&#]+', value)
    youtube_id_match = youtube_id_match or re.search(r'(?<=be/)[^&#]+', value)
    return youtube_id_match.group(0) if youtube_id_match else None

@register.filter
def add_timestamp_links(text):
    lines = text.split('\n')
    result = []
    for line in lines:
        parts = line.split(' ')
        new_parts = []
        for part in parts:
            if ':' in part and part.endswith(','):
                timestamp = part[:-1]
                link = '<a href="#" onclick="seekToTimestamp({});">{}</a>, '.format(timestamp, part)
                new_parts.append(link)
            else:
                new_parts.append(part)
        result.append(' '.join(new_parts))
    return mark_safe('\n'.join(result))
