from django import forms
from .models import Board

class BoardCreateForm(forms.ModelForm):
    class Meta:
        model = Board
        fields = ['input_type', 'input_text', 'input_youtube', 'input_video', ] # 'title', 'note', 'favorite', 'total_text', 'summary_text', 'timeline_text',
        widgets = {
            # base.html
            'input_type': forms.Select(choices=Board.input_type_choices, attrs={'class': 'input_type'}),
            'input_text': forms.Textarea(attrs={'id': 'input_text'}),
            'input_youtube': forms.TextInput(attrs={'id': 'input_youtube'}),
            'input_video': forms.ClearableFileInput(attrs={'id': 'input_video'}),
            # 'title': forms.TextInput(attrs={'class': 'form-control'}),
            # 'total_text': forms.Textarea(attrs={'class': 'form-control'}), # , 'rows': 5
            # 'summary_text': forms.Textarea(attrs={'class': 'form-control'}),
            # 'timeline_text': forms.Textarea(attrs={'class': 'form-control'}),
            # 'note': forms.Textarea(attrs={'class': 'form-control'}),
        }

class BoardNoteForm(forms.ModelForm):
    class Meta:
        model = Board
        fields = ['note']
        widgets = {
            'note': forms.Textarea(attrs={'id': 'user_note'}),
        }
