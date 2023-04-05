# from django.contrib.auth import get_user_model, login, authenticate, logout
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse_lazy, reverse
from django.views.generic import ListView, DetailView, CreateView, View
from django.http import JsonResponse, HttpResponseForbidden, HttpResponseRedirect
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings

from django.db.models import Q

from .models import Board
from .forms import BoardCreateForm, BoardNoteForm

import os
import json
import torch
import whisper
import pytube
from pytube import YouTube
from pydub import AudioSegment
from moviepy.editor import *
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

# error log
import logging

#logging.basicConfig(level=logging.DEBUG)
#logger = logging.getLogger(__name__)

#def my_view(request):
    #logger.info('boards/views.py Log start!')


# Create your views here.

# BoardListView
# 모든 보드 목록
# LoginRequiredMixin을 상속받아 로그인한 사용자만 접근할 수 있도록 설정
class BoardListView(LoginRequiredMixin, ListView):
    model = Board
    template_name = 'boards/board_list.html'
    # paginate_by = 10

    def get_queryset(self):
        # Filter the queryset to include only boards created by the current user
        # return Board.objects.filter(user_id=self.request.user.id)
        query = self.request.GET.get('q')
        if query:
            return Board.objects.filter(
                Q(title__icontains=query) | Q(note__icontains=query) | Q(total_text__icontains=query) | Q(summary_text__icontains=query),
                user_id=self.request.user.id
            )
        else:
            return Board.objects.filter(user_id=self.request.user.id)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['board_list'] = self.get_queryset()
        return context

    def post(self, request, *args, **kwargs):
        if request.POST.get('board_ids'):
            board_ids = request.POST.getlist('board_ids')
            Board.objects.filter(id__in=board_ids).delete()
            # boards = Board.objects.filter(id__in=board_ids)
            # for board in boards:
            #     board.delete()
        return redirect('boards:board_list')


# 즐겨찾기 인 경우 override함.
class FavoriteBoardListView(BoardListView):
    def get_queryset(self):
        queryset = super().get_queryset().filter(favorite=True)
        query = self.request.GET.get('q')
        if query:
            queryset = queryset.filter(
                Q(title__icontains=query) | Q(note__icontains=query) | Q(total_text__icontains=query) | Q(summary_text__icontains=query)
            )
        return queryset

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['favorite_board_list'] = self.get_queryset()
        return context

    def post(self, request): # 즐겨찾기 해제
        if 'board_fav_ids' in request.POST:
            # board_ids = request.POST.get('board_fav_ids')
            board_ids = request.POST.get('board_fav_ids').split(',')
            boards = Board.objects.filter(id__in=board_ids)
            for board in boards:
                board.favorite = not board.favorite
                board.save()
        return redirect('boards:favorite_board_list')


# 즐겨찾기 추가/삭제
def modifiy_favorite(request, pk):
    board = get_object_or_404(Board, pk=pk)
    board.favorite = not board.favorite
    board.save()
    response_data = {'favorite': board.favorite}

    if request.method == 'POST':
        if request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest':
            return JsonResponse(response_data)
    return HttpResponseRedirect(reverse('boards:board_detail', args=[pk]))



# 상세 보드
class BoardDetailView(LoginRequiredMixin, DetailView):
    model = Board
    template_name = 'boards/board_detail.html'
    form_class = BoardNoteForm  # BoardCreateForm
    # success_url = reverse_lazy('board_detail')

    # 로그인한 사용자와 요약 작성자가 일치하지 않으면 에러 페이지 반환
    def get_object(self, queryset=None):
        obj = get_object_or_404(Board, id=self.kwargs['pk'])
        if obj.user_id != self.request.user:
            raise HttpResponseForbidden("You are not allowed to view this summary.")
        return obj

    def form_valid(self, form):
        form.save()
        return super().form_valid(form)

    def get_success_url(self):
        return reverse('boards:board_detail', kwargs={'pk': self.object.pk})


# 보드 삭제
class BoardDeleteView(LoginRequiredMixin, View):
    def post(self, request):
        if 'board_ids' in request.POST:
            board_ids = request.POST.get('board_ids').split('-')
            Board.objects.filter(id__in=board_ids).delete()
        return redirect('boards:board_list')




# 영상 업로드 & youtube 다운 & 긴 텍스트 요약 처리

# chocolatey 로 ffmpge 설치 했으나 경로 지정 필요
def get_ffmpeg_path():
    search_paths = [
        r'C:\Program Files\ffmpeg\bin',
        r'C:\Program Files (x86)\ffmpeg\bin',
        r'C:\ProgramData\chocolatey\bin',
    ]

    for path in search_paths:
        ffmpeg_path = os.path.join(path, 'ffmpeg.exe')
        if os.path.exists(ffmpeg_path):
            return ffmpeg_path

    return None

ffmpeg_path = get_ffmpeg_path()
if ffmpeg_path:
    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]
    os.environ["FFMPEG_BINARY"] = ffmpeg_path # FFmpeg 경로 설정
    AudioSegment.ffmpeg = ffmpeg_path  # pydub 패키지가 ffmpeg.exe 경로를 인식하도록 함.
else:
    print("FFmpeg not found in any search path")


tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')


# 텍스트 분리
def split_text(text, max_length):
    words = text.split()
    parts = []
    current_part = []

    for word in words:
        current_part.append(word)
        if len(tokenizer.encode(" ".join(current_part))) > max_length:
            current_part.pop()
            parts.append(" ".join(current_part))
            current_part = [word]

    if current_part:
        parts.append(" ".join(current_part))

    return parts


# 텍스트 요약
def summarize(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

    output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5, no_repeat_ngram_size=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)


# 전문 텍스트 요약
def summarize_long_text(text):
    max_input_length = 1000  # 몇 개의 여유 토큰을 남겨두기 위해 1026보다 작은 값을 사용합니다.
    text_parts = split_text(text, max_input_length)

    summarized_parts = []
    for part in text_parts:
        summary = summarize(part)
        summarized_parts.append(summary)

    return " ".join(summarized_parts)


# timestamp track -> timeline 생성
def create_timelined_text(segments):
    timelined_text = []
    for segment in segments:
        segment_start = round(segment['start'], 2)
        segment_text = segment['text']
        segment_t = round(segment_start, 2)
        segment_text_with_t = f'<span t="{segment_t}" data-lexical-text="true" style="" onclick="seekToTimestamp(\'{segment_t}\');">{segment_text}</span>'
        timelined_text.append(segment_text_with_t)
    return "\n".join(timelined_text)


import re
def youtube_url_validation(url):
    youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')

    youtube_regex_match = re.findall(youtube_regex, url)
    if youtube_regex_match:
        return youtube_regex_match[0][-1]


# 요약 종합 & 보드 생성
class BoardCreateView(LoginRequiredMixin, CreateView):
    model = Board
    form_class = BoardCreateForm
    template_name = "boards/board_form.html"

    def post(self, request, *args, **kwargs):
        #logger.info('BoardCreateForm form_valid called!')
        #logger.debug('Debugging information')
        form = self.get_form()
        if form.is_valid():
            return self.form_valid(form)
        else:
            return self.form_invalid(form)


    # def form_validated(self, form):
    def form_valid(self, form):
        #logger.info('BoardCreateForm form_valid called!')
        #logger.debug('Debugging information')

        board = form.save(commit=False)
        form.instance.user_id = self.request.user

        # Handle text summary
        input_text = form.cleaned_data['input_text']
        if input_text:
            board.total_text = input_text
            board.summary_text = summarize_long_text(input_text)
            board.timeline_text = ""
            board.save()
            # Define the URL to redirect to
            redirect_url = reverse('boards:board_detail', args=[board.id])
            return redirect(redirect_url)

        # Handle YouTube links
        input_youtube = form.cleaned_data['input_youtube']
        if input_youtube:

            # 동영상 다운로드를 위한 경로 설정
            VIDEO_DIR = os.path.join(settings.MEDIA_ROOT, 'youtube/')
            if not os.path.exists(VIDEO_DIR):
                os.mkdir(VIDEO_DIR)

            # 다운로드할 동영상의 URL
            youtube = pytube.YouTube(input_youtube)

            # 비디오 다운로드
            video = youtube.streams.filter(file_extension='mp4').first()
            # only_audio=True, -> 음성만
            video.download(output_path=VIDEO_DIR) #  filename=f'audio_

            # Run deep learning model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            whispermodel = whisper.load_model("small", device=device) # medium

            result = whispermodel.transcribe(os.path.join(VIDEO_DIR,video.default_filename))
            original_text = result["text"]
            segments = result["segments"]
            # Board 인스턴스에 저장.
            board.title = youtube.title
            board.total_text = result["text"]
            board.summary_text = summarize_long_text(original_text)
            board.timeline_text = create_timelined_text(segments)
            board.total_text = original_text

            board.save()

            os.remove(os.path.join(VIDEO_DIR,video.default_filename))
            redirect_url = reverse('boards:board_detail', args=[board.id])
            return redirect(redirect_url)

        # Video file processing
        input_video = form.cleaned_data['input_video']
        if input_video:
            # 파일 업로드가 있는 경우
            file_name = default_storage.save(input_video.name, ContentFile(input_video.read()))
            file_path = default_storage.path(file_name)

            #whisper로 자막화 하는 코드
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            whispermodel = whisper.load_model("small", device=device)
            result = whispermodel.transcribe(file_path)
            # original_text = result["text"]
            segments = result["segments"]
            # Board 인스턴스에 저장.
            board.title = file_path.split('\\')[-1] # youtube.title
            board.total_text = result["text"]
            board.summary_text = summarize_long_text(result["text"])
            board.timeline_text = create_timelined_text(segments)
            board.input_video = input_video # 업로드한 파일
            board.save()

            # 업로드 된 파일 삭제
            default_storage.delete(file_path)

            redirect_url = reverse('boards:board_detail', args=[board.id])
            return redirect(redirect_url)

        success_url = reverse('boards:board_list')
        #logger.info(f'Redirecting to success_url -> {success_url}')
        return super().form_valid(form)

    def get_success_url(self):
        return reverse('boards:board_list')

