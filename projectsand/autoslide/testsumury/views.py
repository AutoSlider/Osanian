from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
import torch
import whisper
import pytube
from pytube import YouTube
from moviepy.editor import *
from pydub import AudioSegment
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile


tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')


def summarize(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

    output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5, no_repeat_ngram_size=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def summarizer(request):
    summary = ""
    if request.method == "POST":
        input_text = request.POST["text"]
        summary = summarize(input_text)

    return render(request, "summarizer.html", {"summary": summary})

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


def summarize_long_text(text):
    max_input_length = 1000  # 몇 개의 여유 토큰을 남겨두기 위해 1026보다 작은 값을 사용합니다.
    text_parts = split_text(text, max_input_length)

    summarized_parts = []
    for part in text_parts:
        summary = summarize(part)
        summarized_parts.append(summary)

    return " ".join(summarized_parts)


def summarizer2(request):
    if request.method == "POST":
        input_text = request.POST.get("input_text", "").strip()
        youtube_url = request.POST.get("youtube_url", "").strip()
        video_file = request.FILES.get("video_file", None)

        if input_text:
            # 텍스트 입력이 있는 경우
            text = input_text
            # 긴 텍스트를 요약하는 함수 사용
            summary = summarize_long_text(text)
            timelined_text = "타임라인 정보가 없습니다."
        elif youtube_url:
            # 유튜브 링크 입력이 있는 경우
            youtube = pytube.YouTube(youtube_url)
            video = youtube.streams.first()
            video.download()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            whispermodel = whisper.load_model("small", device=device)
            result = whispermodel.transcribe(video.default_filename)
            text = result["text"]
            segments = result["segments"]

            timelined_text = create_timelined_text(segments)
            summary = summarize_long_text(text)

            os.remove(video.default_filename)
        elif video_file:
            # 파일 업로드가 있는 경우
            file_name = default_storage.save(video_file.name, ContentFile(video_file.read()))
            file_path = default_storage.path(file_name)

            #whisper로 자막화 하는 코드
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            whispermodel = whisper.load_model("small", device=device)
            result = whispermodel.transcribe(file_path)
            text = result["text"]
            segments = result["segments"]

            timelined_text = create_timelined_text(segments)
            summary = summarize_long_text(text)

            # 업로드 된 파일 삭제
            default_storage.delete(file_path)

        else:
            return render(request, "summarizeryoutube.html")

        return render(request, "summarizeryoutube.html", {
            "youtube_url": youtube_url,
            "original_text": text,
            "timeline_text": timelined_text,
            "summary": summary
        })
    else:
        return render(request, "summarizeryoutube.html")


def create_timelined_text(segments):
    timelined_text = []
    for segment in segments:
        start_time = round(segment['start'], 2)
        end_time = round(segment['end'], 2)
        text = segment['text']
        timelined_text.append(f"{start_time}-{end_time}\n{text}")
    return "\n".join(timelined_text)

