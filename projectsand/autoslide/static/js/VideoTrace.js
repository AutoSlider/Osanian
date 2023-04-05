$(document).ready(function() {
    console.log('VideoTrace.js Loaded');

    // 유튜브 API를 비동기적으로 로드합니다.
    if ('{{ summary.youtube_url }}') {
        const tag = document.createElement('script');
        tag.src = "https://www.youtube.com/iframe_api";
        const firstScriptTag = document.getElementsByTagName('script')[0];
        firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);
    }

    // 작성된 글자를 클릭할 때 호출되는 함수입니다.
    function seekToTimestamp(time) {
        if (player && typeof player.seekTo === "function") {
            player.seekTo(time, true);
        } else if (video) {
            video.currentTime = time;
        }
    }

    // 영상 시간에 따라 자막 색상을 변경하는 함수를 작성하고, 이 함수를 비디오의 'timeupdate' 이벤트 리스너에 연결

    function updateCaptionColor() {
        let currentTime;

        if (player && typeof player.getCurrentTime === "function") {
            currentTime = player.getCurrentTime();
        } else if (video) {
            currentTime = video.currentTime;
        } else {
            return;
        }
    }

    const captionSpans = document.querySelectorAll('[data-lexical-text]');
    console.log("updateCaptionColor called, current time: ", currentTime);
    captionSpans.forEach((span, index) => {
        const spanTime = parseFloat(span.getAttribute('t'));
        console.log(spanTime); // 현재 spanTime 값을 콘솔에 출력

        let nextSpanTime;

        if (index + 1 < captionSpans.length) {
            nextSpanTime = parseFloat(captionSpans[index + 1].getAttribute('t'));
        } else {
            nextSpanTime = Infinity; // 마지막 span이므로 끝 시간을 무한대로 설정
        }

        if (currentTime >= spanTime && currentTime < nextSpanTime) {
            span.style.backgroundColor = 'red';
        } else {
            span.style.backgroundColor = '';
        }
    });




    //   data-lexical-time 속성이 있는 글자를 클릭하면 해당 시간대로 비디오를 이동시키는 코드
    document.querySelectorAll('[data-lexical-time]').forEach((span) => {
        span.addEventListener('click', () => {
            const timestamp = span.getAttribute('data-lexical-time');
            seekToTimestamp(parseTimestamp(timestamp));
        });
    });

    // "핏"과 같은 형식으로 작성된 글자에서 시간대를 파싱하는 함수입니다.
    function parseTimestamp(timestamp) {
        const parts = timestamp.split(':');
        const hours = parseInt(parts[0]);
        const minutes = parseInt(parts[1]);
        const seconds = parseFloat(parts[2].replace(',', '.'));
        return hours * 3600 + minutes * 60 + seconds;
    }

    let player;
    let video;

    // 유튜브 API에서 로드되면 호출되는 함수입니다.
    function onYouTubeIframeAPIReady() {
        const youtubeId = '{{ summary.youtube_url|get_youtube_id }}';
        if (youtubeId) {
            player = new YT.Player('youtube-player', {
                height: '360',
                width: '640',
                videoId: youtubeId, // 재생할 유튜브 비디오의 ID를 입력합니다.
                events: {
                    'onReady': onVideoLoaded,
                }
            });
        }
    }


    // 비디오가 로드된 후 호출되는 함수입니다.
    function onVideoLoaded() {
    setInterval(updateCaptionColor, 100); // 0.1초마다 updateCaptionColor() 함수를 호출합니다.
    console.log('onVideoLoaded called');
    }
    // HTML5 비디오를 초기화합니다.

    document.addEventListener('DOMContentLoaded', function() {
        video = document.getElementById('uploadVideo');
        video.addEventListener('loadedmetadata', onVideoLoaded); // 비디오가 로드된 후에 onVideoLoaded() 함수를 호출합니다.
        video.addEventListener('error', function() {
            console.log('Error loading video'); // 에러가 발생한 경우 콘솔에 메시지를 출력합니다.
        });
    });
});