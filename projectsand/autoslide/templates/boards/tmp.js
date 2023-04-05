// $(".textSection").find(".content").text('{{ object.total_text }}');
$(".textSection").find(".content").css("background-color","#010075");
// text_input 에서 전문, 요약문 선택
/*
var check = $("#toggleText");
check.on('click', (event) => {
$(".opt").toggleClass("on");
if ($(".total_text").hasClass("on")){
$(".textSection .content").text("");
$(".textSection .content").text("{{ object.total_text }}");
} else{
$(".textSection .content").text("");
$(".textSection .content").text("{{ object.summary_text }}");
}
});
*/
/*
// 유튜브 API를 비동기적으로 로드합니다.
function youtubeUrlValidation(url) {
    const youtubeRegex = /(https?:\/\/)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)\/(watch\?v=|embed\/|v\/|.+\?v=)?([^&=%\?]{11})/;
    const youtubeRegexMatch = url.match(youtubeRegex);
    if (youtubeRegexMatch) { return youtubeRegexMatch[6]; }
    return null;
}
var video_id = youtubeUrlValidation("{{object.input_youtube}}");
var tag = document.createElement('script');
tag.src = "https://www.youtube.com/iframe_api";
var firstScriptTag = document.getElementsByTagName('script')[0];
firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);
var player;
var done = false;

function stopVideo() {
    player.stopVideo();
}
function onPlayerReady(event) {
    event.target.playVideo();
}
function onPlayerStateChange(event){
    if (event.data === 0) { stopVideo() }
}
function onYouTubeIframeAPIReady() {
    player = new YT.Player('player', {
        height: '325',
        width: '525',
        videoId: video_id,
        playerVars: {
        'playsinline': 1
        },
        events: {
        //'onReady': onPlayerReady,  // 새로고침 시 자동재생 안 되도록 주석 처리
        'onStateChange': onPlayerStateChange
        }
    });
}
*/