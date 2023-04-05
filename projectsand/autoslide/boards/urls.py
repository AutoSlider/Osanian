from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.views.decorators.csrf import csrf_exempt

app_name = 'boards'

urlpatterns = [
    path('list/', views.BoardListView.as_view(), name='board_list'),  # board_list 기본 & 목록
    path('favorite/', views.FavoriteBoardListView.as_view(), name='favorite_board_list'),
    path('new/', views.BoardCreateView.as_view(), name='board_create'),
    path('<int:pk>/', views.BoardDetailView.as_view(), name='board_detail'),
    path('<int:pk>/modifav', views.modifiy_favorite, name='modifiy_favorite'),
    # path('<int:pk>/update/', views.BoardUpdateView.as_view(), name='board_update'),
    # path('<int:pk>/delete/', views.BoardDeleteView.as_view(), name='board_delete'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
#  MEDIA_URL로 접근하면 MEDIA_ROOT 경로에 저장된 파일을 서비스하도록 설정함. 이제 파일 업로드 뷰에서 파일을 업로드하면 MEDIA_ROOT에 파일이 저장되고, 데이터베이스에는 파일 경로가 저장됨.

handler403 = 'django.views.defaults.permission_denied'
handler404 = 'django.views.defaults.page_not_found'
handler500 = 'django.views.defaults.server_error'
