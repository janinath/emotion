from django.urls import path
from . import views
app_name = "realtime_emotion"
urlpatterns = [
    path("", views.index, name="index"),
    path('emotion_detection/', views.emotion_detection, name='emotion_detection'),
]