# yourapp/routing.py
from django.urls import re_path
from .controller import webrtc

websocket_urlpatterns = [
    re_path(r'^ws/webrtc/?$', webrtc.WebRTCConsumer.as_asgi()),
]