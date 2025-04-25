# consumers.py
import asyncio
import json
import threading

from channels.generic.websocket import AsyncWebsocketConsumer
from aiortc import RTCPeerConnection, MediaStreamTrack, RTCSessionDescription, RTCIceCandidate

import cv2
import av
from av import VideoFrame

# 全局变量缓存最新帧
latest_frame = None
frame_lock = threading.Lock()

# WebRTC 视频轨道（返回最新帧）
class LiveVideoTrack(MediaStreamTrack):
    kind = "video"
    _counter = 0

    def __init__(self):
        super().__init__()
        self.start_time = asyncio.get_event_loop().time()

    async def recv(self):
        global latest_frame

        # 等待直到有有效帧
        while True:
            with frame_lock:
                if latest_frame is not None:
                    break
            await asyncio.sleep(0.001)  # 避免阻塞事件循环

        # 转换为 AV VideoFrame
        av_frame = VideoFrame.from_ndarray(latest_frame, format="rgb24")

        # 设置时间戳（必须）
        av_frame.pts = self._counter
        av_frame.time_base = av.TimeBase(1, 1000)  # 时间基为毫秒
        self._counter += 1

        return av_frame


from aiortc import RTCPeerConnection


async def handle_offer(offer):
    pc = RTCPeerConnection()
    pc.addTrack(LiveVideoTrack())  # 添加实时视频轨道

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return answer