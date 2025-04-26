# consumers.py
import asyncio
import json
from typing import override

import numpy as np
from av.video.frame import VideoFrame
from channels.generic.websocket import AsyncWebsocketConsumer
from aiortc import RTCPeerConnection, VideoStreamTrack, RTCSessionDescription


class FaceVideoTransformTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.latest_frame = None
        self.lock = asyncio.Lock()

    async def recv(self):
        print('track recv')
        # 此方法在aiortc接收帧时自动调用
        pts, time_base = await self.next_timestamp()
        async with self.lock:
            if self.latest_frame is None:
                # 返回黑帧直到有数据
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                frame = self.latest_frame
        # 转换帧为av.VideoFrame格式
        av_frame = VideoFrame.from_ndarray(frame, format='bgr24')
        av_frame.pts = pts
        av_frame.time_base = time_base
        return av_frame

    async def update_frame(self, new_frame):
        async with self.lock:
            self.latest_frame = new_frame

class WebRTCConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        await self.accept()
        self.pc = RTCPeerConnection()
        self.video_track = FaceVideoTransformTrack()  # 处理视频的Track

    async def disconnect(self, close_code):
        await self.pc.close()

    @override
    async def receive(self, text_data=None, bytes_data=None):
        try:
            data = json.loads(text_data)
            print(f"Received data: {data}")
            # 处理SDP Offer/Answer交换
            if data['type'] == 'offer':
                await self.handle_offer(data)
        except Exception as e:
            print(f"Error processing message: {e}")


    async def handle_offer(self, offer):
        # 添加本地视频Track
        self.pc.addTrack(self.video_track)
        # 设置远端描述并创建Answer
        await self.pc.setRemoteDescription(offer)
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        # 发送Answer给前端
        await self.send(json.dumps({
            'type': 'answer',
            'sdp': self.pc.localDescription.sdp
        }))