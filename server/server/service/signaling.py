import json

from channels.generic.websocket import AsyncWebsocketConsumer
from aiortc import RTCPeerConnection, VideoStreamTrack, RTCSessionDescription


class WebRTCConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.pc = RTCPeerConnection()

    async def receive(self, text_data):
        data = json.loads(text_data)
        if data['type'] == 'offer':
            # 创建Answer
            await self.pc.setRemoteDescription(
                RTCSessionDescription(sdp=data['sdp'], type='offer')
            )
            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)

            # 添加视频处理轨道
            self.pc.addTrack(VideoProcessorTrack())

            await self.send(text_data=json.dumps({
                'type': 'answer',
                'sdp': self.pc.localDescription.sdp
            }))


class VideoProcessorTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.processor = FaceProcessor()

    async def recv(self):
        frame = await self.next_timestamp()

        # 获取视频帧
        img = frame.to_ndarray(format="bgr24")

        # 人脸识别处理
        result_img = self.processor.detect_faces(img)

        # 重构视频帧
        new_frame = VideoFrame.from_ndarray(result_img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame