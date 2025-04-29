import asyncio
import json
import aiortc
import aiortc.sdp
import cv2
import numpy as np
from av import VideoFrame
from channels.generic.websocket import AsyncWebsocketConsumer
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack

class ProcessedVideoTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.frame_queue = asyncio.Queue(maxsize=1)
        self.running = True
        self.count = 0
        

    async def recv(self):
        # print("videotrackrev")
        pts, time_base = await self.next_timestamp()
        frame = await self.frame_queue.get()
        pic = frame.to_ndarray(format="bgr24")
        cv2.putText(pic, f'{self.count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame = VideoFrame.from_ndarray(pic, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        self.count += 1
        # cv2.imshow("Processed Frame", frame.to_ndarray(format="bgr24"))
        return frame

    async def put_frame(self, frame):
        if self.frame_queue.full():
            await self.frame_queue.get()  # 丢弃旧帧
        await self.frame_queue.put(frame)

class CameraVideoTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)  # 打开摄像头

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Camera frame not available")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

class WebRTCConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pc = None
        self.processed_track = None
        self.account = None

    async def connect(self):
        await self.accept()
        
        # 配置ICE服务器
        config = RTCConfiguration(iceServers=[
            RTCIceServer(urls="stun:stun.l.google.com:19302"),
            # RTCIceServer(urls="turn:your-turn-server.com", username="user", credential="password")
        ])
        self.pc = RTCPeerConnection(config)
        self.processed_track = ProcessedVideoTrack()
        # self.processed_track = CameraVideoTrack()
        # # 添加候选收集监听
        @self.pc.on("icecandidate")
        async def on_icecandidate(candidate):
            print(f"ICE candidate generated: {candidate}")
            if candidate:
                await self.send(json.dumps({
                    "type": "candidate",
                    "candidate": candidate.candidate,
                    "sdpMid": candidate.sdpMid,
                    "sdpMLineIndex": candidate.sdpMLineIndex
                }))

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"Connection state changed: {self.pc.connectionState}")
            if self.pc.connectionState == "failed":
                await self.pc.close()

        @self.pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            print(f"ICE connection state changed: {self.pc.iceConnectionState}")
            pass

    async def disconnect(self, close_code):
        print("Disconnecting WebRTC...")
        if self.pc:
            await self.pc.close()
        if self.processed_track:
            self.processed_track.running = False
        print("WebRTC disconnected.")

    async def receive(self, text_data=None, bytes_data=None):
        data = json.loads(text_data)
        # print(f'{self.pc.connectionState}')
        if data['type'] == 'offer':
            await self.handle_offer(data)
        elif data['type'] == 'candidate':
            await self.handle_candidate(data['candidate'])
            # print(f"Received ICE candidate: {data['candidate']}")
            pass
        elif data['type'] == 'account':
            self.account = data['data']
            print(f"Account set: {self.account}")
        elif data['type'] == 'data':
            await self.handle_data(data['data'])
        else:
            print(f"Unknown message type: {data['type']}")
        

    async def handle_candidate(self, data):
        # print(f'Received ICE candidate')
        candidate = aiortc.sdp.candidate_from_sdp(data["candidate"])
        candidate.sdpMid = data["sdpMid"]
        candidate.sdpMLineIndex = data["sdpMLineIndex"]
        await self.pc.addIceCandidate(candidate)

    async def handle_offer(self, offer_data):
        self.pc.addTrack(self.processed_track)
        @self.pc.on("track")
        async def on_track(track):
            print(f"Track received: {track.kind}")
            if track.kind == "video":
                while True:
                    try:
                        frame = await track.recv()
                        # 示例：处理接收到的视频帧
                        # img = frame.to_ndarray(format="bgr24")
                        # processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        # # cv2.imshow("Processed Frame", processed)
                        # processed_frame = VideoFrame.from_ndarray(
                        #     processed,
                        #     format="gray"
                        # )
                        await self.processed_track.put_frame(frame=frame)
                    except asyncio.CancelledError:
                        print("Video track processing cancelled.")
                        break
                    except Exception as e:
                        print(f"Error processing video frame: {e}")
                        break
        offer = RTCSessionDescription(
            sdp=offer_data["sdp"], 
            type=offer_data["type"]
        )
        await self.pc.setRemoteDescription(offer)
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        await self.send(json.dumps({
            "type": answer.type,
            "sdp": answer.sdp
        }))


                    
    async def handle_data(self, data):
        print(f"Handling extra data: {data}")
        # 处理额外数据
        pass