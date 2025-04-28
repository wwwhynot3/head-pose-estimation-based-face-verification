import asyncio
import json
import aiortc
import aiortc.sdp
import cv2
import numpy as np
from av import VideoFrame
from channels.generic.websocket import AsyncWebsocketConsumer
from aiortc import RTCConfiguration, RTCIceGatherer, RTCIceServer, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.rtcrtpsender import RTCRtpSender
from aiortc.rtcicetransport import RTCIceCandidate

class ProcessedVideoTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.frame_queue = asyncio.Queue(maxsize=1)
        self.running = True

    async def recv(self):
        print("videotrackrev")
        pts, time_base = await self.next_timestamp()
        print(f"Received frame with pts: {pts}, time_base: {time_base}")
        frame = await self.frame_queue.get()
        print('1')
        frame.pts = pts
        print('2')
        frame.time_base = time_base
        print('3')
        # cv2.imshow("Processed Frame", frame.to_ndarray(format="bgr24"))
        print('4')
        return frame

    async def put_frame(self, frame):
        print("put_frame")
        if self.frame_queue.full():
            await self.frame_queue.get()  # 丢弃旧帧
        await self.frame_queue.put(frame)

class WebRTCConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pc = None
        self.processed_track = None

    async def connect(self):
        await self.accept()
        
        # 配置ICE服务器
        config = RTCConfiguration(iceServers=[
            RTCIceServer(urls="stun:stun.l.google.com:19302"),
            # RTCIceServer(urls="turn:your-turn-server.com", username="user", credential="password")
        ])
        self.pc = RTCPeerConnection(config)
        self.processed_track = ProcessedVideoTrack()

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

    async def receive(self, text_data):
        data = json.loads(text_data)
        print(f'{self.pc.connectionState}')
        if data['type'] == 'offer':
            await self.handle_offer(data)
        elif data['type'] == 'candidate':
            await self.handle_candidate(data['candidate'])
            # print(f"Received ICE candidate: {data['candidate']}")
            pass
        elif data['type'] == 'data':
            await self.handle_data(data['data'])
        else:
            print(f"Unknown message type: {data['type']}")
        

    async def handle_candidate(self, data):
        print(f'Received ICE candidate')
        candidate = aiortc.sdp.candidate_from_sdp(data["candidate"])
        candidate.sdpMid = data["sdpMid"]
        candidate.sdpMLineIndex = data["sdpMLineIndex"]
        await self.pc.addIceCandidate(candidate)

    async def handle_offer(self, offer_data):
        # print(f"Handling SDP Offer...{offer_data['sdp']}")
        # 添加处理轨道
        # self.pc.addTrack(self.processed_track)
        # candidate = aiortc.sdp.candidate_from_sdp(offer_data["sdp"])
        # 设置ICE候选
        # self.pc.addIceCandidate(candidate)
        # 设置远端描述
        offer = RTCSessionDescription(
            sdp=offer_data["sdp"], 
            type=offer_data["type"]
        )
        await self.pc.setRemoteDescription(offer)

        # 创建并发送 Answer
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        
        # print(f"Sending SDP Answer: {answer.sdp}")
        await self.send(json.dumps({
            "type": answer.type,
            "sdp": answer.sdp
        }))

        # 处理传入的视频流
        @self.pc.on("track")
        async def on_track(track):
            print(f"Track received: {track.kind}")
            if track.kind == "video":
                while True:
                    try:
                        frame = await track.recv()
                        img = frame.to_ndarray(format="bgr24")
                        cv2.imshow("Received Frame", img)
                        # 示例：将帧转为灰度
                        processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        processed_frame = VideoFrame.from_ndarray(
                            processed, 
                            format="bgr24"  # 注意格式匹配
                        )
                        
                        await self.processed_track.put_frame(processed_frame)
                    except asyncio.CancelledError:
                        print("Video track processing cancelled.")
                        break
                    except Exception as e:
                        print(f"Error processing video frame: {e}")
                        break
                    
    async def handle_data(self, data):
        print(f"Handling extra data: {data}")
        # 处理额外数据
        pass