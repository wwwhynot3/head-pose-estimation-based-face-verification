<template>
    <ion-page>
      <ion-header :translucent="true">
        <ion-toolbar>
          <ion-buttons slot="start">
            <ion-menu-button color="primary"></ion-menu-button>
          </ion-buttons>
          <ion-title>WebRTC Camera</ion-title>
        </ion-toolbar>
      </ion-header>
  
      <ion-content :fullscreen="true">
        <div class="video-container">
          <video ref="videoElement" autoplay playsinline></video>
        </div>
      </ion-content>
    </ion-page>
  </template>
  
  <script setup lang="ts">
  import { ref, onMounted, onUnmounted } from 'vue';
  import { IonButtons, IonContent, IonHeader, IonMenuButton, IonPage, IonTitle, IonToolbar } from '@ionic/vue';
  
  const videoElement = ref<HTMLVideoElement>();
  let localStream: MediaStream;
  let peerConnection: RTCPeerConnection;
  const ws = new WebSocket('ws://127.0.0.1:8000/ws/webrtc');
  
  // 初始化摄像头
  const initCamera = async () => {
    try {
      localStream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: 1280,
          height: 720,
          facingMode: 'environment' 
        }
      });
      
      if (videoElement.value) {
        videoElement.value.srcObject = localStream;
      }
      
      initWebRTC();
    } catch (error) {
      console.error('Camera access error:', error);
    }
  };
  
  // 初始化WebRTC连接
  const initWebRTC = () => {
    peerConnection = new RTCPeerConnection({
      // 局域网内，不需要STUN/TURN服务器
      // iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
    });
    
    // 添加本地视频轨道
    localStream.getTracks().forEach(track => {
      peerConnection.addTrack(track, localStream);
    });
  
    // ICE候选处理
    peerConnection.onicecandidate = ({ candidate }) => {
      console.log('ICE candidate:', candidate);
      if (candidate) {
        console.log('Sending candidate to server:', candidate);
        ws.send(JSON.stringify({
          type: 'candidate',
          candidate: candidate.toJSON()
        }));
      }
    };
    // ws.send(JSON.stringify({
    //   type: 'join',
    //   room: 'room1'
    // }));
    // 处理信令服务器消息
    ws.onmessage = async (event) => {
      const message = JSON.parse(event.data);
      console.log('Received message:', message);
      if (message.type === 'offer') {
        await peerConnection.setRemoteDescription(message);
        const answer = await peerConnection.createAnswer();
        await peerConnection.setLocalDescription(answer);
        ws.send(JSON.stringify(answer));
      }
    };
  };
  
  // 生命周期
  onMounted(() => {
    if (navigator.mediaDevices) {
      initCamera();
    } else {
      console.error('MediaDevices API not supported');
    }
  });
  
  onUnmounted(() => {
    localStream?.getTracks().forEach(track => track.stop());
    peerConnection?.close();
    ws.close();
  });
  </script>
  
  <style scoped>
  .video-container {
    width: 100%;
    height: 100%;
    position: relative;
  }
  
  video {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transform: scaleX(-1); /* 镜像翻转 */
  }
  </style>