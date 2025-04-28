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
  
      <ion-content :fullscreen="false">
        <div class="video-container">
          <video ref="remoteVideo" autoplay playsinline></video>
        </div>
        <div class="video-container">
          <video ref="localVideo" autoplay playsinline></video>
        </div>
      </ion-content>
      <ion-content :fullscreen="false">
        <div class="video-container">
          <video ref="localVideo" autoplay playsinline></video>
        </div>
      </ion-content>
    </ion-page>
  </template>
  
  <script setup lang="ts">
  import { ref, onMounted, onUnmounted } from 'vue';
  import { IonButtons, IonContent, IonHeader, IonMenuButton, IonPage, IonTitle, IonToolbar } from '@ionic/vue';
  
  const localVideo = ref<HTMLVideoElement>();
  const remoteVideo = ref<HTMLVideoElement>();
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
      
      if (localVideo.value) {
        localVideo.value.srcObject = localStream;
      }
      
      initWebRTC();
    } catch (error) {
      console.error('Camera access error:', error);
    }
  };
  
  // 初始化WebRTC连接
  const initWebRTC = () => {
    peerConnection = new RTCPeerConnection();
  
    // 添加本地视频轨道
    localStream.getTracks().forEach(track => {
      console.log('Adding local track:', track);
      peerConnection.addTrack(track, localStream);
    });
  
    // 监听远程轨道
    peerConnection.ontrack = (event) => {
      if (remoteVideo.value && event.streams[0]) {
        console.log('Received remote stream:' , event.streams[0]);
        remoteVideo.value.srcObject = event.streams[0];
      }
    };
    
    // ICE候选处理
    peerConnection.onicecandidate = ({ candidate }) => {
      // console.log('ICE candidate data:', candidate);
      if (candidate?.candidate) {

        ws.send(JSON.stringify({
          type: 'candidate',
          candidate: candidate.toJSON(),
        }));
      }
    };
    peerConnection.oniceconnectionstatechange = () => {
      console.log('ICE connection state:', peerConnection.iceConnectionState);
    };
    peerConnection.onconnectionstatechange = () => {
      console.log('Peer connection state:', peerConnection.connectionState);
    };
    // 处理信令服务器消息
    ws.onmessage = async (event) => {
      const message = JSON.parse(event.data);
      console.log('Received message:', message);
      if (message.type === 'answer') {
        // 设置远端描述
        await peerConnection.setRemoteDescription(new RTCSessionDescription(message));
      } 
      else if (message.type === 'candidate') {
        // 添加远端 ICE 候选
        await peerConnection.addIceCandidate(new RTCIceCandidate(message.candidate));
        // console.log('Received ICE candidate:', message.candidate);
      }else if(message.type === 'data_channel'){
        console.log('Received data channel:', message);
      }
    };
  
    // 创建并发送offer
    peerConnection.createOffer().then(offer => {
      peerConnection.setLocalDescription(offer)
        .then(() => {
          console.log('Sending offer:', offer);
          ws.send(JSON.stringify(offer));
        });
    });
  };
  
  onMounted(() => {
    if (navigator.mediaDevices) {
      initCamera();
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