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

    <!-- <ion-content :fullscreen="false">
      <div class="video-container">
        <video ref="remoteVideo" autoplay playsinline></video>
      </div>
    </ion-content>
    <ion-content :fullscreen="false">
      <div class="video-container">
        <video ref="localVideo" autoplay playsinline></video>
      </div>
    </ion-content> -->
    <!-- <ion-content :fullscreen="false">
      <div class="video-container">
        <button class="toggle-button" @click="toggleVideoDisplayMode">
          ChangeView
        </button>
        <div v-if="videoDisplayMode === 'localStream'">
          <video ref="localVideo" autoplay playsinline></video>
        </div>
        <div v-else-if="videoDisplayMode === 'remoteStream'">
          <video ref="remoteVideo" autoplay playsinline></video>
        </div>
        <div v-else-if="videoDisplayMode === 'bothStream'" class="both-streams">
          <video ref="localVideo" autoplay playsinline></video>
          <video ref="remoteVideo" autoplay playsinline></video>
        </div>
      </div>
    </ion-content> -->
    <ion-content :fullscreen="false">
      <div class="video-container">
        <button class="toggle-button" @click="toggleVideoDisplayMode">
          ChangeView
        </button>
        <div
          v-show="
            videoDisplayMode === 'localStream' ||
            videoDisplayMode === 'bothStream'
          "
          :class="{ 'both-streams': videoDisplayMode === 'bothStream' }"
        >
          <video ref="localVideo" autoplay playsinline></video>
        </div>
        <div
          v-show="
            videoDisplayMode === 'remoteStream' ||
            videoDisplayMode === 'bothStream'
          "
          :class="{ 'both-streams': videoDisplayMode === 'bothStream' }"
        >
          <video ref="remoteVideo" autoplay playsinline></video>
        </div>
      </div>
    </ion-content>
    <div v-if="showSourceSelection" class="modal-overlay">
      <div class="modal">
        <h3>Select Video Source</h3>
        <button @click="selectSource('camera')">Camera</button>
        <button @click="selectSource('network')">Network Video</button>
        <button @click="selectSource('file')">Local File</button>
      </div>
    </div>
  </ion-page>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from "vue";
import {
  IonButtons,
  IonContent,
  IonHeader,
  IonMenuButton,
  IonPage,
  IonTitle,
  IonToolbar,
} from "@ionic/vue";

const localVideo = ref<HTMLVideoElement>();
const remoteVideo = ref<HTMLVideoElement>();
let localStream: MediaStream;
let peerConnection: RTCPeerConnection;
const ws = new WebSocket("ws://127.0.0.1:8000/ws/webrtc");
// 控制弹窗显示
const showSourceSelection = ref(true);
// 控制视频显示模式
const videoDisplayMode = ref<"localStream" | "remoteStream" | "bothStream">(
  "localStream"
);

// 切换视频显示模式
const toggleVideoDisplayMode = () => {
  if (videoDisplayMode.value === "localStream") {
    videoDisplayMode.value = "remoteStream";
  } else if (videoDisplayMode.value === "remoteStream") {
    videoDisplayMode.value = "bothStream";
  } else {
    videoDisplayMode.value = "localStream";
  }
};
const selectSource = async (sourceType: "camera" | "network" | "file") => {
  try {
    if (sourceType === "camera") {
      await switchVideoSource("camera");
    } else if (sourceType === "network") {
      const url = prompt("Enter the network video URL:");
      if (url) {
        await switchVideoSource("network", url);
      }
    } else if (sourceType === "file") {
      const fileInput = document.createElement("input");
      fileInput.type = "file";
      fileInput.accept = "video/*";
      fileInput.onchange = async (event: Event) => {
        const file = (event.target as HTMLInputElement).files?.[0];
        if (file) {
          const fileURL = URL.createObjectURL(file);
          await switchVideoSource("file", fileURL);
        }
      };
      fileInput.click();
    }
    showSourceSelection.value = false; // 关闭弹窗
  } catch (error) {
    console.error("Error selecting video source:", error);
  }
};
const switchVideoSource = async (
  sourceType: "camera" | "network" | "file",
  source?: string
) => {
  try {
    // 停止当前的本地流
    if (localStream) {
      localStream.getTracks().forEach((track) => track.stop());
    }
    if (sourceType === "camera") {
      // 使用摄像头作为视频源
      localStream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
    } else if (sourceType === "network" && source) {
      // 使用网络视频源
      localStream = await fetchNetworkStream(source);
    } else if (sourceType === "file" && source) {
      // 使用本地视频文件
      localStream = await fetchFileStream(source);
    } else {
      throw new Error("Invalid source type or missing source URL");
    }

    // 将本地流绑定到视频元素
    if (localVideo.value) {
      localVideo.value.srcObject = localStream;
    }

    // 更新 WebRTC 连接中的视频轨道
    const videoTrack = localStream.getVideoTracks()[0];
    if (peerConnection) {
      // const senders = peerConnection.getSenders();
      // const videoSender = senders.find(
      //   (sender) => sender.track?.kind === "video"
      // );
      // if (videoSender) {
      //   videoSender.replaceTrack(videoTrack);
      // } else {
      //   peerConnection.addTrack(videoTrack, localStream);
      // }
      const sender = peerConnection
        .getSenders()
        .find((s) => s.track?.kind === "video");
      if (sender) {
        await sender.replaceTrack(videoTrack);
        // 触发重新协商
        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);
        ws.send(JSON.stringify(offer));
      }
    } else {
      // 如果没有现有的连接，则初始化新的连接
      initWebRTC();
    }

    console.log(`Switched video source to: ${sourceType}`);
  } catch (error) {
    console.error("Error switching video source:", error);
  }
};

// Helper function to fetch network video stream
const fetchNetworkStream = async (url: string): Promise<MediaStream> => {
  const video = document.createElement("video");
  video.src = url;
  video.crossOrigin = "anonymous";
  await video.play();

  const stream = (
    video as HTMLVideoElement & { captureStream?: () => MediaStream }
  ).captureStream?.();
  if (!stream) {
    throw new Error("captureStream is not supported in this browser.");
  }
  return stream;
};

// Helper function to fetch local file video stream
const fetchFileStream = async (filePath: string): Promise<MediaStream> => {
  const video = document.createElement("video");
  video.src = filePath;
  await video.play();
  const stream = (
    video as HTMLVideoElement & { captureStream?: () => MediaStream }
  ).captureStream?.();
  if (!stream) {
    throw new Error("Failed to capture stream from video element.");
  }
  return stream;
};

// 初始化WebRTC连接
const initWebRTC = () => {
  peerConnection = new RTCPeerConnection();

  // 添加本地视频轨道
  localStream.getTracks().forEach((track) => {
    console.log("Adding local track:", track);
    peerConnection.addTrack(track, localStream);
  });

  // ICE候选处理
  peerConnection.onicecandidate = ({ candidate }) => {
    // console.log('ICE candidate data:', candidate);
    if (candidate?.candidate) {
      ws.send(
        JSON.stringify({
          type: "candidate",
          candidate: candidate.toJSON(),
        })
      );
    }
  };
  peerConnection.oniceconnectionstatechange = () => {
    console.log("ICE connection state:", peerConnection.iceConnectionState);
  };
  peerConnection.onconnectionstatechange = () => {
    console.log("Peer connection state:", peerConnection.connectionState);
  };
  // 处理信令服务器消息
  ws.onmessage = async (event) => {
    const message = JSON.parse(event.data);
    console.log("Received message:", message);
    if (message.type === "answer") {
      // 设置远端描述
      await peerConnection.setRemoteDescription(
        new RTCSessionDescription(message)
      );
    } else if (message.type === "candidate") {
      // 添加远端 ICE 候选
      await peerConnection.addIceCandidate(
        new RTCIceCandidate(message.candidate)
      );
      // console.log('Received ICE candidate:', message.candidate);
    } else if (message.type === "data_channel") {
      console.log("Received data channel:", message);
    }
  };

  // 创建并发送offer
  peerConnection.createOffer().then((offer) => {
    peerConnection.setLocalDescription(offer).then(() => {
      console.log("Sending offer:", offer);
      ws.send(JSON.stringify(offer));
    });
  });

  // 监听远程轨道
  peerConnection.ontrack = (event) => {
    if (remoteVideo.value && event.streams[0]) {
      remoteVideo.value.srcObject = event.streams[0];
    }
  };
};

onMounted(() => {
  if (navigator.mediaDevices) {
    // switchVideoSource("camera").then(() => {
    //   initWebRTC();
    // });
  }
});

onUnmounted(() => {
  localStream?.getTracks().forEach((track) => track.stop());
  peerConnection?.close();
  ws.close();
});
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.5); /* 半透明背景 */
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.modal {
  background: var(--ion-background-color, #f0f0f0); /* 使用系统背景颜色 */
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  text-align: center;
  width: 90%; /* 弹窗宽度适配 */
  max-width: 400px; /* 最大宽度 */
}

.modal h3 {
  margin-bottom: 20px;
  color: var(--ion-text-color, #000); /* 提示文字为对比色 */
}

.modal button {
  display: block;
  width: 100%; /* 按钮宽度一致 */
  margin: 10px 0;
  padding: 10px 20px;
  font-size: 16px;
  text-align: center; /* 字体居中 */
  cursor: pointer;
  border: none;
  border-radius: 4px;
  background-color: var(--ion-color-primary, #007bff); /* 按钮背景色 */
  color: var(--ion-color-light, #fff); /* 按钮文字颜色 */
  transition: background-color 0.3s ease;
}

.modal button:hover {
  background-color: var(--ion-color-primary-shade, #0056b3); /* 按钮悬停颜色 */
}

.modal button:active {
  background-color: var(--ion-color-primary-tint, #3399ff); /* 按钮按下颜色 */
}
.video-container {
  width: 100vw;
  height: 100vh; /* 确保容器有明确尺寸 */
  overflow: hidden; /* 隐藏溢出内容 */
  position: relative;
}

.both-streams {
  display: flex;
  height: 100%;
  width: 100%;
  gap: 4px; /* 视频间距 */
}

.both-streams video {
  flex: 1; /* 平分宽度 */
  min-width: 0; /* 修复 flex 溢出问题 */
  height: 100%;
  transform: scaleX(-1); /* 保持镜像翻转 */
  object-fit: cover; /* 保持视频比例 */
}

/* .video-container {
  width: 100%;
  height: 100%;
  position: relative;
} */

video {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transform: scaleX(-1); /* 镜像翻转 */
}
/* .both-streams {
  display: flex;
  justify-content: space-between;
}

.both-streams video {
  width: 49%; 
} */

.toggle-button {
  position: absolute;
  top: 10px;
  right: 10px;
  z-index: 1001;
  background-color: var(--ion-color-primary, #007bff);
  color: var(--ion-color-light, #fff);
  border: none;
  border-radius: 4px;
  padding: 10px 15px;
  cursor: pointer;
  font-size: 14px;
}

.toggle-button:hover {
  background-color: var(--ion-color-primary-shade, #0056b3);
}
</style>
