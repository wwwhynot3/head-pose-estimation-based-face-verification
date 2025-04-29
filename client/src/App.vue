<template>
  <ion-page>
    <ion-content :fullscreen="true">
      <div class="video-container">
        <button class="toggle-button" @click="toggleVideoDisplayMode">
          🔄
        </button>
        <!-- 视频容器添加 flex 居中 -->
        <div v-show="videoDisplayMode === 'localStream'">
          <video ref="localVideo" autoplay playsinline></video>
        </div>
        <div v-show="videoDisplayMode === 'remoteStream'">
          <video ref="remoteVideo" autoplay playsinline></video>
        </div>
      </div>
    </ion-content>
    <!-- 修改后的设置按钮 -->
    <div
      class="settings-container"
      @mouseover="isHovering = true"
      @mouseleave="isHovering = false"
    >
      <button class="settings-button" @click.stop="showSourceSelection = true">
        <!-- 添加.stop修饰符 -->
        <span class="icon">⚙️</span>
        <span class="text">视频源设置</span>
      </button>
    </div>

    <div
      v-if="showSourceSelection"
      class="modal-overlay"
      @click.self="showSourceSelection = false"
    >
      <!-- 修改后的modal部分 -->
      <div class="modal">
        <div class="modal-header">
          <h3>选择视频源</h3>
          <button
            class="close-button"
            @click.stop="showSourceSelection = false"
          >
            &times;
          </button>
        </div>
        <div class="button-group">
          <button @click="selectSource('camera')">📷 本地相机</button>
          <button @click="selectSource('network')">🌐 网络视频源</button>
          <button @click="selectSource('file')">📁 本地文件</button>
        </div>
      </div>
    </div>
    <div
      v-if="showCameraSelection"
      class="modal-overlay"
      @click.self="showCameraSelection = false"
    >
      <div class="modal">
        <div class="modal-header">
          <h3>选择摄像头</h3>
          <button
            class="close-button"
            @click.stop="showCameraSelection = false"
          >
            &times;
          </button>
        </div>
        <div class="button-group">
          <button
            v-for="device in videoDevices"
            :key="device.deviceId"
            @click="selectCamera(device.deviceId)"
          >
            {{ device.label || `摄像头 ${device.deviceId + 1}` }}
          </button>
        </div>
      </div>
    </div>
  </ion-page>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from "vue";
import { IonContent, IonPage } from "@ionic/vue";

const localVideo = ref<HTMLVideoElement>();
const remoteVideo = ref<HTMLVideoElement>();
let localStream: MediaStream;
let peerConnection: RTCPeerConnection;
const ws = new WebSocket("ws://127.0.0.1:8000/ws/webrtc");
// 控制弹窗显示
const showSourceSelection = ref(true);
const isHovering = ref(false);
// 控制视频显示模式
const videoDisplayMode = ref<"localStream" | "remoteStream">("localStream");
const showCameraSelection = ref(false);
const videoDevices = ref<MediaDeviceInfo[]>([]);

// 获取视频设备列表
const getVideoDevices = async () => {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    videoDevices.value = devices.filter(
      (device) => device.kind === "videoinput"
    );
    // 自动选择第一个摄像头（如果只有一个）
    if (videoDevices.value.length === 1) {
      selectCamera(videoDevices.value[0].deviceId);
    }
  } catch (error) {
    console.error("获取摄像头列表失败:", error);
  }
};

// 切换视频显示模式
const toggleVideoDisplayMode = () => {
  if (videoDisplayMode.value === "localStream") {
    videoDisplayMode.value = "remoteStream";
  } else {
    videoDisplayMode.value = "localStream";
  }
};
const selectSource = async (sourceType: "camera" | "network" | "file") => {
  try {
    if (sourceType === "camera") {
      // await switchVideoSource("camera");
      await getVideoDevices();
      if (videoDevices.value.length > 1) {
        showCameraSelection.value = true;
      }
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
// 新增摄像头选择方法
const selectCamera = async (deviceId: string) => {
  try {
    showCameraSelection.value = false;
    await switchVideoSource("camera", deviceId);
  } catch (error) {
    console.error("切换摄像头失败:", error);
    alert("无法切换摄像头，请检查设备权限");
  }
};
const switchVideoSource = async (
  sourceType: "camera" | "network" | "file",
  source?: string
) => {
  try {
    console.log("Switching video source to:", sourceType, source);
    // 停止当前的本地流
    if (localStream) {
      localStream.getTracks().forEach((track) => track.stop());
    }
    if (sourceType === "camera") {
      const constraints: MediaStreamConstraints = {
        video: source
          ? {
              deviceId: { exact: source },
              width: {
                min: 1280,
                ideal: 1920,
                max: 2560,
              },
              height: {
                min: 720,
                ideal: 1080,
                max: 1440,
              },
            }
          : true,
        audio: false,
      };
      localStream = await navigator.mediaDevices.getUserMedia(constraints);
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
      // 在加载网络视频源的尝试
      // // 等待视频元数据加载完成
      // await new Promise<void>((resolve, reject) => {
      //   if (!localVideo.value) return reject("Video element not found");

      //   localVideo.value.onloadedmetadata = () => resolve();
      //   localVideo.value.onerror = (err) => reject(err);

      //   // 设置超时防止卡死
      //   setTimeout(() => reject("视频元数据加载超时"), 10000);
      // });

      // // 尝试播放视频
      // await localVideo.value.play();
    }
    // 更新 WebRTC 连接中的视频轨道
    const videoTrack = localStream.getVideoTracks()[0];
    if (peerConnection) {
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
  // 暂时无法解决跨域问题，以下为尝试
  // console.log("Fetching network video stream from:", url);

  // const video = document.createElement("video");
  // // 添加到DOM（即使隐藏）
  // video.style.position = "fixed";
  // video.style.opacity = "0";
  // video.style.pointerEvents = "none";
  // video.style.top = "-1000px";
  // document.body.appendChild(video);

  // try {
  //   video.src = url;
  //   video.crossOrigin = "anonymous";
  //   video.muted = true; // 解决自动播放限制
  //   video.preload = "auto";

  //   // 等待元数据加载
  //   await new Promise((resolve, reject) => {
  //     video.onloadedmetadata = resolve;
  //     video.onerror = reject;
  //     setTimeout(() => reject(new Error("视频加载超时")), 10000); // 10秒超时
  //   });
  //   console.log("视频元数据加载成功", url);
  //   // 尝试播放
  //   await video.play();

  //   // 等待视频实际开始播放
  //   await new Promise((resolve, reject) => {
  //     const checkPlay = () => {
  //       if (!video.paused) return resolve(true);
  //       setTimeout(checkPlay, 100);
  //     };
  //     checkPlay();
  //     setTimeout(() => reject(new Error("播放未能启动")), 5000);
  //   });
  //   console.log("视频播放成功", url);
  //   const stream = (video as any).captureStream();
  //   if (!stream) {
  //     throw new Error("浏览器不支持captureStream");
  //   }
  //   console.log("视频流捕获成功", url);
  //   return stream;
  // } catch (err) {
  //   console.log("ERROR WHEN LOADING URL", url);
  //   throw err;
  // } finally {
  //   // 清理视频元素
  //   video.pause();
  //   video.removeAttribute("src");
  //   video.load();
  //   document.body.removeChild(video);
  // }
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
  if (!localStream) {
    console.error("Local stream not initialized");
    return;
  }
  peerConnection = new RTCPeerConnection();

  // 添加本地视频轨道
  localStream.getTracks().forEach((track) => {
    if (track.kind === "video") {
      console.log("Adding local track:", track);
      peerConnection.addTrack(track, localStream);
    } else {
      console.warn("Unsupported track kind:", track.kind);
    }
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
onMounted(async () => {
  try {
    // 先获取基础流以激活设备枚举
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    stream.getTracks().forEach((track) => track.stop());
    await getVideoDevices();
  } catch (error) {
    console.error("初始化摄像头失败:", error);
  }
});

onUnmounted(() => {
  localStream?.getTracks().forEach((track) => track.stop());
  peerConnection?.close();
  ws.close();
});
</script>

<style scoped>
/* 修改视频容器样式 */
.video-container {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  background: #000; /* 黑边颜色 */
}

/* 修改视频元素样式 */
video {
  /* 核心修改点 */
  object-fit: contain; /* 替换原来的cover */

  /* 动态尺寸控制 */
  max-width: 100%;
  max-height: 100%;
  width: auto;
  height: auto;

  /* 居中显示 */
  margin: auto;
}

/* 确保ion-content无内边距 */
ion-content {
  --padding-start: 0;
  --padding-end: 0;
  --padding-top: 0;
  --padding-bottom: 0;
}

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

.settings-container {
  position: fixed;
  bottom: 20px;
  left: 0;
  z-index: 1000;
  transition: all 0.3s ease;
}

.settings-button {
  display: flex;
  align-items: center;
  background: rgba(var(--ion-color-primary-rgb), 0.9);
  color: white;
  border: none;
  border-radius: 0 15px 15px 0;
  padding: 8px 15px;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
  overflow: hidden;
  white-space: nowrap;
  width: auto;
  max-width: 200px;
  height: 30px;
}

.settings-container:hover .settings-button {
  padding-right: 20px;
  background: var(--ion-color-primary);
}

.settings-button .text {
  opacity: 0;
  max-width: 0;
  transition: all 0.3s ease;
  margin-left: 8px;
}

.settings-container:hover .text {
  opacity: 1;
  max-width: 200px;
}

.settings-button .icon {
  font-size: 16px;
  transition: transform 0.3s ease;
}

.settings-container:hover .icon {
  transform: rotate(180deg);
}

.settings-button:not(:hover) {
  width: 5px;
  padding: 8px 5px;
  background: rgba(var(--ion-color-primary-rgb), 0.5);
}

.settings-button:not(:hover)::after {
  content: "";
  position: absolute;
  left: 0;
  top: 0;
  width: 30px;
  height: 100%;
}

.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.modal {
  background: var(--ion-background-color, #f0f0f0);
  padding: 1.2rem;
  border-radius: 8px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  width: 90%;
  max-width: 400px;
  position: relative;
}

.modal-header {
  position: relative;
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: 1rem;
  padding: 0 30px; /* 为关闭按钮留空间 */
}

.modal h3 {
  margin: 0;
  text-align: center;
  font-size: 1.2rem;
  color: var(--ion-text-color);
}

.button-group button {
  display: block;
  width: 100%;
  margin: 10px 0;
  padding: 10px 20px;
  font-size: 16px;
  border: none;
  border-radius: 4px;
  background-color: var(--ion-color-primary, #007bff);
  color: var(--ion-color-light, #fff);
  cursor: pointer;
  transition: background-color 0.3s ease;
}

.button-group button:hover {
  background-color: var(--ion-color-primary-shade, #0056b3);
}

.close-button {
  background: none !important;
  border: none;
  font-size: 1.5rem;
  line-height: 1;
  cursor: pointer;
  color: #666;
  padding: 0 0 0 1rem;
  transition: color 0.3s ease;
  margin-top: -2px;
  position: absolute;
  right: 0%;
}

.close-button:hover {
  color: #ff0000;
}
</style>
