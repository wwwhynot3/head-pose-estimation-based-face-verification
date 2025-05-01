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
      <div class="modal">
        <div class="modal-header">
          <h3>通用设置</h3>
          <button
            class="close-button"
            @click.stop="showSourceSelection = false"
          >
            &times;
          </button>
        </div>
        <hr class="divider" />
        <div class="modal-header">
          <h4>账号设置</h4>
        </div>
        <div class="button-group horizontal">
          <button
            class="account-button"
            @click="currentUser ? logout() : login()"
          >
            {{ currentUser ? "🔓 登出" : "🔒 登陆" }}
          </button>
          <span class="user-info">
            {{ currentUser ? "用户名: " + currentUser : "未登录" }}
          </span>
        </div>

        <div class="button-group">
          <button @click="triggerFaceRegistration">📮注册人脸</button>
          <input
            ref="facebankFileInput"
            type="file"
            accept="image/*"
            style="display: none"
            @change="handleFacebankFileUpload"
          />
        </div>
        <hr class="divider" />
        <div class="modal-header">
          <h4>视频源设置</h4>
        </div>
        <div class="button-group">
          <button @click="selectSource('camera')">📷 本地相机</button>
          <button @click="selectSource('network')">🌐 网络视频源</button>
          <!-- 示例视频源 https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4 -->
          <button @click="selectSource('file')">🎞️ 本地视频文件</button>
          <button @click="triggerFaceRecognition()">🖼️ 本地图像文件</button>
          <input
            ref="facepictureFileInput"
            type="file"
            accept="image/*"
            style="display: none"
            @change="handleFacepictureFileUpload"
          />
        </div>
        <hr class="divider" />
        <div class="button-group horizontal">
          <label>
            <input type="checkbox" v-model="no_person_warning" />
            检测不到人脸是否警告
          </label>
          <div>
            <label for="warning-threshold">警告阈值 (秒):</label>
            <input
              id="warning-threshold"
              type="number"
              v-model="no_person_warning_timeout"
              min="1"
              style="width: 60px; margin-left: 5px"
            />
          </div>
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
    <!-- Login Modal -->
    <div
      v-if="showLoginModal"
      class="modal-overlay"
      @click.self="showLoginModal = false"
    >
      <div class="modal">
        <div class="modal-header">
          <h3>登录账号</h3>
          <button class="close-button" @click="showLoginModal = false">
            &times;
          </button>
        </div>
        <div class="form-group">
          <label for="serverAddress">服务器地址</label>
          <input
            id="serverAddress"
            v-model="serverAddress"
            type="text"
            placeholder="请输入服务器地址"
            @click.stop
          />
        </div>
        <div class="form-group">
          <label for="username">用户名</label>
          <input
            id="username"
            v-model="username"
            type="text"
            placeholder="请输入用户名"
            @click.stop
          />
        </div>
        <div class="form-group">
          <label for="password">密码</label>
          <input
            id="password"
            v-model="password"
            type="password"
            placeholder="请输入密码"
            @click.stop
          />
        </div>
        <div class="button-group">
          <button @click="handleLogin">登录</button>
        </div>
      </div>
    </div>
    <div
      v-if="showWarningModal"
      class="modal-overlay"
      @click.self="closeWarningModal"
    >
      <div class="modal">
        <div class="modal-header">
          <h3>警告</h3>
          <button class="close-button" @click="closeWarningModal">
            &times;
          </button>
        </div>
        <div class="modal-body">
          <p>{{ warningMessage }}</p>
        </div>
        <div class="button-group">
          <button @click="closeWarningModal">确定</button>
        </div>
      </div>
    </div>
  </ion-page>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from "vue";
import { IonContent, IonPage } from "@ionic/vue";
// import { CapacitorHttp, HttpResponse } from "@capacitor/core";
// import { Filesystem, Directory, Encoding } from "@capacitor/filesystem";

const localVideo = ref<HTMLVideoElement>();
const remoteVideo = ref<HTMLVideoElement>();
let localStream: MediaStream;
let peerConnection: RTCPeerConnection;
// const ws = new WebSocket("ws://127.0.0.1:8000/ws/webrtc");
let ws: WebSocket;
const no_person_warning = ref<boolean>(true);
let last_waring_timestamp = 0;
const no_person_warning_timeout = ref<number>(10);
const showWarningModal = ref(false);
const warningMessage = ref("");
// 控制弹窗显示
const showSourceSelection = ref<boolean>(true);
const isHovering = ref(false);
// 控制视频显示模式
const videoDisplayMode = ref<"localStream" | "remoteStream">("localStream");
const showCameraSelection = ref(false);
const videoDevices = ref<MediaDeviceInfo[]>([]);
const currentUser = ref<string | null>(null);
// Login modal fields
const showLoginModal = ref(false);
const logined = ref(false);
const serverAddress = ref("192.168.31.192:8000");
const username = ref("");
const password = ref("");
// 人脸注册
const facebankFileInput = ref<HTMLInputElement | null>(null);
const facepictureFileInput = ref<HTMLInputElement | null>(null);
let account: string;
const getWs = () => {
  return new WebSocket("wss://" + serverAddress.value + "/ws/webrtc");
};
const getAccountUrl = () => {
  return "https://" + serverAddress.value + "/account/";
};
const getMediaUrl = () => {
  return "https://" + serverAddress.value + "/media/";
};
const fetchRequest = async (
  url: string,
  options: {
    headers?: Record<string, string>;
    body?: any;
    method?: "POST";
  } = {}
) => {
  try {
    console.log("Request URL:", url);
    console.log("Request Options:", options);

    // 发起请求 ----------- capacitorjs
    // const response = await fetch(url, options);
    // const response = await CapacitorHttp.post({
    //   url: url,
    //   headers: options.headers,
    //   data: options.body,
    // });
    // // 检查响应状态
    // if (response.status !== 200) {
    //   const errorData = await response.data;
    //   console.error("Error Response:", errorData);
    //   throw new Error(errorData.message || `HTTP Error: ${response.status}`);
    // }
    // // 返回解析后的 JSON 数据
    // return await response.data;
    // 构建查询字符串（仅适用于 GET 或需要查询参数的请求）

    console.log("Request URL:", url);
    console.log("Request Options:", options);

    // 发起请求
    const response = await fetch(url, options);

    // 检查响应状态
    if (!response.ok) {
      const errorData = await response.json();
      console.error("Error Response:", errorData);
      throw new Error(errorData.message || `HTTP Error: ${response.status}`);
    }

    // 返回解析后的 JSON 数据
    return await response.json();
  } catch (error) {
    console.error("Fetch Error:", error);
    throw new Error(`网络错误: ${error}`);
  }
};
// 打开警告弹窗
const openWarningModal = (message: string) => {
  warningMessage.value = message;
  showWarningModal.value = true;
};

// 关闭警告弹窗
const closeWarningModal = () => {
  showWarningModal.value = false;
};

// 示例：替换 alert 的地方调用 openWarningModal
const handleRecognitionWarning = () => {
  openWarningModal("当前没有检测到人脸");
};
const handleLogin = async () => {
  /*
  if (!serverAddress.value || !username.value || !password.value) {
    alert("请填写所有字段！");
    return;
  }
    */
  if (!serverAddress.value) {
    alert("请填写服务器地址！");
    return;
  }
  account = username.value ? username.value : "default";
  // Simulate login process
  currentUser.value = account;
  const res = await fetchRequest(getAccountUrl() + "register", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      account: account,
    }),
  });
  if (!res) {
    alert("登录失败，请检查服务器地址或网络连接");
    return;
  } else {
    alert(`登录成功！欢迎 ${currentUser.value}`);
    showLoginModal.value = false;
    logined.value = true;
  }
};
const login = () => {
  if (currentUser.value) {
    alert("您已登录！");
  } else {
    showLoginModal.value = true;
  }
};
const logout = () => {
  if (confirm("确定要登出吗？")) {
    currentUser.value = null;
    // Clear fields
    serverAddress.value = "127.0.0.1:8000";
    username.value = "";
    password.value = "";
    location.reload();
  }
};
const triggerFaceRegistration = () => {
  console.log("triggerFaceRegistration");
  facebankFileInput.value?.click();
};
const triggerFaceRecognition = () => {
  console.log("triggerFaceRecognition");
  facepictureFileInput.value?.click();
};
const handleFacebankFileUpload = async (event: Event) => {
  console.log("handleFaceFileUpload");
  const file = (event.target as HTMLInputElement).files?.[0];
  console.log("file", file);
  if (!file) {
    alert("请选择图片文件！");
    return;
  }

  const formData = new FormData();
  formData.append("account", account);
  formData.append("file", file);

  try {
    const res = await fetchRequest(getAccountUrl() + "register_face", {
      method: "POST",
      body: formData,
    });
    if (res?.code === 200) {
      alert("人脸注册成功！");
    } else {
      alert(`人脸注册失败: ${res?.data || "未知错误"}`);
    }
  } catch (error) {
    console.error("Face registration error:", error);
    alert("人脸注册失败，请检查网络连接或服务器配置");
  }
};
const handleFacepictureFileUpload = async (event: Event) => {
  console.log("handleFacepictureFileUpload");
  const file = (event.target as HTMLInputElement).files?.[0];
  if (!file) {
    alert("请选择图片文件！");
    return;
  }

  const formData = new FormData();
  formData.append("account", account);
  formData.append("file", file);

  try {
    // const response: HttpResponse = await CapacitorHttp.post({
    //   url: getMediaUrl() + "picture",
    //   responseType: "blob",
    //   data: formData,
    // });
    // if (response.status === 200) {
    //   // 将响应转为 Blob
    //   const base64Data = btoa(
    //     new Uint8Array(response.data).reduce(
    //       (data, byte) => data + String.fromCharCode(byte),
    //       ""
    //     )
    //   );
    //   await Filesystem.writeFile({
    //     path: "after_" + file.name,
    //     data: base64Data,
    //     directory: Directory.Documents, // 可指定目录（如 Downloads、Cache）
    //     encoding: Encoding.UTF8,
    //   });
    //   const uri = await Filesystem.getUri({
    //     directory: Directory.ExternalStorage,
    //     path: "after_" + file.name,
    //   });
    //   console.log("File saved at:", uri);
    const response = await fetch(getMediaUrl() + "picture", {
      method: "POST",
      body: formData,
    });
    if (response.ok) {
      // 将响应转为 Blob
      const blob = await response.blob();
      // 创建临时链接并触发下载
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "processed_image.jpg"; // 指定下载文件名
      a.click();

      // 释放内存
      window.URL.revokeObjectURL(url);
    } else {
      alert("下载失败");
    }
  } catch (error) {
    console.error("Face registration error:", error);
    alert("识别失败，请检查网络连接或服务器配置");
  }
};
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
    console.log("Selected camera device ID:", deviceId);
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
    }
    // 更新 WebRTC 连接中的视频轨道
    const videoTrack = localStream.getVideoTracks()[0];
    if (peerConnection) {
      const sender = peerConnection
        .getSenders()
        .find((s) => s.track?.kind === "video");
      if (sender) {
        await sender.replaceTrack(videoTrack);
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
  try {
    await new Promise((resolve, reject) => {
      video.onloadedmetadata = resolve;
      video.onerror = reject;
      setTimeout(() => reject(new Error("视频加载超时")), 10000); // 10秒超时
    });
  } catch (error) {
    console.error("视频加载失败:", error);
    // 弹窗提示加载失败
    alert("视频加载失败，请检查视频源地址");
  }

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
    if (peerConnection.connectionState === "connected") {
      last_waring_timestamp = Date.now() / 1000;
    } else if (peerConnection.connectionState === "disconnected") {
      alert("WebRtc连接断开，请检查网络或重新登录");
      location.reload();
    }
  };
  // 处理信令服务器消息
  ws.onmessage = async (event) => {
    const message = JSON.parse(event.data);
    // console.log("Received message:", message);
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
    } else if (message.type === "recognition") {
      const timestamp: number = message.timestamp;
      const result: Array<any> = message.result;
      // const score: Number = message.score;
      if (no_person_warning.value) {
        if (result.length === 0) {
          if (
            timestamp - last_waring_timestamp >
            no_person_warning_timeout.value
          ) {
            handleRecognitionWarning();
            last_waring_timestamp = timestamp;
          }
        } else {
          last_waring_timestamp = timestamp;
        }
      } else {
        last_waring_timestamp = timestamp;
      }
    } else {
      console.warn("Unknown message type:", message.type);
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
    console.log("Received remote track:", event.track);
    if (remoteVideo.value && event.streams[0]) {
      remoteVideo.value.srcObject = event.streams[0];
    }
  };
};
onMounted(async () => {
  try {
    showLoginModal.value = true;
    await new Promise<void>((resolve) => {
      const unwatch = watch(showLoginModal, (value) => {
        if (!value) {
          resolve();
          unwatch(); // 停止监听
        }
      });
    });

    // 用户登录后初始化 WebSocket
    ws = getWs();
    ws.onopen = () => {
      console.log("WebSocket 已连接");
      console.log(`Going to send login account: ${account}`);
      ws.send(
        JSON.stringify({
          type: "login",
          account: account,
        })
      );
    };
    ws.onerror = (error) => {
      console.error("WebSocket 连接错误:", error);
      alert("WebSocket 连接错误，请检查服务器地址或网络连接");
      location.reload();
    };
    ws.onclose = () => {
      console.log("WebSocket 已关闭");
      alert("WebSocket 已关闭，请检查服务器地址或网络连接");
      location.reload();
    };
    console.log("OnMounted...");
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
/* Add styles for the login modal */
.form-group {
  margin-bottom: 1rem;
}

.form-group label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: bold;
}

.form-group input {
  width: 100%;
  padding: 0.5rem;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 1rem;
}
.user-info {
  margin-right: 0px;
  font-size: 1rem;
  text-decoration: underline;
  color: var(--ion-text-color, #333);
}

.divider {
  border: none;
  border-top: 1px solid #ccc;
  margin: 1rem 0;
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
.modal h3 {
  margin: 0;
  text-align: center;
  font-size: 1.2rem;
  color: var(--ion-text-color);
}
.modal-header {
  position: relative;
  display: flex;
  justify-content: center; /* 默认居中 */
  align-items: center;
  margin-bottom: 1rem;
  padding: 0 30px; /* 为关闭按钮留空间 */
}

.modal-header h3 {
  margin: 0;
  text-align: center; /* 居中对齐 */
  font-size: 1.5rem;
  color: var(--ion-text-color);
}

.modal-header h4 {
  margin: 0;
  text-align: left; /* 左对齐 */
  font-size: 1.2rem;
  color: var(--ion-text-color);
  width: 100%; /* 确保占满父容器宽度 */
  margin-left: -50px;
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
.button-group.horizontal {
  display: flex;
  justify-content: space-between; /* Space between elements */
  align-items: center;
  width: 100%; /* Ensure full width for alignment */
}

.account-button {
  flex: 0 0 38%; /* 确保按钮宽度为父容器的 61.8% */
}

.account-button:hover {
  background-color: var(--ion-color-primary-shade, #0056b3);
}

.user-info {
  font-size: 1rem;
  text-decoration: underline;
  color: var(--ion-text-color, #333);
  margin-left: auto; /* 将用户信息推到右侧 */
  text-align: right; /* 确保文字右对齐 */
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
