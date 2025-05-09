<template>
    <ion-page>
      <ion-header :translucent="true">
        <ion-toolbar>
          <ion-buttons slot="start">
            <ion-menu-button color="primary"></ion-menu-button>
          </ion-buttons>
          <ion-title>{{ $route.params.id }}</ion-title>
        </ion-toolbar>
      </ion-header>
  
      <ion-content :fullscreen="true">
        <ion-header collapse="condense">
          <ion-toolbar>
            <ion-title size="large">{{ $route.params.id }}</ion-title>
          </ion-toolbar>
        </ion-header>
  
        <div class="camera-container">
        <!-- 拍照按钮 -->
        <button @click="takePicture">拍摄照片</button>
        
        <!-- 图片预览 -->
        <img v-if="imageUrl" :src="imageUrl" alt="拍摄的图片" class="preview-image">
        
        <!-- 加载状态 -->
        <div v-if="loading" class="loading">图片加载中...</div>
        
        <!-- 错误提示 -->
        <div v-if="error" class="error">{{ errorMessage }}</div>
    </div>

      </ion-content>
    </ion-page>
  </template>
  
  <script setup lang="ts">
  import { IonButtons, IonContent, IonHeader, IonMenuButton, IonPage, IonTitle, IonToolbar } from '@ionic/vue';
  import { ref } from 'vue';
import { Camera, CameraResultType } from '@capacitor/camera';

// 响应式数据
const imageUrl = ref<string | null>(null);
const loading = ref(false);
const error = ref(false);
const errorMessage = ref('');

// 拍照方法
const takePicture = async () => {
  try {
    loading.value = true;
    error.value = false;

    const image = await Camera.getPhoto({
      quality: 90,
      allowEditing: true,
      resultType: CameraResultType.Uri
    });

    if (image.webPath) {
      imageUrl.value = image.webPath;
    } else {
      throw new Error('无法获取图片路径');
    }
  } catch (err) {
    error.value = true;
    errorMessage.value = `拍照失败: ${(err as Error).message}`;
    console.error('Camera Error:', err);
  } finally {
    loading.value = false;
  }
};
  </script>
  
  <style scoped>
  #container {
    text-align: center;
    position: absolute;
    left: 0;
    right: 0;
    top: 50%;
    transform: translateY(-50%);
  }
  
  #container strong {
    font-size: 20px;
    line-height: 26px;
  }
  
  #container p {
    font-size: 16px;
    line-height: 22px;
    color: #8c8c8c;
    margin: 0;
  }
  
  #container a {
    text-decoration: none;
  }
  </style>
  