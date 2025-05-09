import { createRouter, createWebHistory } from "@ionic/vue-router";
import { RouteRecordRaw } from "vue-router";

const routes: Array<RouteRecordRaw> = [
  // {
  //   path: '',
  //   redirect: '/folder/Inbox'
  // },
  {
    path: "",
    component: () => import("../views/CameraWebrtc.vue"),
  },
  {
    path: "/folder/CameraWebrtc",
    component: () => import("../views/CameraWebrtc.vue"),
  },
  {
    path: "/folder/:id",
    component: () => import("../views/FolderPage.vue"),
  },
];

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
});

export default router;
