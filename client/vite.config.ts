/// <reference types="vitest" />

import legacy from "@vitejs/plugin-legacy";
import vue from "@vitejs/plugin-vue";
import path from "path";
import { defineConfig } from "vite";
import fs from "fs";
// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue(), legacy()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  test: {
    globals: true,
    environment: "jsdom",
  },
  server: {
    https: {
      // key: fs.readFileSync(process.env.VITE_IP_KEY_PATH),
      // cert: fs.readFileSync(process.env.VITE_IP_CERT_PATH),
      key: fs.readFileSync(
        "/home/wwwhynot3/workspace/head-pose-estimation-based-face-verification/192.168.31.192-key.pem"
      ),
      cert: fs.readFileSync(
        "/home/wwwhynot3/workspace/head-pose-estimation-based-face-verification/192.168.31.192.pem"
      ),
    },
  },
});
