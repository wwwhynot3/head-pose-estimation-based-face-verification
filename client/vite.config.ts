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
      key: fs.readFileSync("/home/wwwhynot3/192.168.31.192-key.pem"),
      cert: fs.readFileSync("/home/wwwhynot3/192.168.31.192.pem"),
    },
    proxy: {
      '/account': {
        target: 'http'
      }
    }
  },
});
