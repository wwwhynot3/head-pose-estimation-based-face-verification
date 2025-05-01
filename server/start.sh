#!/usr/bin/env bash
sudo /home/wwwhynot3/workspace/head-pose-estimation-based-face-verification/server/.venv/bin/uvicorn server.asgi:application --host 0.0.0.0 --port 8000 --ssl-certfile "/home/wwwhynot3/workspace/head-pose-estimation-based-face-verification/192.168.31.192.pem" --ssl-keyfile "/home/wwwhynot3/workspace/head-pose-estimation-based-face-verification/192.168.31.192-key.pem"
# sudo /home/wwwhynot3/workspace/head-pose-estimation-based-face-verification/server/.venv/bin/uvicorn server.asgi:application --host 0.0.0.0 --port 8000 --ssl-certfile "$VITE_IP_CERT_PATH" --ssl-keyfile "$VITE_IP_KEY_PATH"

