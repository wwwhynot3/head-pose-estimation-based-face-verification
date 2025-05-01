#!/bin/bash

# 获取本机IP地址
get_ip() {
    local ip
    ip=$(hostname -I | awk '{print $1}')
    if [ -z "$ip" ]; then
        ip=$(ip route get 1.2.3.4 | awk '{print $7}' | head -1)
    fi
    echo "$ip"
}
IP=$(get_ip)
export VITE_IP_CERT_PATH="$(pwd)/$IP.pem"
export VITE_IP_KEY_PATH="$(pwd)/$IP-key.pem"
if [ ! -t "$IP_CERT_PATH" ] &&  [ ! -t "$IP_KEY_PATH" ]; then
    mkcert -install "$IP"
fi
echo "IP: $IP"
echo "IP_CERT_PATH: $VITE_IP_CERT_PATH"
echo "IP_KEY_PATH: $VITE_IP_KEY_PATH"
#export IP_CERT_PATH="$HOME/$IP.pem"
#export IP_KEY_PATH="$HOME/$IP-key.pem"


