#!/usr/bin/env bash
# GlobalBrine deployment script for Tencent Cloud Ubuntu server
# Usage: ssh into server, clone repo, then run this script.
set -euo pipefail

echo "=== 1. Install Docker & Docker Compose ==="
if ! command -v docker &>/dev/null; then
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
    echo "Docker installed. You may need to re-login for group changes."
fi

echo "=== 2. Clone repo ==="
REPO_DIR="/opt/globalbrine"
if [ ! -d "$REPO_DIR" ]; then
    sudo mkdir -p "$REPO_DIR"
    sudo chown "$USER:$USER" "$REPO_DIR"
    git clone https://github.com/JiahaoZhang-Public/GlobalBrine-LithiumModel.git "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "=== 3. Build frontend ==="
if command -v node &>/dev/null; then
    cd web/app && npm ci && npm run build && cd ../..
else
    echo "Installing Node.js 20..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
    cd web/app && npm ci && npm run build && cd ../..
fi

echo "=== 4. Build and start services ==="
cd deploy
docker compose up -d --build

echo ""
echo "=== Deployment complete! ==="
echo "Visit: http://$(curl -s ifconfig.me)"
