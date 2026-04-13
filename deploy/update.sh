#!/usr/bin/env bash
# One-click deploy: pull latest main, rebuild frontend + API, restart.
# Usage (from local machine):
#   bash deploy/update.sh
# Or directly on the server:
#   bash /opt/globalbrine/deploy/update.sh --local
set -euo pipefail

REMOTE="globalbrine"
REPO_DIR="/opt/globalbrine"

# ---------------------------------------------------------------------------
# Detect mode: remote (SSH from local) or local (run on server)
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--local" ]]; then
    echo "=== Running locally on server ==="
    cd "$REPO_DIR"

    echo "[1/4] Pulling latest main..."
    sudo git fetch origin main
    sudo git reset --hard origin/main
    echo "  $(git log --oneline -1)"

    echo "[2/4] Building frontend..."
    cd web/app
    sudo npm ci --silent
    sudo npm run build 2>&1 | tail -3
    cd "$REPO_DIR"

    echo "[3/4] Rebuilding API container..."
    cd deploy
    sudo docker compose up -d --build 2>&1 | tail -5

    echo "[4/4] Verifying..."
    sleep 3
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
    if [[ "$STATUS" == "200" ]]; then
        echo "  API health check: OK"
    else
        echo "  WARNING: API health returned $STATUS"
    fi
    echo ""
    echo "=== Deploy complete! ==="
    echo "  $(curl -s http://localhost:8000/api/v1/model | python3 -c 'import sys,json; print(f"v{json.load(sys.stdin)[\"version\"]}")' 2>/dev/null || echo 'version unknown')"
else
    echo "=== Deploying to $REMOTE ($REPO_DIR) ==="
    ssh "$REMOTE" "bash $REPO_DIR/deploy/update.sh --local"
fi
