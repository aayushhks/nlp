#!/usr/bin/env bash
# Provision an Ubuntu host (e.g. an AWS EC2 free-tier t3.micro / t2.micro) to
# run the scratchlm Gradio demo. Idempotent: safe to re-run.
#
#   REPO_URL=... BRANCH=main bash deploy/setup.sh
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/aayushhks/transformer-autoregressive-lm-from-scratch.git}"
BRANCH="${BRANCH:-main}"
APP_DIR="${APP_DIR:-/opt/scratchlm}"

echo "==> adding a 2G swap file (lets torch install/run on 1G instances)"
if [ ! -f /swapfile ]; then
  sudo fallocate -l 2G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

echo "==> installing system packages"
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip git

echo "==> fetching source into ${APP_DIR}"
sudo mkdir -p "${APP_DIR}"
sudo chown "$(id -u):$(id -g)" "${APP_DIR}"
if [ -d "${APP_DIR}/.git" ]; then
  git -C "${APP_DIR}" fetch --depth 1 origin "${BRANCH}"
  git -C "${APP_DIR}" checkout "${BRANCH}"
  git -C "${APP_DIR}" reset --hard "origin/${BRANCH}"
else
  git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" "${APP_DIR}"
fi

echo "==> creating virtualenv and installing (CPU-only torch keeps it small)"
python3 -m venv "${APP_DIR}/.venv"
"${APP_DIR}/.venv/bin/pip" install --upgrade pip
"${APP_DIR}/.venv/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu
"${APP_DIR}/.venv/bin/pip" install -e "${APP_DIR}[demo]"

echo "==> done. checkpoints present:"
ls -1 "${APP_DIR}/checkpoints/"*.pt 2>/dev/null || echo "  (none found — train one with scripts/train_lm.py)"
