#!/usr/bin/env bash
# EC2 user-data: paste into the launch wizard under
# "Advanced details > User data". Runs once on first boot, provisions the
# instance, and starts the demo as a systemd service on port 7860.
#
# Target AMI: Ubuntu Server 22.04 or 24.04 LTS (free-tier eligible).
set -euxo pipefail
exec > /var/log/scratchlm-userdata.log 2>&1

export REPO_URL="${REPO_URL:-https://github.com/aayushhks/transformer-autoregressive-lm-from-scratch.git}"
export BRANCH="${BRANCH:-main}"
export APP_DIR=/opt/scratchlm

apt-get update -y
apt-get install -y git
git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" "${APP_DIR}"

# Provision the environment (swap, python, venv, deps).
bash "${APP_DIR}/deploy/setup.sh"
chown -R ubuntu:ubuntu "${APP_DIR}"

# Install and start the service.
install -m 644 "${APP_DIR}/deploy/scratchlm-demo.service" /etc/systemd/system/scratchlm-demo.service
systemctl daemon-reload
systemctl enable --now scratchlm-demo
