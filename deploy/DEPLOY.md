# Deploying the demo on AWS (free tier)

This runs the Gradio demo 24/7 on a single **EC2 free-tier instance**. The free
tier covers **750 instance-hours/month for 12 months** — enough for one instance
running continuously for a year. After 12 months a `t3.micro` is roughly
**$7–9/month**, so set a billing alarm and tear it down when you're done
(see [Teardown](#teardown)).

> Want free *forever* for this exact Gradio app? Hugging Face Spaces hosts it at
> no cost with no time limit — see [`app/README.md`](../app/README.md). The steps
> below are the AWS path.

## Architecture

A single Ubuntu EC2 instance runs the app as a **systemd service** behind the
instance's public IP on port **7860**. CPU-only PyTorch keeps the footprint
small, and a 2 GB swap file covers the 1 GB RAM of a micro instance. The tiny
checkpoints are committed to the repo, so the box is self-contained — no
training required.

## 1. Launch the instance

1. EC2 → **Launch instance**.
2. **AMI:** Ubuntu Server 24.04 LTS (or 22.04) — "Free tier eligible".
3. **Instance type:** `t3.micro` (or `t2.micro` — whichever shows "Free tier
   eligible" in your region).
4. **Key pair:** create/select one so you can SSH in.
5. **Network → Security group**, add inbound rules:
   - SSH, TCP **22**, Source **My IP**
   - Custom TCP, port **7860**, Source **Anywhere (0.0.0.0/0)** — the demo
6. **Storage:** 8–30 GB gp3 (the free tier includes 30 GB).
7. (Optional, fully automated) expand **Advanced details → User data** and paste
   the contents of [`deploy/user-data.sh`](user-data.sh). The instance will
   provision itself and the demo will be live a few minutes after boot — skip
   straight to [step 3](#3-open-the-demo).
8. **Launch instance.**

## 2. Provision manually (skip if you used user-data)

SSH in and run the setup script:

```bash
ssh -i your-key.pem ubuntu@<PUBLIC_IP>

curl -fsSL https://raw.githubusercontent.com/aayushhks/transformer-autoregressive-lm-from-scratch/main/deploy/setup.sh -o setup.sh
bash setup.sh

# install + start the service
sudo install -m 644 /opt/scratchlm/deploy/scratchlm-demo.service /etc/systemd/system/scratchlm-demo.service
sudo systemctl daemon-reload
sudo systemctl enable --now scratchlm-demo
```

## 3. Open the demo

```
http://<PUBLIC_IP>:7860
```

Generation streams token-by-token; the "Compare strategies" tab shows greedy vs
temperature vs nucleus side by side.

## Managing the service

```bash
sudo systemctl status scratchlm-demo      # is it running?
sudo journalctl -u scratchlm-demo -f      # live logs
sudo systemctl restart scratchlm-demo     # restart

# update to the latest code, then restart
cd /opt/scratchlm && git pull && sudo systemctl restart scratchlm-demo
```

## Alternative: Docker

If you'd rather run a container (install Docker first):

```bash
cd /opt/scratchlm
docker compose -f deploy/docker-compose.yml up -d --build
```

`restart: unless-stopped` brings it back after reboots. The image is CPU-only
and bundles the checkpoints.

## Keep it cheap

- **Set a billing alarm** (Billing → Budgets) for ~$1 so you're warned if you
  leave the free tier.
- Stay on a **single micro instance** — that is what the 750 free hours cover.
- A static **Elastic IP** is free *only while attached to a running instance*;
  release it if you stop the instance long-term.

## Teardown

```
EC2 → Instances → select → Instance state → Terminate
```

Then delete the security group and release any Elastic IP. Terminating stops all
charges for the instance.
