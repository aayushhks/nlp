# CPU-only image for the scratchlm Gradio demo.
# Build from the repo root:
#   docker build -f deploy/Dockerfile -t scratchlm-demo .
FROM python:3.11-slim

WORKDIR /app

# CPU-only torch first, so the demo install below does not pull CUDA wheels.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY scratchlm ./scratchlm
COPY app ./app
COPY checkpoints ./checkpoints
RUN pip install --no-cache-dir ".[demo]"

ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False
EXPOSE 7860
CMD ["python", "app/demo.py"]
