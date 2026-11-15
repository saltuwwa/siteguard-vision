FROM python:3.11-slim-bookworm
ARG INSTALL_ML=false
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 YOLO_CONFIG_DIR=/tmp/ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml /app/
COPY README.md LICENSE /app/
COPY src /app/src
RUN if [ "$INSTALL_ML" = true ]; then \
      pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 --index-url "$TORCH_INDEX" \
      && pip install --no-cache-dir '.[ml]'; \
    else pip install --no-cache-dir .; fi
RUN useradd --uid 10001 --create-home siteguard && mkdir -p /app/runs && chown siteguard /app/runs
COPY configs /app/configs
USER siteguard
EXPOSE 8000
CMD ["siteguard", "serve", "--host", "0.0.0.0"]
