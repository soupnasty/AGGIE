# AGGIE Voice Assistant - Docker Image
#
# Build:  docker build -t aggie .
# Run:    docker compose up

# Note: Using Python 3.11 because tflite-runtime (required by openwakeword)
# doesn't support Python 3.12 yet
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    portaudio19-dev \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
WORKDIR /build
COPY pyproject.toml .
COPY src/ src/

# Install the package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Download wake word models during build and verify
RUN python -c "import openwakeword; openwakeword.utils.download_models(); print('Models downloaded')" && \
    echo "=== Models in package ===" && \
    find /opt/venv/lib/python3.11/site-packages/openwakeword -name "*.tflite" -o -name "*.onnx" | head -20 && \
    echo "=== Resources dir ===" && \
    ls -la /opt/venv/lib/python3.11/site-packages/openwakeword/resources/models/ || echo "No models dir"


# --- Runtime image ---
FROM python:3.11-slim

# Install runtime dependencies including PulseAudio support
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    libsndfile1 \
    pulseaudio-utils \
    libpulse0 \
    libasound2-plugins \
    && rm -rf /var/lib/apt/lists/*

# Configure ALSA to use PulseAudio
RUN echo "pcm.!default { type pulse }" > /etc/asound.conf && \
    echo "ctl.!default { type pulse }" >> /etc/asound.conf

# Copy virtual environment from builder (includes downloaded models)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Make openwakeword models directory writable (for runtime downloads if needed)
RUN chmod -R 777 /opt/venv/lib/python3.11/site-packages/openwakeword/resources/models/ || true

# Create non-root user
RUN useradd -m -s /bin/bash aggie
USER aggie
WORKDIR /home/aggie

# Create directories for config and model cache
RUN mkdir -p /home/aggie/.config/aggie \
             /home/aggie/.cache \
             /home/aggie/.local/share/piper-voices

# Copy default config
COPY --chown=aggie:aggie config/aggie.yaml.example /home/aggie/.config/aggie/config.yaml

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV XDG_CONFIG_HOME=/home/aggie/.config
ENV XDG_CACHE_HOME=/home/aggie/.cache

# IPC socket location (inside container)
ENV AGGIE_SOCKET_PATH=/tmp/aggie.sock

# Entry point
ENTRYPOINT ["aggie"]
CMD ["--debug"]
