FROM nvcr.io/nvidia/pytorch:25.09-py3

# Minimal, ready-for-training Dockerfile stacked on NVIDIA NGC PyTorch image.
# NOTE: The NGC image bundles a compatible PyTorch and flash-attn. We explicitly
# avoid reinstalling `torch`, `torchvision` or `flash-attn` from PyPI to prevent
# breaking that bundled setup.

ENV PYTHONUNBUFFERED=1
WORKDIR /workspace

# Copy requirements and install Python deps but do NOT reinstall torch/torchvision
# or flash-attn from PyPI. This mirrors common patterns for NGC-based images.
COPY requirements.txt /workspace/requirements.txt
RUN /bin/bash -lc "\
    pip install --upgrade pip setuptools wheel && \
    if [ -f /workspace/requirements.txt ]; then \
        grep -v -E '^\s*torch(\b|=|<|>)' /workspace/requirements.txt | \
        grep -v -E '^\s*torchvision(\b|=|<|>)' | \
        grep -v -E '^\s*flash-attn(\b|=|<|>)' > /tmp/reqs_no_torch.txt || true; \
        pip install -r /tmp/reqs_no_torch.txt || true; \
    fi"

# Copy repo after dependencies to maximize build cache reuse
COPY . /workspace

# Provide mount points for dataset, results and huggingface cache
VOLUME ["/workspace/data", "/workspace/results", "/workspace/hf_cache"]

# Create a non-root user optionally (kept commented — enable if you want a non-root user)
# RUN useradd -m developer && chown -R developer:developer /workspace
# USER developer

# Note: We intentionally DO NOT set an ENTRYPOINT or default CMD here.
# The image is intended to be inert so a host-side launcher (e.g. run_train.sh)
# can invoke the container and pass the exact command to run inside.

# Default env vars (can be overridden at docker run time)
ENV DATA_DIR=/workspace/data
ENV RESULTS_DIR=/workspace/results
ENV HF_HOME=/workspace/hf_cache
