# StockAgent — Multi-stage Docker build
# =======================================
# Stage 1: builder — install dependencies into a virtualenv
# Stage 2: runtime — copy venv + app code, run as non-root
#
# Usage:
#   docker build -t stockagent .
#   docker build --build-arg EXTRAS="viz" -t stockagent .   # include matplotlib/numpy
#   docker run --rm stockagent status
#   docker run --rm -v $(pwd)/data:/app/data stockagent analyze AAPL

ARG EXTRAS=""

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ARG EXTRAS

WORKDIR /build

# Install build dependencies (needed for pandas/ta C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first for layer caching
COPY pyproject.toml requirements.txt ./

# Create virtualenv and install dependencies
# EXTRAS="" → core only; EXTRAS="viz" → include matplotlib/numpy
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    if [ -n "$EXTRAS" ]; then \
        /opt/venv/bin/pip install -e ".[$EXTRAS]" --no-cache-dir; \
    else \
        /opt/venv/bin/pip install -e "." --no-cache-dir; \
    fi

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv

# Make venv the default Python
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create non-root user
RUN useradd --create-home --shell /bin/bash stockagent

# Copy application code (respects .dockerignore)
COPY --chown=stockagent:stockagent . .

# Create persistent data directory and set ownership
RUN mkdir -p /app/data && chown -R stockagent:stockagent /app/data

# Switch to non-root user
USER stockagent

# Expose data directory as volume for persistent SQLite databases
VOLUME ["/app/data"]

# Default command: show status
# Override at runtime: docker run stockagent analyze AAPL
ENTRYPOINT ["python", "cli.py"]
CMD ["status"]
