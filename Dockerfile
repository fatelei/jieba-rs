# Multi-stage build for Rust jieba
FROM rust:1.75-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python build tools
RUN pip3 install --no-cache-dir maturin

WORKDIR /app

# Copy Cargo files first for better caching
COPY Cargo.toml Cargo.lock ./
COPY src ./src

# Build the library in release mode
RUN cargo build --release

# Python wheel stage
FROM python:3.11-slim AS python-wheels

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Install maturin
RUN pip install --no-cache-dir maturin

WORKDIR /app

# Copy source code (exclude large data files for faster build)
COPY Cargo.toml Cargo.lock pyproject.toml ./
COPY src ./src
COPY examples/docker_test.py ./examples/
COPY README.md ./

# Build Python wheels for multiple platforms
RUN maturin build --release --out dist --strip

# Final runtime image
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install jieba for comparison
RUN pip install --no-cache-dir jieba

# Create app user
RUN useradd --create-home --shell /bin/bash app

WORKDIR /home/app

# Copy built wheels from builder stage
COPY --from=python-wheels /app/dist/*.whl ./

# Install Rust jieba
RUN pip install --no-cache-dir rust_jieba*.whl

# Copy test scripts
COPY examples/docker_test.py ./

# Change ownership to app user
RUN chown -R app:app /home/app

USER app

# Set environment variables
ENV PYTHONPATH=/home/app
ENV RUST_LOG=info

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import rust_jieba; print('Health check passed')" || exit 1

# Default command - run the test script
CMD ["python", "docker_test.py"]