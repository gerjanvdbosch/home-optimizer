# ---------------------------------------------------------------------------
# Home Optimizer — Home Assistant Addon
#
# Build context: repository root  (docker build -f addon/Dockerfile .)
# Base image: official HA Python base for the target architecture.
# ---------------------------------------------------------------------------

ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base-python:3.12
FROM ${BUILD_FROM}

# Disable .pyc files and enable unbuffered logs (fail-fast visibility).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build dependencies required by numpy/cvxpy on Alpine-based images.
RUN apk add --no-cache \
    build-base \
    libffi-dev \
    musl-dev \
    gcc \
    g++ \
    lapack-dev \
    openblas-dev

WORKDIR /app

# Copy only the files needed for dependency installation first so Docker
# can cache the pip layer when only source code changes.
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the package and all runtime dependencies.
# The `.[dev]` extras are intentionally excluded from the addon image.
RUN pip install --no-cache-dir .

# Copy the addon run script last (cheapest layer to invalidate).
COPY addon/run.sh /run.sh
RUN chmod a+x /run.sh

# Labels required by the HA addon build system.
ARG BUILD_ARCH
ARG BUILD_DATE
ARG BUILD_DESCRIPTION
ARG BUILD_NAME
ARG BUILD_REF
ARG BUILD_REPOSITORY
ARG BUILD_VERSION
LABEL \
    io.hass.name="${BUILD_NAME}" \
    io.hass.description="${BUILD_DESCRIPTION}" \
    io.hass.arch="${BUILD_ARCH}" \
    io.hass.type="addon" \
    io.hass.version="${BUILD_VERSION}" \
    maintainer="gerjanvandenbosch"

CMD ["/run.sh"]

