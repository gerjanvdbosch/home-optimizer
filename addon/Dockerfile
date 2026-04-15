# ---------------------------------------------------------------------------
# Home Optimizer — Home Assistant Addon
#
# Build context: addon/  (set by the HA supervisor — repo root is NOT available)
# The Python package is installed directly from GitHub, which is the standard
# pattern for HA addons; the supervisor cannot pass the repo root as context.
# ---------------------------------------------------------------------------

ARG BUILD_FROM=ghcr.io/home-assistant/aarch64-base-python:3.12-alpine3.18
FROM ${BUILD_FROM}

# Disable .pyc files and enable unbuffered logs (fail-fast visibility).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build + VCS dependencies required by numpy/cvxpy and pip's git-based install.
RUN apk add --no-cache \
    build-base \
    libffi-dev \
    musl-dev \
    gcc \
    g++ \
    lapack-dev \
    openblas-dev \
    git

WORKDIR /app

# Git ref (branch, tag, or SHA) to install.
# The release pipeline overrides this with the concrete version tag.
# Default: main — always gets the latest pushed code.
ARG PACKAGE_REF=main

# Install the package from GitHub.
# The HA supervisor uses the addon directory as the Docker build context, so
# files from the repository root (src/, pyproject.toml) are not accessible via
# COPY.  Installing from the git URL is the canonical solution for HA addons.
RUN pip install --no-cache-dir \
    "git+https://github.com/gerjanvandenbosch/home-optimizer.git@${PACKAGE_REF}"

# Copy the addon run script.
# Path is relative to the build context (= addon/), so no "addon/" prefix.
COPY run.sh /run.sh
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
