# ---------------------------------------------------------------------------
# Home Optimizer — Home Assistant Addon
#
# Build context: repo root (bevat src/, pyproject.toml én dit Dockerfile).
# Het pakket wordt via COPY lokaal geïnstalleerd — geen GitHub-authenticatie
# nodig tijdens de build.
#
# BUILD_FROM wordt door de HA supervisor automatisch ingesteld op de
# correcte arch-specifieke Python base image (build.yaml is deprecated).
# Lokaal bouwen: geef het argument mee, bijv.:
#   docker build --build-arg BUILD_FROM=ghcr.io/home-assistant/amd64-base-python:3.12-alpine3.18 .
# ---------------------------------------------------------------------------

ARG BUILD_FROM
FROM ${BUILD_FROM}

# Disable .pyc files and enable unbuffered logs (fail-fast visibility).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build dependencies required by numpy/cvxpy.
# git is niet meer nodig — pakket wordt lokaal geïnstalleerd.
RUN apk add --no-cache \
    build-base \
    libffi-dev \
    musl-dev \
    gcc \
    g++ \
    lapack-dev \
    openblas-dev

WORKDIR /app

# Kopieer de pakketdefinitie en broncode vanuit de repo root.
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Installeer het pakket lokaal (inclusief alle dependencies).
RUN pip install --no-cache-dir .

# Kopieer het addon run-script (ligt nu ook in de repo root).
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

