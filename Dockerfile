# ---------------------------------------------------------------------------
# Home Optimizer — Home Assistant Addon
#
# Build context: repo root (bevat src/, pyproject.toml én dit Dockerfile).
#
# Waarom debian-base en niet alpine-base:
#   Alpine vereist compilatie van numpy/cvxpy from source op ARM (~30 min).
#   Debian heeft pre-built manylinux/piwheels ARM wheels → build < 5 min.
#
# BUILD_FROM wordt door de HA supervisor automatisch ingesteld op de
# correcte arch-specifieke image. De debian-base is een multi-arch manifest
# (aarch64, amd64, armv7, armhf, i386) — geen build.yaml nodig.
# Lokaal bouwen:
#   docker build --build-arg BUILD_FROM=ghcr.io/hassio-addons/debian-base:8.1.4 .
# ---------------------------------------------------------------------------

ARG BUILD_FROM=ghcr.io/hassio-addons/debian-base:8.1.4
FROM ${BUILD_FROM}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# --- Laag 1: Python installatie via apt (zelden gewijzigd → bijna altijd gecached) ---
# python3-minimal + pip + venv zijn genoeg; geen build-tools nodig omdat
# numpy/cvxpy/scipy pre-built ARM wheels hebben op PyPI en piwheels.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-minimal \
        python3-pip \
        python3-venv && \
    python3 -m venv /opt/venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*

WORKDIR /app

# --- Laag 2: pakketdefinitie + broncode (opnieuw gebouwd alleen bij codewijziging) ---
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Installeer het pakket inclusief dependencies.
# piwheels.org levert pre-compiled ARM wheels voor de Pi → geen compilatie.
RUN /opt/venv/bin/pip install --no-cache-dir \
    --extra-index-url https://www.piwheels.org/simple \
    .

# --- Laag 3: run-script ---
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
