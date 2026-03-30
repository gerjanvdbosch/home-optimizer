#!/usr/bin/with-contenv bashio
set -e

bashio::log.info "Preparing to start..."

export MODEL_DIR="/config/models"
export DB_DIR="/config/db"

mkdir -p "${MODEL_DIR}"
mkdir -p "${DB_DIR}"

export SOLAR="$(bashio::config 'solar' || echo '{}')"
export CLIMATE_ROOM_JSON="$(bashio::config 'climate.room_schedule')"
export CLIMATE_DHW_JSON="$(bashio::config 'climate.dhw_schedule')"
export LOG_LEVEL="$(bashio::config 'log_level')"

python3 -u ./src/coordinator.py
