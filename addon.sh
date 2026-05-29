#!/usr/bin/with-contenv bashio

set -euo pipefail

bashio::log.info "Starting Home Optimizer addon..."
bashio::log.info "Options: $(bashio::config)"

exec python -m addon
