#!/usr/bin/with-contenv bashio

set -euo pipefail

bashio::log.info "Starting Home Optimizer..."

exec python -m web.app
