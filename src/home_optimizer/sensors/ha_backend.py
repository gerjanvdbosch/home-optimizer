"""Home Assistant REST API sensor backend.

Works in two modes:

Addon mode (running inside HA as an addon)
  - base_url resolved from env HA_BASE_URL, defaults to "http://supervisor/core"
  - token    resolved from env SUPERVISOR_TOKEN (injected by the HA supervisor)

Standalone mode (running outside HA on the same network)
  - base_url = "http://homeassistant.local:8123"  (or set HA_BASE_URL env var)
  - token    = long-lived access token from HA profile page  (or set HA_TOKEN env var)

Entity unit scaling
  HA climate / thermometer entities report temperature in °C — no conversion needed.
  Power entities may be in W or kW depending on the integration.
  Use ``HAEntityConfig(entity_id, scale=0.001)`` to convert W → kW.
  Likewise use scaling to convert m³/h → L/min or similar if your integration
  exposes different hydraulic units.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import httpx

from .base import LiveReadings, SensorBackend

_HA_ADDON_BASE = "http://supervisor/core"


@dataclass(frozen=True, slots=True)
class HAEntityConfig:
    """Reference to one HA entity with an optional unit-scaling factor.

    Parameters
    ----------
    entity_id:
        Full HA entity ID, e.g. ``"sensor.living_room_temperature"``.
    scale:
        Multiply the raw numeric state by this factor before returning.
        Use ``0.001`` to convert watts → kilowatts.  Default ``1.0``.

    Examples
    --------
    >>> HAEntityConfig("sensor.pv_power_w", scale=0.001)   # W → kW
    >>> HAEntityConfig("sensor.heat_pump_power_kw")         # already kW
    """

    entity_id: str
    scale: float = field(default=1.0)

    def __post_init__(self) -> None:
        if not self.entity_id:
            raise ValueError("entity_id must not be empty.")
        if self.scale == 0.0:
            raise ValueError("scale must be non-zero.")


class HomeAssistantBackend(SensorBackend):
    """Read the complete telemetry snapshot from the Home Assistant REST API.

    Parameters
    ----------
    All numeric parameters are entity configs with an optional scaling factor.
    ``hp_mode_entity_id`` is read as a raw text state because operating modes are
    categorical rather than numeric.
    base_url:
        HA base URL.  Falls back to ``HA_BASE_URL`` env var, then the
        supervisor addon endpoint ``http://supervisor/core``.
    token:
        Long-lived access token or supervisor token.  Falls back to
        ``HA_TOKEN`` env var, then ``SUPERVISOR_TOKEN`` env var.
    timeout:
        HTTP request timeout in seconds.  Default ``10.0``.

    Examples
    --------
    Standalone (LAN access, outside HA)::

        from home_optimizer.sensors import HomeAssistantBackend, HAEntityConfig

        backend = HomeAssistantBackend(
            room_temperature=HAEntityConfig("sensor.living_room_temperature"),
            outdoor_temperature=HAEntityConfig("sensor.outdoor_temperature"),
            hp_supply_temperature=HAEntityConfig("sensor.heat_pump_supply_temperature"),
            hp_return_temperature=HAEntityConfig("sensor.heat_pump_return_temperature"),
            hp_flow_lpm=HAEntityConfig("sensor.heat_pump_flow_lpm"),
            hp_electric_power=HAEntityConfig("sensor.heat_pump_power_kw"),
            hp_mode_entity_id="sensor.heat_pump_mode",
            grid_import=HAEntityConfig("sensor.grid_import_power_w", scale=0.001),
            grid_export=HAEntityConfig("sensor.grid_export_power_w", scale=0.001),
            pv_output=HAEntityConfig("sensor.pv_power_w", scale=0.001),
            thermostat_setpoint=HAEntityConfig("sensor.room_setpoint_temperature"),
            dhw_top_temperature=HAEntityConfig("sensor.dhw_top_temperature"),
            dhw_bottom_temperature=HAEntityConfig("sensor.dhw_bottom_temperature"),
            base_url="http://homeassistant.local:8123",
            token="YOUR_LONG_LIVED_TOKEN",
        )
        readings = backend.read_all()

    Addon mode (no base_url / token needed — supervisor injects them)::

        backend = HomeAssistantBackend(
            room_temperature=HAEntityConfig("sensor.living_room_temperature"),
            outdoor_temperature=HAEntityConfig("sensor.outdoor_temperature"),
            hp_supply_temperature=HAEntityConfig("sensor.heat_pump_supply_temperature"),
            hp_return_temperature=HAEntityConfig("sensor.heat_pump_return_temperature"),
            hp_flow_lpm=HAEntityConfig("sensor.heat_pump_flow_lpm"),
            hp_electric_power=HAEntityConfig("sensor.heat_pump_power_kw"),
            hp_mode_entity_id="sensor.heat_pump_mode",
            grid_import=HAEntityConfig("sensor.grid_import_power_kw"),
            grid_export=HAEntityConfig("sensor.grid_export_power_kw"),
            pv_output=HAEntityConfig("sensor.pv_power_kw"),
            thermostat_setpoint=HAEntityConfig("sensor.room_setpoint_temperature"),
            dhw_top_temperature=HAEntityConfig("sensor.dhw_top_temperature"),
            dhw_bottom_temperature=HAEntityConfig("sensor.dhw_bottom_temperature"),
        )
    """

    def __init__(
        self,
        *,
        room_temperature: HAEntityConfig,
        outdoor_temperature: HAEntityConfig,
        hp_supply_temperature: HAEntityConfig,
        hp_return_temperature: HAEntityConfig,
        hp_flow_lpm: HAEntityConfig,
        hp_electric_power: HAEntityConfig,
        hp_mode_entity_id: str,
        grid_import: HAEntityConfig,
        grid_export: HAEntityConfig,
        pv_output: HAEntityConfig,
        thermostat_setpoint: HAEntityConfig,
        dhw_top_temperature: HAEntityConfig,
        dhw_bottom_temperature: HAEntityConfig,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._room_temperature = room_temperature
        self._outdoor_temperature = outdoor_temperature
        self._hp_supply_temperature = hp_supply_temperature
        self._hp_return_temperature = hp_return_temperature
        self._hp_flow_lpm = hp_flow_lpm
        self._hp_electric_power = hp_electric_power
        self._hp_mode_entity_id = hp_mode_entity_id
        self._grid_import = grid_import
        self._grid_export = grid_export
        self._pv_output = pv_output
        self._thermostat_setpoint = thermostat_setpoint
        self._dhw_top_temperature = dhw_top_temperature
        self._dhw_bottom_temperature = dhw_bottom_temperature
        if not self._hp_mode_entity_id:
            raise ValueError("hp_mode_entity_id must not be empty.")

        resolved_url = base_url or os.environ.get("HA_BASE_URL") or _HA_ADDON_BASE
        self._base_url = resolved_url.rstrip("/")

        resolved_token = (
            token
            or os.environ.get("HA_TOKEN")
            or os.environ.get("SUPERVISOR_TOKEN")
            or ""
        )
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {resolved_token}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _fetch_raw_state(self, entity_id: str) -> str:
        """GET /api/states/<entity_id> and return the raw state string."""
        url = f"{self._base_url}/api/states/{entity_id}"
        try:
            response = self._client.get(url)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ConnectionError(
                f"HA API returned HTTP {exc.response.status_code} for entity "
                f"{entity_id!r}. Check entity_id and HA token."
            ) from exc
        except httpx.HTTPError as exc:
            raise ConnectionError(
                f"Could not reach Home Assistant at {self._base_url}: {exc}"
            ) from exc

        raw_state = str(response.json().get("state", "unavailable"))
        if raw_state in ("unavailable", "unknown", "none", ""):
            raise ValueError(
                f"Entity {entity_id!r} reports state {raw_state!r}. "
                "Ensure the entity is available in Home Assistant."
            )
        return raw_state

    def _fetch_state(self, cfg: HAEntityConfig) -> float:
        """GET /api/states/<entity_id> and return the numeric state × scale."""
        raw_state = self._fetch_raw_state(cfg.entity_id)
        try:
            return float(raw_state) * cfg.scale
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Entity {cfg.entity_id!r} state {raw_state!r} is not numeric."
            ) from exc

    def _fetch_text_state(self, entity_id: str) -> str:
        """Return the raw string state of an HA entity."""
        return self._fetch_raw_state(entity_id).strip()

    # ------------------------------------------------------------------
    # SensorBackend interface
    # ------------------------------------------------------------------

    def read_all(self) -> LiveReadings:
        """Read one complete telemetry snapshot from Home Assistant."""
        return LiveReadings(
            room_temperature_c=self._fetch_state(self._room_temperature),
            outdoor_temperature_c=self._fetch_state(self._outdoor_temperature),
            hp_supply_temperature_c=self._fetch_state(self._hp_supply_temperature),
            hp_return_temperature_c=self._fetch_state(self._hp_return_temperature),
            hp_flow_lpm=max(self._fetch_state(self._hp_flow_lpm), 0.0),
            hp_electric_power_kw=max(self._fetch_state(self._hp_electric_power), 0.0),
            hp_mode=self._fetch_text_state(self._hp_mode_entity_id),
            grid_import_kw=max(self._fetch_state(self._grid_import), 0.0),
            grid_export_kw=max(self._fetch_state(self._grid_export), 0.0),
            pv_output_kw=max(self._fetch_state(self._pv_output), 0.0),
            thermostat_setpoint_c=self._fetch_state(self._thermostat_setpoint),
            dhw_top_temperature_c=self._fetch_state(self._dhw_top_temperature),
            dhw_bottom_temperature_c=self._fetch_state(self._dhw_bottom_temperature),
            timestamp=self.now_utc(),
        )

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> "HomeAssistantBackend":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

