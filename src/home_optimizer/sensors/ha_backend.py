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
    shutter_living_room:
        Shutter position entity [%].  100 = fully open, 0 = fully closed.
        Many HA cover integrations expose this as ``cover.living_room``
        with attribute ``current_position``, but it can also be a
        ``sensor`` entity that directly reports the percentage.
    defrost_active_entity_id:
        Entity reporting ``"on"``/``"off"`` (or ``1``/``0``) for the
        heat-pump defrost sub-state (§14.1, Kalman correction).
    booster_heater_active_entity_id:
        Entity reporting ``"on"``/``"off"`` for the DHW booster element
        (§11, top-layer power injection).
    boiler_ambient_temperature:
        Ambient temperature sensor inside the boiler cupboard [°C] (T_amb, §8.3).
    refrigerant_condensation_temperature:
        Measured refrigerant condensation temperature T_cond [°C] (§14.1).
    refrigerant_temperature:
        Measured refrigerant evaporator / suction temperature T_evap [°C] (§14.1).
    base_url:
        HA base URL.  Falls back to ``HA_BASE_URL`` env var, then the
        supervisor addon endpoint ``http://supervisor/core``.
    token:
        Long-lived access token or supervisor token.  Falls back to
        ``HA_TOKEN`` env var, then ``SUPERVISOR_TOKEN`` env var.
    timeout:
        HTTP request timeout in seconds.  Default ``10.0``.

    .. important::

        This backend provides a **placeholder value** (``0.0``) for the
        weather-sourced :class:`LiveReadings` field ``t_mains_estimated_c``.
        In production, wrap this backend with
        :class:`~home_optimizer.sensors.WeatherAugmentedBackend` so that the
        seasonal T_mains estimate is injected before telemetry is persisted.
        Telemetry collected without the wrapper will have zero T_mains, which
        renders DHW energy-balance training data unusable.

    Examples
    --------
    Standalone (LAN access, outside HA)::

        from home_optimizer.sensors import HomeAssistantBackend, HAEntityConfig

        backend = HomeAssistantBackend(
            room_temperature=HAEntityConfig("sensor.living_room_temperature"),
            outdoor_temperature=HAEntityConfig("sensor.outdoor_temperature"),
            hp_supply_temperature=HAEntityConfig("sensor.heat_pump_supply_temperature"),
            hp_supply_target_temperature=HAEntityConfig("sensor.heat_pump_supply_target_temperature"),
            hp_return_temperature=HAEntityConfig("sensor.heat_pump_return_temperature"),
            hp_flow_lpm=HAEntityConfig("sensor.heat_pump_flow_lpm"),
            hp_electric_power=HAEntityConfig("sensor.heat_pump_power_kw"),
            hp_mode_entity_id="sensor.heat_pump_mode",
            p1_net_power=HAEntityConfig("sensor.p1_net_power_w", scale=0.001),
            pv_output=HAEntityConfig("sensor.pv_power_w", scale=0.001),
            thermostat_setpoint=HAEntityConfig("sensor.room_setpoint_temperature"),
            dhw_top_temperature=HAEntityConfig("sensor.dhw_top_temperature"),
            dhw_bottom_temperature=HAEntityConfig("sensor.dhw_bottom_temperature"),
            shutter_living_room=HAEntityConfig("sensor.shutter_living_room_pct"),
            defrost_active_entity_id="binary_sensor.heat_pump_defrost",
            booster_heater_active_entity_id="binary_sensor.dhw_booster_heater",
            boiler_ambient_temperature=HAEntityConfig("sensor.boiler_ambient_temp"),
            refrigerant_condensation_temperature=HAEntityConfig("sensor.refrigerant_condensation_temp"),
            refrigerant_temperature=HAEntityConfig("sensor.refrigerant_temp"),
            base_url="http://homeassistant.local:8123",
            token="YOUR_LONG_LIVED_TOKEN",
        )
        readings = backend.read_all()
    """

    def __init__(
        self,
        *,
        room_temperature: HAEntityConfig,
        outdoor_temperature: HAEntityConfig,
        hp_supply_temperature: HAEntityConfig,
        hp_supply_target_temperature: HAEntityConfig,
        hp_return_temperature: HAEntityConfig,
        hp_flow_lpm: HAEntityConfig,
        hp_electric_power: HAEntityConfig,
        hp_mode_entity_id: str,
        p1_net_power: HAEntityConfig,
        pv_output: HAEntityConfig,
        thermostat_setpoint: HAEntityConfig,
        dhw_top_temperature: HAEntityConfig,
        dhw_bottom_temperature: HAEntityConfig,
        shutter_living_room: HAEntityConfig,
        defrost_active_entity_id: str,
        booster_heater_active_entity_id: str,
        boiler_ambient_temperature: HAEntityConfig,
        refrigerant_condensation_temperature: HAEntityConfig,
        refrigerant_temperature: HAEntityConfig,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._room_temperature = room_temperature
        self._outdoor_temperature = outdoor_temperature
        self._hp_supply_temperature = hp_supply_temperature
        self._hp_supply_target_temperature = hp_supply_target_temperature
        self._hp_return_temperature = hp_return_temperature
        self._hp_flow_lpm = hp_flow_lpm
        self._hp_electric_power = hp_electric_power
        self._hp_mode_entity_id = hp_mode_entity_id
        self._p1_net_power = p1_net_power
        self._pv_output = pv_output
        self._thermostat_setpoint = thermostat_setpoint
        self._dhw_top_temperature = dhw_top_temperature
        self._dhw_bottom_temperature = dhw_bottom_temperature
        self._shutter_living_room = shutter_living_room
        self._defrost_active_entity_id = defrost_active_entity_id
        self._booster_heater_active_entity_id = booster_heater_active_entity_id
        self._boiler_ambient_temperature = boiler_ambient_temperature
        self._refrigerant_condensation_temperature = refrigerant_condensation_temperature
        self._refrigerant_temperature = refrigerant_temperature
        if not self._hp_mode_entity_id:
            raise ValueError("hp_mode_entity_id must not be empty.")
        if not self._defrost_active_entity_id:
            raise ValueError("defrost_active_entity_id must not be empty.")
        if not self._booster_heater_active_entity_id:
            raise ValueError("booster_heater_active_entity_id must not be empty.")

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
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_entity_json(self, entity_id: str) -> dict:
        """GET /api/states/<entity_id> and return the full JSON response dict.

        This is the low-level HTTP helper used by all public read methods.
        It handles connection and HTTP errors uniformly (Fail-Fast §5).

        Args:
            entity_id: Full HA entity ID, e.g. ``"sensor.living_room_temperature"``.

        Returns:
            Full HA state JSON dict, e.g.
            ``{"entity_id": ..., "state": ..., "attributes": {...}}``.

        Raises:
            ConnectionError: On network or HTTP errors.
        """
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
        return response.json()

    def _fetch_raw_state(self, entity_id: str) -> str:
        """GET /api/states/<entity_id> and return the raw state string."""
        data = self._fetch_entity_json(entity_id)
        raw_state = str(data.get("state", "unavailable"))
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

    def _fetch_bool_state(self, entity_id: str) -> bool:
        """Return the boolean state of a HA binary_sensor or switch entity.

        Accepts HA canonical values ``"on"``/``"off"`` as well as numeric
        ``"1"``/``"0"``.  Any unrecognised state raises ``ValueError`` so
        the snapshot is never populated with ambiguous data (Fail-Fast).
        """
        raw = self._fetch_raw_state(entity_id).lower()
        if raw in ("on", "1", "true"):
            return True
        if raw in ("off", "0", "false"):
            return False
        raise ValueError(
            f"Entity {entity_id!r} returned unrecognised boolean state {raw!r}. "
            "Expected 'on'/'off', '1'/'0', or 'true'/'false'."
        )

    # ------------------------------------------------------------------
    # Zone / location helpers
    # ------------------------------------------------------------------

    def fetch_zone_location(self, zone_entity_id: str = "zone.home") -> tuple[float, float]:
        """Fetch the geographic coordinates of a Home Assistant zone entity.

        HA exposes zone coordinates as entity *attributes* (``latitude`` and
        ``longitude``), not as the numeric state.  This method retrieves the
        full entity JSON and extracts those attributes.

        Typical use at addon startup::

            latitude, longitude = backend.fetch_zone_location("zone.home")

        Args:
            zone_entity_id: HA entity ID of the zone.  Default ``"zone.home"``.

        Returns:
            Tuple ``(latitude [°N], longitude [°E])`` as floats.

        Raises:
            ConnectionError: If the HA REST API is unreachable.
            ValueError: If the entity is unavailable or the ``latitude`` /
                ``longitude`` attributes are absent.
        """
        data = self._fetch_entity_json(zone_entity_id)
        attributes: dict = data.get("attributes", {})
        lat = attributes.get("latitude")
        lon = attributes.get("longitude")
        if lat is None or lon is None:
            missing = [k for k, v in (("latitude", lat), ("longitude", lon)) if v is None]
            raise ValueError(
                f"Zone entity {zone_entity_id!r} is missing attribute(s) "
                f"{missing!r}.  Ensure the entity is a valid HA zone with a "
                "configured home location."
            )
        return float(lat), float(lon)

    # ------------------------------------------------------------------
    # SensorBackend interface
    # ------------------------------------------------------------------

    def read_all(self) -> LiveReadings:
        """Read one complete telemetry snapshot from Home Assistant."""
        return LiveReadings(
            room_temperature_c=self._fetch_state(self._room_temperature),
            outdoor_temperature_c=self._fetch_state(self._outdoor_temperature),
            hp_supply_temperature_c=self._fetch_state(self._hp_supply_temperature),
            hp_supply_target_temperature_c=self._fetch_state(self._hp_supply_target_temperature),
            hp_return_temperature_c=self._fetch_state(self._hp_return_temperature),
            hp_flow_lpm=max(self._fetch_state(self._hp_flow_lpm), 0.0),
            hp_electric_power_kw=max(self._fetch_state(self._hp_electric_power), 0.0),
            hp_mode=self._fetch_text_state(self._hp_mode_entity_id),
            p1_net_power_kw=self._fetch_state(self._p1_net_power),
            pv_output_kw=max(self._fetch_state(self._pv_output), 0.0),
            thermostat_setpoint_c=self._fetch_state(self._thermostat_setpoint),
            dhw_top_temperature_c=self._fetch_state(self._dhw_top_temperature),
            dhw_bottom_temperature_c=self._fetch_state(self._dhw_bottom_temperature),
            shutter_living_room_pct=self._fetch_state(self._shutter_living_room),
            defrost_active=self._fetch_bool_state(self._defrost_active_entity_id),
            booster_heater_active=self._fetch_bool_state(self._booster_heater_active_entity_id),
            boiler_ambient_temp_c=self._fetch_state(self._boiler_ambient_temperature),
            refrigerant_condensation_temp_c=self._fetch_state(self._refrigerant_condensation_temperature),
            refrigerant_temp_c=self._fetch_state(self._refrigerant_temperature),
            # Weather field: placeholder 0.0 — must be overridden by WeatherAugmentedBackend.
            # T_mains = 0.0 is a sentinel indicating the wrapper is not in use.
            t_mains_estimated_c=0.0,
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

