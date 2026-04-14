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
  Use HAEntityConfig(entity_id, scale=0.001) to convert W → kW.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import httpx

from .base import SensorBackend

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
    """Read live sensor states from the Home Assistant REST API.

    Parameters
    ----------
    room_temp:
        Entity config for the room-air temperature sensor (°C).
    pv_power:
        Entity config for the PV inverter / smart meter production sensor.
    hp_power:
        Entity config for the total heat-pump electrical power sensor.
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
            room_temp=HAEntityConfig("sensor.living_room_temperature"),
            pv_power=HAEntityConfig("sensor.pv_power_w", scale=0.001),
            hp_power=HAEntityConfig("sensor.heat_pump_power_kw"),
            base_url="http://homeassistant.local:8123",
            token="YOUR_LONG_LIVED_TOKEN",
        )
        readings = backend.read_all()

    Addon mode (no base_url / token needed — supervisor injects them)::

        backend = HomeAssistantBackend(
            room_temp=HAEntityConfig("sensor.living_room_temperature"),
            pv_power=HAEntityConfig("sensor.pv_power_kw"),
            hp_power=HAEntityConfig("sensor.heat_pump_power_kw"),
        )
    """

    def __init__(
        self,
        room_temp: HAEntityConfig,
        pv_power: HAEntityConfig,
        hp_power: HAEntityConfig,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._room_temp = room_temp
        self._pv_power = pv_power
        self._hp_power = hp_power

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

    def _fetch_state(self, cfg: HAEntityConfig) -> float:
        """GET /api/states/<entity_id> and return the numeric state × scale."""
        url = f"{self._base_url}/api/states/{cfg.entity_id}"
        try:
            response = self._client.get(url)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ConnectionError(
                f"HA API returned HTTP {exc.response.status_code} for entity "
                f"{cfg.entity_id!r}. Check entity_id and HA token."
            ) from exc
        except httpx.HTTPError as exc:
            raise ConnectionError(
                f"Could not reach Home Assistant at {self._base_url}: {exc}"
            ) from exc

        raw_state: str = response.json().get("state", "unavailable")
        if raw_state in ("unavailable", "unknown", "none", ""):
            raise ValueError(
                f"Entity {cfg.entity_id!r} reports state {raw_state!r}. "
                "Ensure the entity is available in Home Assistant."
            )
        try:
            return float(raw_state) * cfg.scale
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Entity {cfg.entity_id!r} state {raw_state!r} is not numeric."
            ) from exc

    # ------------------------------------------------------------------
    # SensorBackend interface
    # ------------------------------------------------------------------

    def get_room_temperature_c(self) -> float:
        """Return current room temperature T_r [°C]."""
        return self._fetch_state(self._room_temp)

    def get_pv_power_kw(self) -> float:
        """Return current PV production P_pv [kW].  Clipped to ≥ 0."""
        return max(self._fetch_state(self._pv_power), 0.0)

    def get_hp_power_kw(self) -> float:
        """Return current total heat-pump power P_hp_total [kW].  Clipped to ≥ 0."""
        return max(self._fetch_state(self._hp_power), 0.0)

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

