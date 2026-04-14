"""Abstract sensor backend interface.

All backends (Home Assistant, local/MQTT, mock) implement SensorBackend.
The MPC core only depends on this interface — never on a specific backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class LiveReadings:
    """Snapshot of all live sensor values required by one MPC cycle.

    Attributes
    ----------
    room_temperature_c:  T_r — measured room-air temperature [°C].
    pv_power_kw:         P_pv — current PV production [kW].  0.0 if no PV.
    hp_power_kw:         P_hp_total — current total heat-pump electrical power [kW].
    timestamp:           UTC timestamp of the reading.
    """

    room_temperature_c: float
    pv_power_kw: float
    hp_power_kw: float
    timestamp: datetime

    @property
    def net_import_kw(self) -> float:
        """Estimated grid import: HP load minus PV [kW].  May be negative (export)."""
        return self.hp_power_kw - self.pv_power_kw


class SensorBackend(ABC):
    """Abstract interface for reading live sensor values.

    Implement this for every data source (HA REST API, MQTT, CSV, …).
    """

    @abstractmethod
    def get_room_temperature_c(self) -> float:
        """Return the current room-air temperature T_r [°C]."""

    @abstractmethod
    def get_pv_power_kw(self) -> float:
        """Return the current PV production P_pv [kW].  Return 0.0 if no PV."""

    @abstractmethod
    def get_hp_power_kw(self) -> float:
        """Return the current total heat-pump electrical power P_hp_total [kW]."""

    def read_all(self) -> LiveReadings:
        """Read all sensors and return a single timestamped snapshot."""
        return LiveReadings(
            room_temperature_c=self.get_room_temperature_c(),
            pv_power_kw=self.get_pv_power_kw(),
            hp_power_kw=self.get_hp_power_kw(),
            timestamp=datetime.now(tz=timezone.utc),
        )

