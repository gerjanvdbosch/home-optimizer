"""Abstract sensor backend interface for telemetry collection.

All backends return the same physics-oriented :class:`LiveReadings` snapshot.
The telemetry layer stores these values for later model identification,
forecast training, and MPC backtesting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite


def _assert_finite(name: str, value: float) -> None:
    """Fail fast when a sensor reports a non-finite numeric value."""
    if not isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}.")


@dataclass(frozen=True, slots=True)
class LiveReadings:
    """Snapshot of all live telemetry values required for storage and training.

    Attributes
    ----------
    room_temperature_c:
        Measured room-air temperature T_r [°C].
    outdoor_temperature_c:
        Outdoor air temperature T_out [°C].
    hp_supply_temperature_c:
        Heat-pump supply-water temperature [°C].  This is the canonical raw
        hydraulic sensor regardless of whether the machine is in UFH or DHW mode.
    hp_return_temperature_c:
        Heat-pump return-water temperature [°C].
    hp_flow_lpm:
        Heat-pump volumetric flow rate [L/min].
    hp_electric_power_kw:
        Total heat-pump electrical power draw [kW].
    hp_mode:
        Heat-pump operating mode (for example ``"off"``, ``"ufh"``,
        ``"dhw"``, ``"defrost"``).
    grid_import_kw:
        Grid import power [kW].
    grid_export_kw:
        Grid export power [kW].
    pv_output_kw:
        PV production [kW].
    thermostat_setpoint_c:
        Active room-temperature setpoint [°C].
    dhw_top_temperature_c:
        DHW tank top-layer temperature T_top [°C].
    dhw_bottom_temperature_c:
        DHW tank bottom-layer temperature T_bot [°C].
    timestamp:
        UTC timestamp of the reading.
    """

    room_temperature_c: float
    outdoor_temperature_c: float
    hp_supply_temperature_c: float
    hp_return_temperature_c: float
    hp_flow_lpm: float
    hp_electric_power_kw: float
    hp_mode: str
    grid_import_kw: float
    grid_export_kw: float
    pv_output_kw: float
    thermostat_setpoint_c: float
    dhw_top_temperature_c: float
    dhw_bottom_temperature_c: float
    timestamp: datetime

    def __post_init__(self) -> None:
        numeric_fields = (
            "room_temperature_c",
            "outdoor_temperature_c",
            "hp_supply_temperature_c",
            "hp_return_temperature_c",
            "hp_flow_lpm",
            "hp_electric_power_kw",
            "grid_import_kw",
            "grid_export_kw",
            "pv_output_kw",
            "thermostat_setpoint_c",
            "dhw_top_temperature_c",
            "dhw_bottom_temperature_c",
        )
        for field_name in numeric_fields:
            _assert_finite(field_name, float(getattr(self, field_name)))

        non_negative_fields = (
            "hp_flow_lpm",
            "hp_electric_power_kw",
            "grid_import_kw",
            "grid_export_kw",
            "pv_output_kw",
        )
        for field_name in non_negative_fields:
            if float(getattr(self, field_name)) < 0.0:
                raise ValueError(f"{field_name} must be non-negative.")

        normalized_hp_mode = self.hp_mode.strip().lower()
        if not normalized_hp_mode:
            raise ValueError("hp_mode must not be empty.")
        object.__setattr__(self, "hp_mode", normalized_hp_mode)
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware.")

    @property
    def net_grid_power_kw(self) -> float:
        """Net grid power [kW] = import − export."""
        return self.grid_import_kw - self.grid_export_kw

    @property
    def hp_delta_t_c(self) -> float:
        """Heat-pump supply-return temperature difference [K ≡ °C]."""
        return self.hp_supply_temperature_c - self.hp_return_temperature_c


class SensorBackend(ABC):
    """Abstract interface for reading one complete telemetry snapshot.

    Implementations must fail fast when a required sensor is unavailable.  The
    persistence layer relies on a fully populated :class:`LiveReadings` object.
    """

    @abstractmethod
    def read_all(self) -> LiveReadings:
        """Read all configured sensors and return a single timestamped snapshot."""

    def now_utc(self) -> datetime:
        """Return the backend timestamp source in UTC.

        A separate method keeps timestamp creation centralised and testable.
        """
        return datetime.now(tz=timezone.utc)

    @abstractmethod
    def close(self) -> None:
        """Release backend resources.

        Most backends are stateless and do not need explicit cleanup.
        """

