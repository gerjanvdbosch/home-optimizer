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
    hp_supply_target_temperature_c:
        Heat-pump target supply-water temperature [°C].  This is the controller
        setpoint (requested leaving-water temperature), used to quantify control
        error versus ``hp_supply_temperature_c`` for model identification.
    hp_return_temperature_c:
        Heat-pump return-water temperature [°C].
    hp_flow_lpm:
        Heat-pump volumetric flow rate [L/min].
    hp_electric_power_kw:
        Total heat-pump electrical power draw [kW].
    hp_mode:
        Heat-pump operating mode (for example ``"off"``, ``"ufh"``,
        ``"dhw"``, ``"defrost"``).
    p1_net_power_kw:
        Net grid power from the P1 smart meter [kW].
        Positive = importing from the grid; negative = exporting to the grid
        (PV surplus).  The P1 port reports a single signed value; splitting
        it into separate import/export columns is not needed.
    pv_output_kw:
        PV production [kW].
    thermostat_setpoint_c:
        Active room-temperature setpoint [°C].
    dhw_top_temperature_c:
        DHW tank top-layer temperature T_top [°C].
    dhw_bottom_temperature_c:
        DHW tank bottom-layer temperature T_bot [°C].
    shutter_living_room_pct:
        Living-room shutter position [%].  100 = fully open (maximum solar
        gain), 0 = fully closed (zero solar gain through glazing).  Used to
        compute the effective solar transmittance η_eff = η × (shutter / 100)
        for the Q_solar disturbance term (§4).
        Must be in [0, 100].
    defrost_active:
        ``True`` when the heat pump is executing a defrost cycle.  During
        defrost the refrigerant cycle is reversed and the HP *extracts* heat
        from the hydraulic circuit instead of delivering it, making P_UFH
        effectively negative.  Exposed separately from ``hp_mode`` because
        many controllers report defrost as a sub-state while ``hp_mode``
        still reads ``"ufh"`` or ``"dhw"``.
    booster_heater_active:
        ``True`` when the DHW booster (electric resistance) element is active.
        The booster heats the **top layer** of the tank (assumption A5 note,
        §11), adding power P_booster_kw to C_top instead of C_bot.  The
        Kalman filter must account for this to avoid model-measurement
        discrepancies during legionella or peak-load events.
    boiler_ambient_temp_c:
        Measured ambient temperature around the DHW boiler [°C].  This is the
        direct measurement of T_amb used in the standby-loss term
        Q_loss = (T_layer − T_amb) / R_loss (§8.3, §9.2).  Replaces the
        estimated value in DHWForecastHorizon.t_amb_c at the current timestep.
    refrigerant_condensation_temp_c:
        Measured refrigerant condensation temperature T_cond [°C].  Used
        directly in the Carnot COP formula (§14.1):
            COP = η_Carnot × T_cond_K / (T_cond_K − T_evap_K)
        Replaces the approximation T_cond ≈ T_supply + Δ_cond when this
        sensor is available.
    refrigerant_temp_c:
        Measured refrigerant evaporator (suction) temperature T_evap [°C].
        Used directly in the Carnot COP formula (§14.1).  Replaces the
        approximation T_evap ≈ T_out − Δ_evap when this sensor is available.
    timestamp:
        UTC timestamp of the reading.
    """

    room_temperature_c: float
    outdoor_temperature_c: float
    hp_supply_temperature_c: float
    hp_supply_target_temperature_c: float
    hp_return_temperature_c: float
    hp_flow_lpm: float
    hp_electric_power_kw: float
    hp_mode: str
    p1_net_power_kw: float
    pv_output_kw: float
    thermostat_setpoint_c: float
    dhw_top_temperature_c: float
    dhw_bottom_temperature_c: float
    # --- New sensors (§4, §9.2, §11, §14.1) ---
    shutter_living_room_pct: float
    defrost_active: bool
    booster_heater_active: bool
    boiler_ambient_temp_c: float
    refrigerant_condensation_temp_c: float
    refrigerant_temp_c: float
    timestamp: datetime

    def __post_init__(self) -> None:
        numeric_fields = (
            "room_temperature_c",
            "outdoor_temperature_c",
            "hp_supply_temperature_c",
            "hp_supply_target_temperature_c",
            "hp_return_temperature_c",
            "hp_flow_lpm",
            "hp_electric_power_kw",
            "p1_net_power_kw",
            "pv_output_kw",
            "thermostat_setpoint_c",
            "dhw_top_temperature_c",
            "dhw_bottom_temperature_c",
            "shutter_living_room_pct",
            "boiler_ambient_temp_c",
            "refrigerant_condensation_temp_c",
            "refrigerant_temp_c",
        )
        for field_name in numeric_fields:
            _assert_finite(field_name, float(getattr(self, field_name)))

        # p1_net_power_kw is signed (negative = export) — only hp/pv must be ≥ 0.
        non_negative_fields = (
            "hp_flow_lpm",
            "hp_electric_power_kw",
            "pv_output_kw",
        )
        for field_name in non_negative_fields:
            if float(getattr(self, field_name)) < 0.0:
                raise ValueError(f"{field_name} must be non-negative.")

        # Shutter is a percentage [0, 100].
        shutter = float(self.shutter_living_room_pct)
        if not 0.0 <= shutter <= 100.0:
            raise ValueError(
                f"shutter_living_room_pct must be in [0, 100], got {shutter}."
            )

        # Coerce bool fields — JSON may deliver 0/1 integers.
        object.__setattr__(self, "defrost_active", bool(self.defrost_active))
        object.__setattr__(self, "booster_heater_active", bool(self.booster_heater_active))

        normalized_hp_mode = self.hp_mode.strip().lower()
        if not normalized_hp_mode:
            raise ValueError("hp_mode must not be empty.")
        object.__setattr__(self, "hp_mode", normalized_hp_mode)
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware.")

    @property
    def net_grid_power_kw(self) -> float:
        """Net grid power [kW] from the P1 meter.

        Positive = importing from the grid; negative = exporting (PV surplus).
        """
        return self.p1_net_power_kw

    @property
    def hp_delta_t_c(self) -> float:
        """Heat-pump supply-return temperature difference [K ≡ °C]."""
        return self.hp_supply_temperature_c - self.hp_return_temperature_c

    @property
    def shutter_fraction(self) -> float:
        """Shutter openness as a unit fraction [0.0–1.0].

        Used to compute effective solar transmittance:
            η_eff = η × shutter_fraction   (§4, Q_solar disturbance)
        """
        return self.shutter_living_room_pct / 100.0


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

