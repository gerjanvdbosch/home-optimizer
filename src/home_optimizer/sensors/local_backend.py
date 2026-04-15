"""Local / standalone sensor backend for the full telemetry snapshot.

The backend can read telemetry from a JSON file, environment variables, or
injected callables.  Every required field must be present; missing values raise
immediately so the persistence layer never stores partial physics data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, TypeAlias

from .base import LiveReadings, SensorBackend

NumericValueSource: TypeAlias = float | Callable[[], float]
TextValueSource: TypeAlias = str | Callable[[], str]
BoolValueSource: TypeAlias = bool | int | Callable[[], bool | int]

_NUMERIC_SENSOR_KEYS: tuple[str, ...] = (
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
    # Seasonal DHW parameter — injected by WeatherAugmentedBackend in production (§9.1).
    # LocalBackend accepts it directly so tests can supply an explicit value.
    "t_mains_estimated_c",
)
_TEXT_SENSOR_KEYS: tuple[str, ...] = ("hp_mode",)
_BOOL_SENSOR_KEYS: tuple[str, ...] = ("defrost_active", "booster_heater_active")
_ALL_SENSOR_KEYS: tuple[str, ...] = _NUMERIC_SENSOR_KEYS + _TEXT_SENSOR_KEYS + _BOOL_SENSOR_KEYS


def _resolve_numeric(source: NumericValueSource) -> float:
    """Evaluate a numeric source: call it if needed, then cast to float."""
    return float(source() if callable(source) else source)


def _resolve_text(source: TextValueSource) -> str:
    """Evaluate a text source: call it if needed, then cast to string."""
    return str(source() if callable(source) else source)


def _resolve_bool(source: BoolValueSource) -> bool:
    """Evaluate a boolean source: call it if needed, then cast to bool.

    JSON delivers 0/1 integers; this coercion ensures a clean Python bool.
    """
    raw = source() if callable(source) else source
    return bool(raw)


class LocalBackend(SensorBackend):
    """Sensor backend for local / offline telemetry capture.

    Parameters
    ----------
    All constructor parameters correspond one-to-one with :class:`LiveReadings`
    fields, except ``timestamp`` which is generated at read time.
    """

    ENV_ROOM_TEMPERATURE_C = "HOME_OPT_ROOM_TEMPERATURE_C"
    ENV_OUTDOOR_TEMPERATURE_C = "HOME_OPT_OUTDOOR_TEMPERATURE_C"
    ENV_HP_SUPPLY_TEMPERATURE_C = "HOME_OPT_HP_SUPPLY_TEMPERATURE_C"
    ENV_HP_SUPPLY_TARGET_TEMPERATURE_C = "HOME_OPT_HP_SUPPLY_TARGET_TEMPERATURE_C"
    ENV_HP_RETURN_TEMPERATURE_C = "HOME_OPT_HP_RETURN_TEMPERATURE_C"
    ENV_HP_FLOW_LPM = "HOME_OPT_HP_FLOW_LPM"
    ENV_HP_ELECTRIC_POWER_KW = "HOME_OPT_HP_ELECTRIC_POWER_KW"
    ENV_HP_MODE = "HOME_OPT_HP_MODE"
    ENV_P1_NET_POWER_KW = "HOME_OPT_P1_NET_POWER_KW"
    ENV_PV_OUTPUT_KW = "HOME_OPT_PV_OUTPUT_KW"
    ENV_THERMOSTAT_SETPOINT_C = "HOME_OPT_THERMOSTAT_SETPOINT_C"
    ENV_DHW_TOP_TEMPERATURE_C = "HOME_OPT_DHW_TOP_TEMPERATURE_C"
    ENV_DHW_BOTTOM_TEMPERATURE_C = "HOME_OPT_DHW_BOTTOM_TEMPERATURE_C"
    ENV_SHUTTER_LIVING_ROOM_PCT = "HOME_OPT_SHUTTER_LIVING_ROOM_PCT"
    ENV_DEFROST_ACTIVE = "HOME_OPT_DEFROST_ACTIVE"
    ENV_BOOSTER_HEATER_ACTIVE = "HOME_OPT_BOOSTER_HEATER_ACTIVE"
    ENV_BOILER_AMBIENT_TEMP_C = "HOME_OPT_BOILER_AMBIENT_TEMP_C"
    ENV_REFRIGERANT_CONDENSATION_TEMP_C = "HOME_OPT_REFRIGERANT_CONDENSATION_TEMP_C"
    ENV_REFRIGERANT_TEMP_C = "HOME_OPT_REFRIGERANT_TEMP_C"
    ENV_T_MAINS_ESTIMATED_C = "HOME_OPT_T_MAINS_ESTIMATED_C"

    def __init__(
        self,
        *,
        room_temperature_c: NumericValueSource,
        outdoor_temperature_c: NumericValueSource,
        hp_supply_temperature_c: NumericValueSource,
        hp_supply_target_temperature_c: NumericValueSource,
        hp_return_temperature_c: NumericValueSource,
        hp_flow_lpm: NumericValueSource,
        hp_electric_power_kw: NumericValueSource,
        hp_mode: TextValueSource,
        p1_net_power_kw: NumericValueSource,
        pv_output_kw: NumericValueSource,
        thermostat_setpoint_c: NumericValueSource,
        dhw_top_temperature_c: NumericValueSource,
        dhw_bottom_temperature_c: NumericValueSource,
        shutter_living_room_pct: NumericValueSource,
        defrost_active: BoolValueSource,
        booster_heater_active: BoolValueSource,
        boiler_ambient_temp_c: NumericValueSource,
        refrigerant_condensation_temp_c: NumericValueSource,
        refrigerant_temp_c: NumericValueSource,
        t_mains_estimated_c: NumericValueSource,
    ) -> None:
        self._room_temperature_c = room_temperature_c
        self._outdoor_temperature_c = outdoor_temperature_c
        self._hp_supply_temperature_c = hp_supply_temperature_c
        self._hp_supply_target_temperature_c = hp_supply_target_temperature_c
        self._hp_return_temperature_c = hp_return_temperature_c
        self._hp_flow_lpm = hp_flow_lpm
        self._hp_electric_power_kw = hp_electric_power_kw
        self._hp_mode = hp_mode
        self._p1_net_power_kw = p1_net_power_kw
        self._pv_output_kw = pv_output_kw
        self._thermostat_setpoint_c = thermostat_setpoint_c
        self._dhw_top_temperature_c = dhw_top_temperature_c
        self._dhw_bottom_temperature_c = dhw_bottom_temperature_c
        self._shutter_living_room_pct = shutter_living_room_pct
        self._defrost_active = defrost_active
        self._booster_heater_active = booster_heater_active
        self._boiler_ambient_temp_c = boiler_ambient_temp_c
        self._refrigerant_condensation_temp_c = refrigerant_condensation_temp_c
        self._refrigerant_temp_c = refrigerant_temp_c
        self._t_mains_estimated_c = t_mains_estimated_c

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_json_file(
        cls,
        path: str | Path,
    ) -> "LocalBackend":
        """Create a backend that re-reads the JSON file on every telemetry sample."""
        resolved_path = Path(path)

        def _read_json_object() -> dict[str, object]:
            if not resolved_path.exists():
                raise FileNotFoundError(
                    f"Sensor file not found: {resolved_path.resolve()}\n"
                    "Create it with all required telemetry keys."
                )
            try:
                data = json.loads(resolved_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {resolved_path}: {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"JSON root in {resolved_path} must be an object.")
            return data

        def _read_numeric(key: str) -> float:
            data = _read_json_object()
            if key not in data:
                raise ValueError(f"Missing required key {key!r} in {resolved_path}.")
            value = data[key]
            try:
                return float(str(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Key {key!r} in {resolved_path} is not numeric: {value!r}"
                ) from exc

        def _read_bool(key: str) -> bool:
            data = _read_json_object()
            if key not in data:
                raise ValueError(f"Missing required key {key!r} in {resolved_path}.")
            return bool(data[key])

        def _read_text(key: str) -> str:
            data = _read_json_object()
            if key not in data:
                raise ValueError(f"Missing required key {key!r} in {resolved_path}.")
            value = str(data[key]).strip()
            if not value:
                raise ValueError(f"Key {key!r} in {resolved_path} must not be empty.")
            return value

        return cls(
            room_temperature_c=lambda: _read_numeric("room_temperature_c"),
            outdoor_temperature_c=lambda: _read_numeric("outdoor_temperature_c"),
            hp_supply_temperature_c=lambda: _read_numeric("hp_supply_temperature_c"),
            hp_supply_target_temperature_c=lambda: _read_numeric("hp_supply_target_temperature_c"),
            hp_return_temperature_c=lambda: _read_numeric("hp_return_temperature_c"),
            hp_flow_lpm=lambda: _read_numeric("hp_flow_lpm"),
            hp_electric_power_kw=lambda: _read_numeric("hp_electric_power_kw"),
            hp_mode=lambda: _read_text("hp_mode"),
            p1_net_power_kw=lambda: _read_numeric("p1_net_power_kw"),
            pv_output_kw=lambda: _read_numeric("pv_output_kw"),
            thermostat_setpoint_c=lambda: _read_numeric("thermostat_setpoint_c"),
            dhw_top_temperature_c=lambda: _read_numeric("dhw_top_temperature_c"),
            dhw_bottom_temperature_c=lambda: _read_numeric("dhw_bottom_temperature_c"),
            shutter_living_room_pct=lambda: _read_numeric("shutter_living_room_pct"),
            defrost_active=lambda: _read_bool("defrost_active"),
            booster_heater_active=lambda: _read_bool("booster_heater_active"),
            boiler_ambient_temp_c=lambda: _read_numeric("boiler_ambient_temp_c"),
            refrigerant_condensation_temp_c=lambda: _read_numeric("refrigerant_condensation_temp_c"),
            refrigerant_temp_c=lambda: _read_numeric("refrigerant_temp_c"),
            t_mains_estimated_c=lambda: _read_numeric("t_mains_estimated_c"),
        )

    @classmethod
    def from_env(cls) -> "LocalBackend":
        """Create a backend that re-reads environment variables on every sample."""

        def _env_numeric(key: str) -> float:
            if key not in os.environ:
                raise ValueError(f"Missing required environment variable {key}.")
            return float(os.environ[key])

        def _env_bool(key: str) -> bool:
            if key not in os.environ:
                raise ValueError(f"Missing required environment variable {key}.")
            return bool(int(os.environ[key]))

        def _env_text(key: str) -> str:
            value = os.environ.get(key, "").strip()
            if not value:
                raise ValueError(f"Missing required environment variable {key}.")
            return value

        return cls(
            room_temperature_c=lambda: _env_numeric(cls.ENV_ROOM_TEMPERATURE_C),
            outdoor_temperature_c=lambda: _env_numeric(cls.ENV_OUTDOOR_TEMPERATURE_C),
            hp_supply_temperature_c=lambda: _env_numeric(cls.ENV_HP_SUPPLY_TEMPERATURE_C),
            hp_supply_target_temperature_c=lambda: _env_numeric(cls.ENV_HP_SUPPLY_TARGET_TEMPERATURE_C),
            hp_return_temperature_c=lambda: _env_numeric(cls.ENV_HP_RETURN_TEMPERATURE_C),
            hp_flow_lpm=lambda: _env_numeric(cls.ENV_HP_FLOW_LPM),
            hp_electric_power_kw=lambda: _env_numeric(cls.ENV_HP_ELECTRIC_POWER_KW),
            hp_mode=lambda: _env_text(cls.ENV_HP_MODE),
            p1_net_power_kw=lambda: _env_numeric(cls.ENV_P1_NET_POWER_KW),
            pv_output_kw=lambda: _env_numeric(cls.ENV_PV_OUTPUT_KW),
            thermostat_setpoint_c=lambda: _env_numeric(cls.ENV_THERMOSTAT_SETPOINT_C),
            dhw_top_temperature_c=lambda: _env_numeric(cls.ENV_DHW_TOP_TEMPERATURE_C),
            dhw_bottom_temperature_c=lambda: _env_numeric(cls.ENV_DHW_BOTTOM_TEMPERATURE_C),
            shutter_living_room_pct=lambda: _env_numeric(cls.ENV_SHUTTER_LIVING_ROOM_PCT),
            defrost_active=lambda: _env_bool(cls.ENV_DEFROST_ACTIVE),
            booster_heater_active=lambda: _env_bool(cls.ENV_BOOSTER_HEATER_ACTIVE),
            boiler_ambient_temp_c=lambda: _env_numeric(cls.ENV_BOILER_AMBIENT_TEMP_C),
            refrigerant_condensation_temp_c=lambda: _env_numeric(cls.ENV_REFRIGERANT_CONDENSATION_TEMP_C),
            refrigerant_temp_c=lambda: _env_numeric(cls.ENV_REFRIGERANT_TEMP_C),
            t_mains_estimated_c=lambda: _env_numeric(cls.ENV_T_MAINS_ESTIMATED_C),
        )

    # ------------------------------------------------------------------
    # SensorBackend interface
    # ------------------------------------------------------------------

    def read_all(self) -> LiveReadings:
        """Read one complete telemetry snapshot.

        The backend resolves all configured sources at call time so tests and
        local scripts can mutate JSON files or environment variables while the
        collector is running.
        """
        return LiveReadings(
            room_temperature_c=_resolve_numeric(self._room_temperature_c),
            outdoor_temperature_c=_resolve_numeric(self._outdoor_temperature_c),
            hp_supply_temperature_c=_resolve_numeric(self._hp_supply_temperature_c),
            hp_supply_target_temperature_c=_resolve_numeric(self._hp_supply_target_temperature_c),
            hp_return_temperature_c=_resolve_numeric(self._hp_return_temperature_c),
            hp_flow_lpm=_resolve_numeric(self._hp_flow_lpm),
            hp_electric_power_kw=_resolve_numeric(self._hp_electric_power_kw),
            hp_mode=_resolve_text(self._hp_mode),
            p1_net_power_kw=_resolve_numeric(self._p1_net_power_kw),
            pv_output_kw=_resolve_numeric(self._pv_output_kw),
            thermostat_setpoint_c=_resolve_numeric(self._thermostat_setpoint_c),
            dhw_top_temperature_c=_resolve_numeric(self._dhw_top_temperature_c),
            dhw_bottom_temperature_c=_resolve_numeric(self._dhw_bottom_temperature_c),
            shutter_living_room_pct=_resolve_numeric(self._shutter_living_room_pct),
            defrost_active=_resolve_bool(self._defrost_active),
            booster_heater_active=_resolve_bool(self._booster_heater_active),
            boiler_ambient_temp_c=_resolve_numeric(self._boiler_ambient_temp_c),
            refrigerant_condensation_temp_c=_resolve_numeric(self._refrigerant_condensation_temp_c),
            refrigerant_temp_c=_resolve_numeric(self._refrigerant_temp_c),
            t_mains_estimated_c=_resolve_numeric(self._t_mains_estimated_c),
            timestamp=self.now_utc(),
        )

    def close(self) -> None:
        """Release backend resources.

        The local backend has no external resources; the method exists to
        satisfy the common :class:`SensorBackend` lifecycle contract.
        """

