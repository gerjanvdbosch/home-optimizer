from __future__ import annotations

from datetime import datetime

from home_optimizer.domain.models import DomainModel


class IdentificationDatasetRow(DomainModel):
    timestamp_utc: datetime
    room_temperature_c: float | None = None
    outdoor_temperature_c: float | None = None
    dhw_top_temperature_c: float | None = None
    dhw_bottom_temperature_c: float | None = None
    hp_electric_power_kw: float | None = None
    hp_mode_raw: str | None = None
    mode_space: int = 0
    mode_dhw: int = 0
    mode_off: int = 1
    pv_output_power_kw: float | None = None
    net_power_kw: float | None = None
    shutter_position_pct: float | None = None
    thermostat_setpoint_c: float | None = None
    room_target_temperature_c: float | None = None
    room_target_min_temperature_c: float | None = None
    room_target_max_temperature_c: float | None = None
    supply_temperature_c: float | None = None
    return_temperature_c: float | None = None
    flow_l_min: float | None = None
    hp_delta_t_c: float | None = None
    thermal_output_estimate_kw: float | None = None
    cop_estimate: float | None = None
    solar_irradiance_w_m2: float | None = None
    solar_gain_proxy_w_m2: float | None = None
    price_import_eur_kwh: float | None = None
    price_export_eur_kwh: float | None = None
    occupied_flag: int = 0
    dhw_draw_detected: int = 0


class IdentificationDataset(DomainModel):
    interval_minutes: int
    start_time_utc: datetime
    end_time_utc: datetime
    rows: list[IdentificationDatasetRow]
