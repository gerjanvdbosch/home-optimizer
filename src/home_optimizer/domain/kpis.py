from __future__ import annotations

from .models import DomainModel


class DailyKpis(DomainModel):
    hp_electric_kwh: float | None = None
    total_import_kwh: float | None = None
    total_export_kwh: float | None = None
    pv_generation_kwh: float | None = None
    self_consumption_ratio: float | None = None
    electricity_cost_eur: float | None = None
    room_temperature_mae_c: float | None = None
    room_comfort_violation_degree_hours: float | None = None
    dhw_comfort_violation_minutes: float | None = None
    thermostat_setpoint_changes: int = 0
    compressor_starts: int = 0
