from __future__ import annotations

from datetime import datetime

from home_optimizer.domain.models import DomainModel
from home_optimizer.domain.series import NumericSeries


class RoomSimulationResult(DomainModel):
    model_id: str
    anchor_time_utc: datetime
    interval_minutes: int
    horizon_steps: int
    predicted_room_temperature: NumericSeries
    actual_room_temperature: NumericSeries
    prediction_error_c: NumericSeries
    room_target_min_temperature: NumericSeries
    room_target_max_temperature: NumericSeries
    outdoor_temperature: NumericSeries
    thermal_output_estimate: NumericSeries
    solar_irradiance: NumericSeries
    solar_gain_proxy: NumericSeries
    shutter_position: NumericSeries
