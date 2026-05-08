from __future__ import annotations

from datetime import datetime, timedelta

from home_optimizer.app.settings import AppSettings
from home_optimizer.domain import NumericPoint, NumericSeries, normalize_utc_timestamp
from home_optimizer.domain.time import ensure_utc
from home_optimizer.features.dataset import MpcDatasetService
from home_optimizer.features.modeling import RoomModelingService, TrainedLinearRoomModel
from home_optimizer.features.simulation.models import RoomSimulationResult


def _empty_series(name: str, unit: str | None) -> NumericSeries:
    return NumericSeries(name=name, unit=unit, points=[])


class RoomSimulationService:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.modeling_service = RoomModelingService()

    def simulate_room(
        self,
        *,
        reader,
        model_id: str,
        model: TrainedLinearRoomModel,
        anchor_time: datetime,
        horizon_steps: int,
    ) -> RoomSimulationResult:
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be greater than zero")

        anchor_time_utc = ensure_utc(anchor_time)
        interval = timedelta(minutes=model.interval_minutes)
        max_lag = max(
            model.config.room_temperature_lags
            + model.config.outdoor_temperature_lags
            + model.config.thermal_output_lags
            + model.config.solar_gain_lags
            + model.config.occupied_flag_lags
        )
        start_time = anchor_time_utc - (interval * max_lag)
        end_time = anchor_time_utc + (interval * (horizon_steps + 1))

        dataset = MpcDatasetService(reader, self.settings).build_dataset(
            start_time=start_time,
            end_time=end_time,
            interval_minutes=model.interval_minutes,
        )
        anchor_index = next(
            (
                index
                for index, row in enumerate(dataset.rows)
                if row.timestamp_utc == anchor_time_utc
            ),
            None,
        )
        if anchor_index is None:
            raise ValueError("anchor_time does not align with dataset interval")
        if anchor_index < max_lag:
            raise ValueError("not enough history before anchor_time for simulation")
        if anchor_index + horizon_steps >= len(dataset.rows):
            raise ValueError("not enough future rows available for requested horizon")

        simulated = self.modeling_service.simulate_horizon(
            model,
            dataset.rows,
            start_index=anchor_index,
            horizon_steps=horizon_steps,
        )
        if len(simulated) != horizon_steps:
            raise ValueError("could not simulate full room horizon")

        predicted_points: list[NumericPoint] = []
        actual_points: list[NumericPoint] = []
        target_min_points: list[NumericPoint] = []
        target_max_points: list[NumericPoint] = []
        outdoor_points: list[NumericPoint] = []
        solar_points: list[NumericPoint] = []

        for step in range(1, horizon_steps + 1):
            row = dataset.rows[anchor_index + step]
            timestamp = normalize_utc_timestamp(row.timestamp_utc)
            predicted_points.append(NumericPoint(timestamp=timestamp, value=simulated[step - 1]))
            if row.room_temperature_c is not None:
                actual_points.append(
                    NumericPoint(timestamp=timestamp, value=row.room_temperature_c)
                )
            if row.room_target_min_temperature_c is not None:
                target_min_points.append(
                    NumericPoint(timestamp=timestamp, value=row.room_target_min_temperature_c)
                )
            if row.room_target_max_temperature_c is not None:
                target_max_points.append(
                    NumericPoint(timestamp=timestamp, value=row.room_target_max_temperature_c)
                )
            if row.outdoor_temperature_c is not None:
                outdoor_points.append(
                    NumericPoint(timestamp=timestamp, value=row.outdoor_temperature_c)
                )
            if row.solar_irradiance_w_m2 is not None:
                solar_points.append(
                    NumericPoint(timestamp=timestamp, value=row.solar_irradiance_w_m2)
                )

        return RoomSimulationResult(
            model_id=model_id,
            anchor_time_utc=anchor_time_utc,
            interval_minutes=model.interval_minutes,
            horizon_steps=horizon_steps,
            predicted_room_temperature=NumericSeries(
                name="predicted_room_temperature",
                unit="°C",
                points=predicted_points,
            ),
            actual_room_temperature=NumericSeries(
                name="actual_room_temperature",
                unit="°C",
                points=actual_points,
            ),
            room_target_min_temperature=NumericSeries(
                name="room_target_min_temperature",
                unit="°C",
                points=target_min_points,
            ),
            room_target_max_temperature=NumericSeries(
                name="room_target_max_temperature",
                unit="°C",
                points=target_max_points,
            ),
            outdoor_temperature=NumericSeries(
                name="outdoor_temperature",
                unit="°C",
                points=outdoor_points,
            ),
            solar_irradiance=NumericSeries(
                name="solar_irradiance",
                unit="W/m2",
                points=solar_points,
            ),
        )
