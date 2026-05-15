from __future__ import annotations

from datetime import datetime, timedelta

from home_optimizer.domain.forecast import ForecastEntry
from home_optimizer.domain.pricing import ElectricityPricingConfig
from home_optimizer.domain.pricing import PriceInterval
from home_optimizer.domain.target_schedule import TemperatureTargetWindow
from home_optimizer.domain.time import ensure_utc
from home_optimizer.features.dataset.ports import DatasetSampleFrameReader
from home_optimizer.features.modeling import RoomRcModel, StoredModelVersion, TrainedLinearRoomModel
from home_optimizer.features.mpc.controller_service import SpaceHeatingMpcControllerService
from home_optimizer.features.mpc.exogenous_features import (
    solar_gain_proxy_to_kw,
    trailing_exp_filter,
)
from home_optimizer.features.mpc.models import (
    ControlModelConversionOptions,
    MpcConstraints,
    MpcControllerRequest,
    MpcHorizonBuildRequest,
    MpcInitialState,
    MpcObjectiveWeights,
    MpcPlan,
    Rc2StateMpcInitialState,
)
from home_optimizer.features.mpc.ports import ActiveRoomModelReaderPort
from home_optimizer.features.mpc.preparation import SpaceHeatingMpcPreparationService


class SpaceHeatingMpcPlanningService:
    def __init__(
        self,
        *,
        samples_reader: DatasetSampleFrameReader,
        active_room_model_reader: ActiveRoomModelReaderPort,
        target_schedule: list[TemperatureTargetWindow],
        electricity_pricing: ElectricityPricingConfig | None = None,
        default_interval_minutes: int = 10,
        controller: SpaceHeatingMpcControllerService | None = None,
    ) -> None:
        self.samples_reader = samples_reader
        self.active_room_model_reader = active_room_model_reader
        self.target_schedule = list(target_schedule)
        self.default_interval_minutes = default_interval_minutes
        self.controller = controller or SpaceHeatingMpcControllerService()
        self.preparation = SpaceHeatingMpcPreparationService(
            samples_reader=samples_reader,
            active_room_model_reader=active_room_model_reader,
            target_schedule=target_schedule,
            electricity_pricing=electricity_pricing,
        )

    def plan(
        self,
        *,
        start_time_utc: datetime,
        model_id: str | None = None,
        interval_minutes: int | None = None,
        horizon_steps: int = 36,
        constraints: MpcConstraints | None = None,
        objective_weights: MpcObjectiveWeights | None = None,
        default_effective_heating_kw: float | None = None,
        max_solver_seconds: float | None = None,
        conversion_options: ControlModelConversionOptions | None = None,
    ) -> MpcPlan:
        version = self._resolve_room_model_version(model_id)
        if version is None:
            if model_id:
                raise ValueError(f"Unknown room model version: {model_id}")
            raise ValueError("No active room model is available for MPC planning")

        source_model = version.model
        if not isinstance(source_model, (TrainedLinearRoomModel, RoomRcModel)):
            raise ValueError("Active room model has an unsupported type for MPC planning")

        resolved_start_time = ensure_utc(start_time_utc)
        resolved_interval_minutes = interval_minutes or self.default_interval_minutes
        resolved_constraints = constraints or MpcConstraints()
        resolved_weights = objective_weights or MpcObjectiveWeights()

        history_rows = (
            max(
                resolved_constraints.min_on_steps,
                resolved_constraints.min_off_steps,
            )
            + 1
        )
        history_rows = self.preparation.history_rows_for_initial_state(
            source_model,
            minimum_history_rows=history_rows,
        )
        initial_rows = self._load_initial_rows(
            start_time_utc=resolved_start_time,
            interval_minutes=resolved_interval_minutes,
            history_rows=history_rows,
        )
        initial_state = self._initial_state_from_rows(
            initial_rows,
            source_model=source_model,
        )
        effective_heating_kw = self._resolve_effective_heating_kw(
            initial_rows,
            fallback_kw=default_effective_heating_kw,
        )
        hp_electric_power_kw = self._resolve_hp_electric_power_kw(
            initial_rows,
            fallback_kw=effective_heating_kw,
        )
        export_price_eur_kwh = self._resolve_export_price_eur_kwh(initial_rows)
        base_load_power_kw = self._resolve_base_load_power_kw(initial_rows)
        forecast_entries = self._load_forecast_entries(
            start_time_utc=resolved_start_time,
            interval_minutes=resolved_interval_minutes,
            horizon_steps=horizon_steps,
        )
        pv_power_input_scale = self._infer_pv_power_input_scale(
            initial_rows,
            forecast_entries=forecast_entries,
            start_time_utc=resolved_start_time,
        )
        initial_filtered_solar_gain_kw = 0.0
        solar_gain_filter_alpha = 0.0
        local_timezone: str | None = None
        if isinstance(source_model, RoomRcModel):
            solar_gain_history_kw = [
                self._row_solar_gain_kw(row, source_model=source_model)
                for row in initial_rows
                if row.timestamp_utc < resolved_start_time
            ]
            solar_gain_filter_alpha = source_model.config.alpha_solar
            initial_filtered_solar_gain_kw = trailing_exp_filter(
                solar_gain_history_kw,
                alpha=solar_gain_filter_alpha,
            )
            local_timezone = source_model.config.local_timezone

        horizon = self.controller.build_horizon(
            MpcHorizonBuildRequest(
                start_time_utc=resolved_start_time,
                horizon_steps=horizon_steps,
                interval_minutes=resolved_interval_minutes,
                target_schedule=self.target_schedule,
                forecast_entries=forecast_entries,
                price_intervals=self._load_price_intervals(
                    start_time_utc=resolved_start_time,
                    interval_minutes=resolved_interval_minutes,
                    horizon_steps=horizon_steps,
                ),
                default_effective_heating_kw=effective_heating_kw,
                default_hp_electric_power_kw=hp_electric_power_kw,
                default_base_load_power_kw=base_load_power_kw,
                default_export_price_eur_kwh=export_price_eur_kwh,
                pv_power_input_scale=pv_power_input_scale,
                solar_gain_filter_alpha=solar_gain_filter_alpha,
                initial_filtered_solar_gain_kw=initial_filtered_solar_gain_kw,
                local_timezone=local_timezone,
            )
        )

        return self.controller.plan_from_source_model(
            MpcControllerRequest(
                interval_minutes=resolved_interval_minutes,
                horizon=horizon,
                constraints=resolved_constraints,
                objective_weights=resolved_weights,
                max_solver_seconds=max_solver_seconds,
            ),
            source_model=source_model,
            initial_state=initial_state,
            horizon=horizon,
            conversion_options=conversion_options,
        )

    def _resolve_room_model_version(self, model_id: str | None) -> StoredModelVersion | None:
        return self.preparation.resolve_room_model_version(model_id)

    def _load_initial_rows(
        self,
        *,
        start_time_utc: datetime,
        interval_minutes: int,
        history_rows: int,
    ):
        return self.preparation.load_initial_rows(
            start_time_utc=start_time_utc,
            interval_minutes=interval_minutes,
            history_rows=history_rows,
        )

    def _initial_state_from_rows(
        self,
        rows,
        *,
        source_model: TrainedLinearRoomModel | RoomRcModel,
    ) -> MpcInitialState | Rc2StateMpcInitialState:
        return self.preparation.initial_state_from_rows(rows, source_model=source_model)

    def _resolve_effective_heating_kw(
        self,
        rows,
        *,
        fallback_kw: float | None,
    ) -> float:
        return self.preparation.resolve_effective_heating_kw(rows, fallback_kw=fallback_kw)

    def _resolve_hp_electric_power_kw(
        self,
        rows,
        *,
        fallback_kw: float,
    ) -> float:
        return self.preparation.resolve_hp_electric_power_kw(rows, fallback_kw=fallback_kw)

    def _resolve_export_price_eur_kwh(self, rows) -> float:
        return self.preparation.resolve_export_price_eur_kwh(rows)

    def _resolve_base_load_power_kw(self, rows) -> float:
        return self.preparation.resolve_base_load_power_kw(rows)

    def _infer_pv_power_input_scale(
        self,
        rows,
        *,
        forecast_entries: list[ForecastEntry],
        start_time_utc: datetime,
    ) -> float:
        return self.preparation.infer_pv_power_input_scale(
            rows,
            forecast_entries=forecast_entries,
            start_time_utc=start_time_utc,
        )

    def _load_forecast_entries(
        self,
        *,
        start_time_utc: datetime,
        interval_minutes: int,
        horizon_steps: int,
    ) -> list[ForecastEntry]:
        return self.preparation.load_forecast_entries(
            start_time_utc=start_time_utc,
            interval_minutes=interval_minutes,
            horizon_steps=horizon_steps,
        )

    def _load_price_intervals(
        self,
        *,
        start_time_utc: datetime,
        interval_minutes: int,
        horizon_steps: int,
    ) -> list[PriceInterval]:
        return self.preparation.load_price_intervals(
            start_time_utc=start_time_utc,
            interval_minutes=interval_minutes,
            horizon_steps=horizon_steps,
        )

    @staticmethod
    def _row_solar_gain_kw(
        row,
        *,
        source_model: RoomRcModel,
    ) -> float:
        solar_gain_proxy_w_m2 = row.solar_gain_proxy_w_m2
        if solar_gain_proxy_w_m2 is None:
            irradiance_w_m2 = float(row.solar_irradiance_w_m2 or 0.0)
            shutter_fraction = 1.0
            if row.shutter_position_pct is not None:
                shutter_fraction = min(max(float(row.shutter_position_pct) / 100.0, 0.0), 1.0)
            solar_gain_proxy_w_m2 = irradiance_w_m2 * shutter_fraction
        return float(
            solar_gain_proxy_to_kw(
                float(solar_gain_proxy_w_m2 or 0.0),
                glass_area_m2=source_model.config.glass_area_m2,
                g_glass=source_model.config.g_glass,
            )
        )
