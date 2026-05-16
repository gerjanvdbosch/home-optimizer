from __future__ import annotations

from datetime import datetime, timedelta
from dataclasses import dataclass

from home_optimizer.domain import (
    FORECAST_TEMPERATURE,
    GTI_LIVING_ROOM_WINDOWS,
    GTI_LIVING_ROOM_WINDOWS_ADJUSTED,
    GTI_PV,
)
from home_optimizer.domain.forecast import ForecastEntry
from home_optimizer.domain.pricing import ElectricityPricingConfig
from home_optimizer.domain.target_schedule import TemperatureTargetWindow
from home_optimizer.domain.time import ensure_utc
from home_optimizer.features.backtest.models import MpcBacktestResult
from home_optimizer.features.backtest.runner import SpaceHeatingMpcBacktestRunner
from home_optimizer.features.dataset.models import MpcDatasetRow
from home_optimizer.features.dataset.ports import DatasetSampleFrameReader
from home_optimizer.features.modeling import RoomRcModel, TrainedLinearRoomModel
from home_optimizer.features.mpc.horizon_builder import MpcHorizonBuilder
from home_optimizer.features.mpc.control_model import to_control_model
from home_optimizer.features.mpc.exogenous_features import (
    continue_exp_filter,
    local_hour_sin_cos,
    solar_gain_proxy_to_kw,
    trailing_exp_filter,
)
from home_optimizer.features.mpc.models import (
    ControlModelConversionOptions,
    MpcConstraints,
    MpcHorizonBuildRequest,
    MpcHorizonStep,
    MpcObjectiveWeights,
)
from home_optimizer.features.mpc.ports import ActiveRoomModelReaderPort
from home_optimizer.features.mpc.preparation import SpaceHeatingMpcPreparationService


@dataclass(frozen=True)
class BacktestForecastReplayHorizon:
    horizon: list[MpcHorizonStep]
    forecast_issue_time_utc: datetime
    forecast_age_minutes: float
    forecast_coverage_ratio: float
    missing_forecast_count: int


@dataclass(frozen=True)
class BacktestForecastCoverage:
    coverage_ratio: float
    missing_forecast_count: int
    latest_created_at_utc: datetime | None
    missing_details: list[str]


class BacktestForecastReplayProvider:
    def __init__(
        self,
        *,
        preparation: SpaceHeatingMpcPreparationService,
        target_schedule: list[TemperatureTargetWindow],
        source_model: TrainedLinearRoomModel | RoomRcModel,
        all_rows: list[MpcDatasetRow],
        effective_heating_kw: float,
        hp_electric_power_kw: float,
        export_price_eur_kwh: float,
    ) -> None:
        self.preparation = preparation
        self.target_schedule = list(target_schedule)
        self.source_model = source_model
        self.all_rows = list(all_rows)
        self.effective_heating_kw = effective_heating_kw
        self.hp_electric_power_kw = hp_electric_power_kw
        self.export_price_eur_kwh = export_price_eur_kwh
        self.horizon_builder = MpcHorizonBuilder()

    def get_forecast_horizon(
        self,
        issue_time_utc: datetime,
        horizon_steps: int,
        interval_minutes: int,
    ) -> BacktestForecastReplayHorizon:
        resolved_issue_time = ensure_utc(issue_time_utc)
        history_rows = [
            row for row in self.all_rows if row.timestamp_utc < resolved_issue_time
        ]
        forecast_start_time_utc = (
            history_rows[0].timestamp_utc
            if history_rows
            else resolved_issue_time
        )
        forecast_end_steps = max(
            int(
                (
                    (resolved_issue_time - forecast_start_time_utc).total_seconds()
                    / 60.0
                )
                / interval_minutes
            ),
            0,
        ) + horizon_steps
        forecast_entries = self.preparation.load_forecast_entries(
            start_time_utc=forecast_start_time_utc,
            interval_minutes=interval_minutes,
            horizon_steps=forecast_end_steps,
            created_at_end_time=resolved_issue_time + timedelta(microseconds=1),
        )
        forecast_entries = self._with_derived_adjusted_solar_forecast(
            forecast_entries,
            issue_time_utc=resolved_issue_time,
            horizon_steps=horizon_steps,
            interval_minutes=interval_minutes,
            history_rows=history_rows,
        )
        coverage = self._forecast_coverage(
            forecast_entries,
            issue_time_utc=resolved_issue_time,
            horizon_steps=horizon_steps,
            interval_minutes=interval_minutes,
        )
        if coverage.missing_forecast_count > 0 or coverage.latest_created_at_utc is None:
            raise ValueError(
                "missing_forecast: archived forecast coverage is incomplete for "
                f"forecast_replay at issue_time={resolved_issue_time.isoformat()} "
                f"coverage_ratio={coverage.coverage_ratio:.3f} "
                f"latest_created_at_utc="
                f"{coverage.latest_created_at_utc.isoformat() if coverage.latest_created_at_utc is not None else 'none'} "
                f"missing={'; '.join(coverage.missing_details[:12])}"
            )

        base_load_forecast_kw = self.preparation.resolve_base_load_power_kw(
            history_rows or self.all_rows[:1]
        )
        pv_power_input_scale = self.preparation.infer_pv_power_input_scale(
            history_rows or self.all_rows[:1],
            forecast_entries=forecast_entries,
            start_time_utc=resolved_issue_time,
        )
        initial_filtered_solar_gain_kw = 0.0
        solar_gain_filter_alpha = 0.0
        local_timezone: str | None = None
        if isinstance(self.source_model, RoomRcModel):
            solar_gain_history_kw = [
                SpaceHeatingMpcBacktestService._solar_gain_input(
                    row,
                    source_model=self.source_model,
                )
                for row in history_rows
            ]
            solar_gain_filter_alpha = self.source_model.config.alpha_solar
            initial_filtered_solar_gain_kw = trailing_exp_filter(
                solar_gain_history_kw,
                alpha=solar_gain_filter_alpha,
            )
            local_timezone = self.source_model.config.local_timezone

        horizon = self.horizon_builder.build(
            MpcHorizonBuildRequest(
                start_time_utc=resolved_issue_time,
                horizon_steps=horizon_steps,
                interval_minutes=interval_minutes,
                target_schedule=self.target_schedule,
                forecast_entries=forecast_entries,
                price_intervals=self.preparation.load_price_intervals(
                    start_time_utc=resolved_issue_time,
                    interval_minutes=interval_minutes,
                    horizon_steps=horizon_steps,
                ),
                default_effective_heating_kw=self.effective_heating_kw,
                default_hp_electric_power_kw=self.hp_electric_power_kw,
                default_base_load_power_kw=base_load_forecast_kw,
                default_export_price_eur_kwh=self.export_price_eur_kwh,
                pv_power_input_scale=pv_power_input_scale,
                solar_gain_filter_alpha=solar_gain_filter_alpha,
                initial_filtered_solar_gain_kw=initial_filtered_solar_gain_kw,
                local_timezone=local_timezone,
            )
        )
        forecast_age_minutes = max(
            (resolved_issue_time - coverage.latest_created_at_utc).total_seconds() / 60.0,
            0.0,
        )
        return BacktestForecastReplayHorizon(
            horizon=horizon,
            forecast_issue_time_utc=coverage.latest_created_at_utc,
            forecast_age_minutes=forecast_age_minutes,
            forecast_coverage_ratio=coverage.coverage_ratio,
            missing_forecast_count=coverage.missing_forecast_count,
        )

    @staticmethod
    def _forecast_coverage(
        forecast_entries: list[ForecastEntry],
        *,
        issue_time_utc: datetime,
        horizon_steps: int,
        interval_minutes: int,
    ) -> BacktestForecastCoverage:
        selected_keys = {
            (entry.name, ensure_utc(entry.forecast_time_utc))
            for entry in forecast_entries
        }
        expected_timestamps = [
            issue_time_utc + timedelta(minutes=interval_minutes * step)
            for step in range(horizon_steps)
        ]
        missing_details: list[str] = []
        actual_count = 0
        expected_count = horizon_steps * 3
        for forecast_time_utc in expected_timestamps:
            if (FORECAST_TEMPERATURE, forecast_time_utc) in selected_keys:
                actual_count += 1
            else:
                missing_details.append(
                    f"{FORECAST_TEMPERATURE}@{forecast_time_utc.isoformat()}"
                )
            if (GTI_PV, forecast_time_utc) in selected_keys:
                actual_count += 1
            else:
                missing_details.append(f"{GTI_PV}@{forecast_time_utc.isoformat()}")
            if (
                (GTI_LIVING_ROOM_WINDOWS_ADJUSTED, forecast_time_utc) in selected_keys
                or (GTI_LIVING_ROOM_WINDOWS, forecast_time_utc) in selected_keys
            ):
                actual_count += 1
            else:
                missing_details.append(
                    f"{GTI_LIVING_ROOM_WINDOWS}@{forecast_time_utc.isoformat()}"
                )
        missing_count = expected_count - actual_count
        latest_created = max(
            (
                ensure_utc(entry.created_at_utc)
                for entry in forecast_entries
                if entry.name
                in {
                    FORECAST_TEMPERATURE,
                    GTI_PV,
                    GTI_LIVING_ROOM_WINDOWS,
                    GTI_LIVING_ROOM_WINDOWS_ADJUSTED,
                }
            ),
            default=None,
        )
        coverage_ratio = (actual_count / expected_count) if expected_count > 0 else 1.0
        return BacktestForecastCoverage(
            coverage_ratio=coverage_ratio,
            missing_forecast_count=missing_count,
            latest_created_at_utc=latest_created,
            missing_details=missing_details,
        )

    def _with_derived_adjusted_solar_forecast(
        self,
        forecast_entries: list[ForecastEntry],
        *,
        issue_time_utc: datetime,
        horizon_steps: int,
        interval_minutes: int,
        history_rows: list[MpcDatasetRow],
    ) -> list[ForecastEntry]:
        expected_forecast_times = {
            issue_time_utc + timedelta(minutes=interval_minutes * step)
            for step in range(horizon_steps)
        }
        existing_adjusted_keys = {
            (entry.name, ensure_utc(entry.forecast_time_utc))
            for entry in forecast_entries
            if entry.name == GTI_LIVING_ROOM_WINDOWS_ADJUSTED
            and ensure_utc(entry.forecast_time_utc) in expected_forecast_times
        }
        if len(existing_adjusted_keys) >= len(expected_forecast_times):
            return forecast_entries

        shutter_open_fraction = self._latest_shutter_open_fraction(history_rows)
        derived_entries: list[ForecastEntry] = []
        for entry in forecast_entries:
            if entry.name != GTI_LIVING_ROOM_WINDOWS:
                continue
            forecast_time_utc = ensure_utc(entry.forecast_time_utc)
            key = (GTI_LIVING_ROOM_WINDOWS_ADJUSTED, forecast_time_utc)
            if key in existing_adjusted_keys:
                continue
            derived_entries.append(
                ForecastEntry(
                    created_at_utc=entry.created_at_utc,
                    forecast_time_utc=entry.forecast_time_utc,
                    name=GTI_LIVING_ROOM_WINDOWS_ADJUSTED,
                    value=float(entry.value) * shutter_open_fraction,
                    unit=entry.unit,
                    source=entry.source,
                )
            )
        if not derived_entries:
            return forecast_entries
        return [*forecast_entries, *derived_entries]

    @staticmethod
    def _latest_shutter_open_fraction(history_rows: list[MpcDatasetRow]) -> float:
        latest_position_pct: float | None = None
        for row in history_rows:
            if row.shutter_position_pct is None:
                continue
            latest_position_pct = float(row.shutter_position_pct)
        if latest_position_pct is None:
            return 1.0
        return min(max(latest_position_pct / 100.0, 0.0), 1.0)


class SpaceHeatingMpcBacktestService:
    def __init__(
        self,
        *,
        samples_reader: DatasetSampleFrameReader,
        active_room_model_reader: ActiveRoomModelReaderPort,
        target_schedule: list[TemperatureTargetWindow],
        electricity_pricing: ElectricityPricingConfig | None = None,
        default_interval_minutes: int = 10,
        runner: SpaceHeatingMpcBacktestRunner | None = None,
    ) -> None:
        self.default_interval_minutes = default_interval_minutes
        self.target_schedule = list(target_schedule)
        self.preparation = SpaceHeatingMpcPreparationService(
            samples_reader=samples_reader,
            active_room_model_reader=active_room_model_reader,
            target_schedule=target_schedule,
            electricity_pricing=electricity_pricing,
        )
        self.runner = runner or SpaceHeatingMpcBacktestRunner()

    def run(
        self,
        *,
        start_time_utc: datetime,
        end_time_utc: datetime,
        model_id: str | None = None,
        horizon_steps: int = 36,
        constraints: MpcConstraints | None = None,
        objective_weights: MpcObjectiveWeights | None = None,
        max_solver_seconds: float | None = None,
        conversion_options: ControlModelConversionOptions | None = None,
        default_effective_heating_kw: float | None = None,
        exogenous_mode: str = "perfect_foresight",
    ) -> MpcBacktestResult:
        resolved_start_time = ensure_utc(start_time_utc)
        resolved_end_time = ensure_utc(end_time_utc)
        if resolved_end_time <= resolved_start_time:
            raise ValueError("end_time_utc must be later than start_time_utc")
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be greater than zero")
        if exogenous_mode not in {"perfect_foresight", "forecast_replay"}:
            raise ValueError(
                "exogenous_mode must be one of: perfect_foresight, forecast_replay"
            )

        version = self.preparation.resolve_room_model_version(model_id)
        if version is None:
            if model_id:
                raise ValueError(f"Unknown room model version: {model_id}")
            raise ValueError("No active room model is available for MPC backtesting")

        source_model = version.model
        if not isinstance(source_model, (TrainedLinearRoomModel, RoomRcModel)):
            raise ValueError("Active room model has an unsupported type for MPC backtesting")

        interval_minutes = self.default_interval_minutes
        resolved_constraints = constraints or MpcConstraints()
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
        initial_rows = self.preparation.load_initial_rows(
            start_time_utc=resolved_start_time,
            interval_minutes=interval_minutes,
            history_rows=history_rows,
        )
        initial_state = self.preparation.initial_state_from_rows(
            initial_rows,
            source_model=source_model,
        )
        dataset = self.preparation.build_dataset(
            start_time=resolved_start_time,
            end_time=resolved_end_time + timedelta(minutes=interval_minutes),
            interval_minutes=interval_minutes,
        )
        backtest_rows = [
            row
            for row in dataset.rows
            if resolved_start_time <= row.timestamp_utc <= resolved_end_time
        ]
        if len(backtest_rows) < 2:
            raise ValueError("Backtest requires at least two historical rows in the selected window")
        inference_rows = [*initial_rows, *backtest_rows]
        effective_heating_kw = self.preparation.resolve_effective_heating_kw(
            inference_rows,
            fallback_kw=default_effective_heating_kw,
        )
        hp_electric_power_kw = self.preparation.resolve_hp_electric_power_kw(
            inference_rows,
            fallback_kw=effective_heating_kw,
        )
        export_price_eur_kwh = self.preparation.resolve_export_price_eur_kwh(inference_rows)
        timeline = self._build_timeline(
            backtest_rows,
            initial_rows=initial_rows,
            source_model=source_model,
            effective_heating_kw=effective_heating_kw,
            hp_electric_power_kw=hp_electric_power_kw,
            export_price_eur_kwh=export_price_eur_kwh,
        )
        historical_hp_on_by_timestamp = {
            row.timestamp_utc: self.preparation.row_hp_on(row)
            for row in backtest_rows
        }
        historical_energy_cost_by_timestamp = {
            row.timestamp_utc: self._historical_energy_cost(
                row,
                interval_minutes=interval_minutes,
            )
            for row in backtest_rows
        }
        forecast_replay_provider = None
        if exogenous_mode == "forecast_replay":
            forecast_replay_provider = BacktestForecastReplayProvider(
                preparation=self.preparation,
                target_schedule=self.target_schedule,
                source_model=source_model,
                all_rows=inference_rows,
                effective_heating_kw=effective_heating_kw,
                hp_electric_power_kw=hp_electric_power_kw,
                export_price_eur_kwh=export_price_eur_kwh,
            )
        return self.runner.run(
            model_id=version.model_id,
            model_type=version.model_type,
            control_model=to_control_model(
                source_model,
                options=conversion_options,
            ),
            timeline=timeline,
            initial_state=initial_state,
            interval_minutes=interval_minutes,
            horizon_steps=horizon_steps,
            constraints=resolved_constraints,
            objective_weights=objective_weights,
            max_solver_seconds=max_solver_seconds,
            historical_hp_on_by_timestamp=historical_hp_on_by_timestamp,
            historical_energy_cost_by_timestamp=historical_energy_cost_by_timestamp,
            exogenous_mode=exogenous_mode,
            forecast_replay_provider=forecast_replay_provider,
        )

    def _build_timeline(
        self,
        backtest_rows: list[MpcDatasetRow],
        *,
        initial_rows: list[MpcDatasetRow],
        source_model: TrainedLinearRoomModel | RoomRcModel,
        effective_heating_kw: float,
        hp_electric_power_kw: float,
        export_price_eur_kwh: float,
    ) -> list[MpcHorizonStep]:
        solar_direct_values = [
            self._solar_gain_input(
                row,
                source_model=source_model,
            )
            for row in backtest_rows
        ]
        solar_filtered_values: list[float] = list(solar_direct_values)
        local_timezone: str | None = None
        if isinstance(source_model, RoomRcModel):
            solar_history_kw = [
                self._solar_gain_input(row, source_model=source_model)
                for row in initial_rows
                if row.timestamp_utc < backtest_rows[0].timestamp_utc
            ]
            solar_filtered_values = continue_exp_filter(
                solar_direct_values,
                alpha=source_model.config.alpha_solar,
                initial_filtered_value=trailing_exp_filter(
                    solar_history_kw,
                    alpha=source_model.config.alpha_solar,
                ),
            )
            local_timezone = source_model.config.local_timezone

        timeline: list[MpcHorizonStep] = []
        for index, row in enumerate(backtest_rows):
            room_target_min = row.room_target_min_temperature_c
            room_target_max = row.room_target_max_temperature_c
            if room_target_min is None or room_target_max is None:
                raise ValueError(
                    f"Missing room target bounds for backtest row at {row.timestamp_utc.isoformat()}"
                )
            net_power_kw = float(row.net_power_kw or 0.0)
            pv_output_kw = float(row.pv_output_power_kw or 0.0)
            hp_power_kw = float(row.hp_electric_power_kw or 0.0)
            base_load_power_kw = net_power_kw + pv_output_kw - hp_power_kw
            hour_sin, hour_cos = local_hour_sin_cos(
                row.timestamp_utc,
                local_timezone=local_timezone,
            )
            timeline.append(
                MpcHorizonStep(
                    timestamp_utc=row.timestamp_utc,
                    outdoor_temp_c=float(row.outdoor_temperature_c or 0.0),
                    solar_gain_kw=solar_direct_values[index],
                    solar_gain_mass_kw=solar_filtered_values[index],
                    solar_irradiance_forecast_w_m2=float(row.solar_irradiance_w_m2 or 0.0),
                    solar_irradiance_realized_w_m2=float(row.solar_irradiance_w_m2 or 0.0),
                    solar_gain_realized_kw=solar_direct_values[index],
                    effective_heating_kw_forecast=effective_heating_kw,
                    hp_electric_power_forecast_kw=hp_electric_power_kw,
                    pv_available_power_forecast_kw=pv_output_kw,
                    pv_available_power_realized_kw=pv_output_kw,
                    base_load_power_forecast_kw=base_load_power_kw,
                    base_load_power_realized_kw=base_load_power_kw,
                    occupied=float(row.occupied_flag),
                    hour_sin=hour_sin,
                    hour_cos=hour_cos,
                    target_temp_c=(
                        float(row.room_target_temperature_c)
                        if row.room_target_temperature_c is not None
                        else None
                    ),
                    temp_min_c=float(room_target_min),
                    temp_max_c=float(room_target_max),
                    price_eur_kwh=float(row.price_import_eur_kwh or 0.0),
                    import_price_eur_kwh=float(row.price_import_eur_kwh or 0.0),
                    export_price_eur_kwh=float(row.price_export_eur_kwh or export_price_eur_kwh),
                    realized_room_temp_c=(
                        float(row.room_temperature_c)
                        if row.room_temperature_c is not None
                        else None
                    ),
                )
            )
        return timeline

    @staticmethod
    def _solar_gain_input(
        row: MpcDatasetRow,
        *,
        source_model: TrainedLinearRoomModel | RoomRcModel,
    ) -> float:
        solar_gain_proxy_w_m2 = row.solar_gain_proxy_w_m2
        if solar_gain_proxy_w_m2 is None:
            irradiance_w_m2 = float(row.solar_irradiance_w_m2 or 0.0)
            shutter_fraction = 1.0
            if row.shutter_position_pct is not None:
                shutter_fraction = min(max(float(row.shutter_position_pct) / 100.0, 0.0), 1.0)
            solar_gain_proxy_w_m2 = irradiance_w_m2 * shutter_fraction
        solar_gain_proxy_w_m2 = float(solar_gain_proxy_w_m2 or 0.0)
        if isinstance(source_model, RoomRcModel):
            return solar_gain_proxy_to_kw(
                solar_gain_proxy_w_m2,
                glass_area_m2=source_model.config.glass_area_m2,
                g_glass=source_model.config.g_glass,
            )
        return solar_gain_proxy_w_m2

    @staticmethod
    def _historical_energy_cost(
        row: MpcDatasetRow,
        *,
        interval_minutes: int,
    ) -> float:
        net_power_kw = float(row.net_power_kw or 0.0)
        import_price = float(row.price_import_eur_kwh or 0.0)
        export_price = float(row.price_export_eur_kwh or 0.0)
        grid_import_kw = max(net_power_kw, 0.0)
        grid_export_kw = max(-net_power_kw, 0.0)
        return float(
            ((import_price * grid_import_kw) - (export_price * grid_export_kw))
            * (interval_minutes / 60.0)
        )
