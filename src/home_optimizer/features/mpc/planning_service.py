from __future__ import annotations

from datetime import datetime, timedelta

from home_optimizer.domain import GTI_PV
from home_optimizer.domain.forecast import ForecastEntry
from home_optimizer.domain.pricing import PriceInterval
from home_optimizer.domain.target_schedule import TemperatureTargetWindow
from home_optimizer.domain.time import ensure_utc, parse_datetime
from home_optimizer.features.dataset import MpcDatasetService
from home_optimizer.features.dataset.models import MpcDatasetRow
from home_optimizer.features.dataset.ports import DatasetSampleFrameReader
from home_optimizer.features.modeling import RoomRcModel, TrainedLinearRoomModel
from home_optimizer.features.modeling.room_2r2c import RoomRC2StateParams
from home_optimizer.features.mpc.controller_service import SpaceHeatingMpcControllerService
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


class SpaceHeatingMpcPlanningService:
    def __init__(
        self,
        *,
        samples_reader: DatasetSampleFrameReader,
        active_room_model_reader: ActiveRoomModelReaderPort,
        target_schedule: list[TemperatureTargetWindow],
        controller: SpaceHeatingMpcControllerService | None = None,
    ) -> None:
        self.samples_reader = samples_reader
        self.active_room_model_reader = active_room_model_reader
        self.target_schedule = list(target_schedule)
        self.controller = controller or SpaceHeatingMpcControllerService()

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
        resolved_interval_minutes = interval_minutes or source_model.interval_minutes
        resolved_constraints = constraints or MpcConstraints()
        resolved_weights = objective_weights or MpcObjectiveWeights()

        history_rows = (
            max(
                resolved_constraints.min_on_steps,
                resolved_constraints.min_off_steps,
            )
            + 1
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
        if model_id is not None:
            return self.active_room_model_reader.get_room_model_version(model_id)
        return self.active_room_model_reader.get_active_room_model_version()

    def _load_initial_rows(
        self,
        *,
        start_time_utc: datetime,
        interval_minutes: int,
        history_rows: int,
    ) -> list[MpcDatasetRow]:
        dataset_service = MpcDatasetService(
            self.samples_reader,
            _MpcDatasetSettings(self.target_schedule),
        )
        dataset = dataset_service.build_dataset(
            start_time=start_time_utc - timedelta(minutes=interval_minutes * history_rows),
            end_time=start_time_utc + timedelta(minutes=interval_minutes),
            interval_minutes=interval_minutes,
        )
        rows = [row for row in dataset.rows if row.timestamp_utc <= start_time_utc]
        if not rows:
            raise ValueError("No dataset rows available to estimate MPC initial state")
        return rows

    def _load_forecast_entries(
        self,
        *,
        start_time_utc: datetime,
        interval_minutes: int,
        horizon_steps: int,
    ) -> list[ForecastEntry]:
        end_time_utc = start_time_utc + timedelta(minutes=interval_minutes * horizon_steps)
        frame = self.samples_reader.read_forecast_values(
            start_time=start_time_utc,
            end_time=end_time_utc,
        )
        entries: list[ForecastEntry] = []
        for row in frame.to_dict(orient="records"):
            entries.append(
                ForecastEntry(
                    created_at_utc=parse_datetime(str(row["created_at_utc"])),
                    forecast_time_utc=parse_datetime(str(row["forecast_time_utc"])),
                    name=str(row["name"]),
                    value=float(row["value"]),
                    unit=str(row["unit"]) if row.get("unit") is not None else None,
                    source=str(row["source"]),
                )
            )
        return entries

    def _load_price_intervals(
        self,
        *,
        start_time_utc: datetime,
        interval_minutes: int,
        horizon_steps: int,
    ) -> list[PriceInterval]:
        end_time_utc = start_time_utc + timedelta(minutes=interval_minutes * horizon_steps)
        frame = self.samples_reader.read_electricity_price_intervals(
            start_time=start_time_utc,
            end_time=end_time_utc,
        )
        intervals: list[PriceInterval] = []
        for row in frame.to_dict(orient="records"):
            intervals.append(
                PriceInterval(
                    start_time_utc=parse_datetime(str(row["start_time_utc"])),
                    end_time_utc=parse_datetime(str(row["end_time_utc"])),
                    source=str(row["source"]),
                    name=str(row["name"]),
                    unit=str(row["unit"]),
                    value=float(row["value"]),
                )
            )
        return intervals

    def _initial_state_from_rows(
        self,
        rows: list[MpcDatasetRow],
        *,
        source_model: TrainedLinearRoomModel | RoomRcModel,
    ) -> MpcInitialState | Rc2StateMpcInitialState:
        latest = rows[-1]
        if latest.room_temperature_c is None:
            raise ValueError(
                "Latest dataset row is missing room_temperature_c for MPC initial state"
            )
        latest_hp_on = self._row_hp_on(latest)
        contiguous_steps = 0
        for row in reversed(rows):
            if self._row_hp_on(row) != latest_hp_on:
                break
            contiguous_steps += 1
        base_state = MpcInitialState(
            room_temp_c=float(latest.room_temperature_c),
            hp_on=latest_hp_on,
            on_steps=contiguous_steps if latest_hp_on else 0,
            off_steps=contiguous_steps if not latest_hp_on else 0,
        )
        if not isinstance(source_model, RoomRcModel):
            return base_state
        params = RoomRC2StateParams.from_dict(source_model.params)
        return Rc2StateMpcInitialState(
            room_temp_c=base_state.room_temp_c,
            mass_temp_c=base_state.room_temp_c + params.initial_mass_offset_c,
            hp_on=base_state.hp_on,
            on_steps=base_state.on_steps,
            off_steps=base_state.off_steps,
        )

    def _resolve_effective_heating_kw(
        self,
        rows: list[MpcDatasetRow],
        *,
        fallback_kw: float | None,
    ) -> float:
        heating_values = [
            float(row.space_heating_output_estimate_kw)
            for row in rows
            if row.space_heating_output_estimate_kw is not None
            and row.space_heating_output_estimate_kw > 0.0
        ]
        if heating_values:
            return float(sum(heating_values) / len(heating_values))
        if fallback_kw is not None:
            return float(fallback_kw)
        raise ValueError(
            "Unable to infer effective heating kW; provide default_effective_heating_kw"
        )

    def _resolve_hp_electric_power_kw(
        self,
        rows: list[MpcDatasetRow],
        *,
        fallback_kw: float,
    ) -> float:
        electric_power_values = [
            float(row.hp_electric_power_kw)
            for row in rows
            if row.hp_electric_power_kw is not None and row.hp_electric_power_kw > 0.0
        ]
        if electric_power_values:
            return float(sum(electric_power_values) / len(electric_power_values))
        return float(fallback_kw)

    @staticmethod
    def _resolve_export_price_eur_kwh(rows: list[MpcDatasetRow]) -> float:
        export_price_values = [
            float(row.price_export_eur_kwh)
            for row in rows
            if row.price_export_eur_kwh is not None and row.price_export_eur_kwh >= 0.0
        ]
        if not export_price_values:
            return 0.0
        return float(export_price_values[-1])

    @staticmethod
    def _resolve_base_load_power_kw(rows: list[MpcDatasetRow]) -> float:
        base_load_values = [
            (
                float(row.net_power_kw)
                + float(row.pv_output_power_kw or 0.0)
                - float(row.hp_electric_power_kw or 0.0)
            )
            for row in rows
            if row.net_power_kw is not None
        ]
        if not base_load_values:
            return 0.0
        return float(sum(base_load_values) / len(base_load_values))

    @staticmethod
    def _infer_pv_power_input_scale(
        rows: list[MpcDatasetRow],
        *,
        forecast_entries: list[ForecastEntry],
        start_time_utc: datetime,
    ) -> float:
        latest_row = rows[-1]
        latest_pv_output_kw = float(latest_row.pv_output_power_kw or 0.0)
        if latest_pv_output_kw <= 0.0:
            return 0.0

        latest_created_by_forecast_time: dict[datetime, tuple[datetime, float]] = {}
        for entry in forecast_entries:
            if entry.name != GTI_PV:
                continue
            forecast_time_utc = ensure_utc(entry.forecast_time_utc)
            created_at_utc = ensure_utc(entry.created_at_utc)
            existing = latest_created_by_forecast_time.get(forecast_time_utc)
            if existing is None or created_at_utc >= existing[0]:
                latest_created_by_forecast_time[forecast_time_utc] = (
                    created_at_utc,
                    float(entry.value),
                )

        start_forecast_value = latest_created_by_forecast_time.get(start_time_utc)
        if start_forecast_value is None or start_forecast_value[1] <= 0.0:
            return 0.0
        return latest_pv_output_kw / start_forecast_value[1]

    @staticmethod
    def _row_hp_on(row: MpcDatasetRow) -> bool:
        heating_output_kw = float(row.space_heating_output_estimate_kw or 0.0)
        return bool(row.mode_space) and not bool(row.mode_off) and heating_output_kw > 0.0


class _MpcDatasetSettings:
    def __init__(self, target_schedule: list[TemperatureTargetWindow]) -> None:
        self.room_target = list(target_schedule)
        self.dhw_target: list[TemperatureTargetWindow] = []
        self.electricity_pricing = _DynamicPricingLike()


class _DynamicPricingLike:
    mode = "dynamic"
