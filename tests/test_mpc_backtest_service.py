from __future__ import annotations

from datetime import datetime, time, timezone

import pytest

from home_optimizer.domain.target_schedule import TemperatureTargetWindow
from home_optimizer.features.backtest.models import (
    MpcBacktestResult,
    MpcBacktestSummary,
)
from home_optimizer.features.backtest.service import SpaceHeatingMpcBacktestService
from home_optimizer.features.dataset.models import MpcDataset, MpcDatasetRow
from home_optimizer.features.modeling import (
    RoomRcConfig,
    RoomRcModel,
    RoomArxConfig,
    RoomArxModel,
    StoredModelVersion,
)
from home_optimizer.features.mpc import MpcInitialState


class _UnusedSamplesReader:
    pass


class _UnusedModelReader:
    def get_room_model_version(self, model_id: str):
        return None

    def get_active_room_model_version(self):
        return None


class _StubRunner:
    def run(self, **kwargs) -> MpcBacktestResult:
        start_time = kwargs["timeline"][0].timestamp_utc
        end_time = kwargs["timeline"][-1].timestamp_utc
        return MpcBacktestResult(
            model_id=kwargs["model_id"],
            model_type=kwargs["model_type"],
            start_time_utc=start_time,
            end_time_utc=end_time,
            interval_minutes=kwargs["interval_minutes"],
            horizon_steps=kwargs["horizon_steps"],
            step_results=[],
            mpc_summary=MpcBacktestSummary(
                comfort_violation_minutes=0,
                degree_minutes_below_comfort=0.0,
                degree_minutes_above_comfort=0.0,
                starts_per_day=0.0,
                runtime_minutes=0,
                estimated_energy_cost_eur=0.0,
            ),
            historical_summary=MpcBacktestSummary(
                comfort_violation_minutes=0,
                degree_minutes_below_comfort=0.0,
                degree_minutes_above_comfort=0.0,
                starts_per_day=0.0,
                runtime_minutes=0,
                estimated_energy_cost_eur=0.0,
            ),
            total_solver_runtime_seconds=0.0,
        )


def test_backtest_service_uses_backtest_window_rows_to_infer_effective_heating_kw(monkeypatch) -> None:
    start_time = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)
    end_time = datetime(2026, 1, 1, 6, 20, tzinfo=timezone.utc)
    service = SpaceHeatingMpcBacktestService(
        samples_reader=_UnusedSamplesReader(),
        active_room_model_reader=_UnusedModelReader(),
        target_schedule=[
            TemperatureTargetWindow(time=time(0, 0), target=20.0, low_margin=0.5, high_margin=1.0)
        ],
        default_interval_minutes=10,
        runner=_StubRunner(),
    )
    version = StoredModelVersion(
        model_id="room-model-active",
        model_type="room_arx",
        created_at_utc=start_time,
        is_active=True,
        model=RoomArxModel(
            trained_from_utc=start_time,
            trained_to_utc=end_time,
            interval_minutes=10,
            config=RoomArxConfig(
                room_temperature_lags=[0],
                outdoor_temperature_lags=[0],
                thermal_output_lags=[0],
                solar_gain_lags=[0],
                occupied_flag_lags=[0],
                shutter_position_lags=[0],
                solar_shutter_interaction_lags=[0],
            ),
            feature_names=[
                "room_temperature_lag_0",
                "outdoor_temperature_lag_0",
                "thermal_output_lag_0",
                "solar_gain_lag_0",
                "occupied_flag_lag_0",
            ],
            intercept=0.0,
            coefficients=[1.0, 0.0, 0.0, 0.0, 0.0],
            sample_count=10,
        ),
    )
    initial_rows = [
        MpcDatasetRow(
            timestamp_utc=start_time,
            room_temperature_c=19.0,
            mode_space=0,
            mode_off=1,
            space_heating_output_estimate_kw=0.0,
        )
    ]
    backtest_rows = [
        MpcDatasetRow(
            timestamp_utc=start_time,
            room_temperature_c=19.0,
            outdoor_temperature_c=5.0,
            room_target_min_temperature_c=19.0,
            room_target_max_temperature_c=21.0,
            space_heating_output_estimate_kw=3.2,
        ),
        MpcDatasetRow(
            timestamp_utc=end_time,
            room_temperature_c=19.5,
            outdoor_temperature_c=5.0,
            room_target_min_temperature_c=19.0,
            room_target_max_temperature_c=21.0,
            space_heating_output_estimate_kw=3.2,
        ),
    ]

    monkeypatch.setattr(service.preparation, "resolve_room_model_version", lambda model_id: version)
    monkeypatch.setattr(service.preparation, "load_initial_rows", lambda **kwargs: initial_rows)
    monkeypatch.setattr(
        service.preparation,
        "initial_state_from_rows",
        lambda *args, **kwargs: MpcInitialState(room_temp_c=19.0),
    )
    monkeypatch.setattr(
        service.preparation,
        "build_dataset",
        lambda **kwargs: MpcDataset(
            interval_minutes=10,
            start_time_utc=start_time,
            end_time_utc=end_time,
            rows=backtest_rows,
        ),
    )

    captured_lengths: list[int] = []

    def resolve_effective_heating_kw(rows, *, fallback_kw):
        captured_lengths.append(len(rows))
        return 3.2

    monkeypatch.setattr(service.preparation, "resolve_effective_heating_kw", resolve_effective_heating_kw)
    monkeypatch.setattr(service.preparation, "resolve_hp_electric_power_kw", lambda rows, *, fallback_kw: 1.5)
    monkeypatch.setattr(service.preparation, "resolve_export_price_eur_kwh", lambda rows: 0.0)
    monkeypatch.setattr(service.preparation, "row_hp_on", lambda row: bool(row.space_heating_output_estimate_kw))

    service.run(
        start_time_utc=start_time,
        end_time_utc=end_time,
        horizon_steps=2,
    )

    assert captured_lengths == [3]


def test_backtest_service_converts_solar_proxy_to_kw_for_2r2c_models(monkeypatch) -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    end_time = datetime(2026, 1, 1, 12, 10, tzinfo=timezone.utc)
    captured_solar_inputs: list[float] = []
    captured_filtered_solar_inputs: list[float] = []

    class _CapturingRunner(_StubRunner):
        def run(self, **kwargs) -> MpcBacktestResult:
            captured_solar_inputs.extend(step.solar_gain_kw for step in kwargs["timeline"])
            captured_filtered_solar_inputs.extend(
                float(step.solar_gain_mass_kw) for step in kwargs["timeline"]
            )
            return super().run(**kwargs)

    service = SpaceHeatingMpcBacktestService(
        samples_reader=_UnusedSamplesReader(),
        active_room_model_reader=_UnusedModelReader(),
        target_schedule=[
            TemperatureTargetWindow(time=time(0, 0), target=20.0, low_margin=0.5, high_margin=1.0)
        ],
        default_interval_minutes=10,
        runner=_CapturingRunner(),
    )
    version = StoredModelVersion(
        model_id="room-model-2r2c",
        model_type="room_2r2c",
        created_at_utc=start_time,
        is_active=True,
        model=RoomRcModel(
            trained_from_utc=start_time,
            trained_to_utc=end_time,
            interval_minutes=10,
            config=RoomRcConfig(glass_area_m2=8.0, g_glass=0.5),
            params={},
            sample_count=10,
        ),
    )
    rows = [
        MpcDatasetRow(
            timestamp_utc=start_time,
            room_temperature_c=20.0,
            outdoor_temperature_c=5.0,
            room_target_min_temperature_c=19.0,
            room_target_max_temperature_c=21.0,
            solar_gain_proxy_w_m2=200.0,
            occupied_flag=0,
        ),
        MpcDatasetRow(
            timestamp_utc=end_time,
            room_temperature_c=20.1,
            outdoor_temperature_c=5.0,
            room_target_min_temperature_c=19.0,
            room_target_max_temperature_c=21.0,
            solar_gain_proxy_w_m2=200.0,
            occupied_flag=0,
        ),
    ]

    monkeypatch.setattr(service.preparation, "resolve_room_model_version", lambda model_id: version)
    monkeypatch.setattr(service.preparation, "load_initial_rows", lambda **kwargs: rows[:1])
    monkeypatch.setattr(
        service.preparation,
        "initial_state_from_rows",
        lambda *args, **kwargs: MpcInitialState(room_temp_c=20.0),
    )
    monkeypatch.setattr(
        service.preparation,
        "build_dataset",
        lambda **kwargs: MpcDataset(
            interval_minutes=10,
            start_time_utc=start_time,
            end_time_utc=end_time,
            rows=rows,
        ),
    )
    monkeypatch.setattr(service.preparation, "resolve_effective_heating_kw", lambda rows, *, fallback_kw: 3.0)
    monkeypatch.setattr(service.preparation, "resolve_hp_electric_power_kw", lambda rows, *, fallback_kw: 1.5)
    monkeypatch.setattr(service.preparation, "resolve_export_price_eur_kwh", lambda rows: 0.0)
    monkeypatch.setattr(service.preparation, "row_hp_on", lambda row: False)

    service.run(
        start_time_utc=start_time,
        end_time_utc=end_time,
        horizon_steps=2,
    )

    assert captured_solar_inputs == [0.8, 0.8]
    assert captured_filtered_solar_inputs == pytest.approx([0.12, 0.222])
