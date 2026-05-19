from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

import pytest

from home_optimizer.domain.target_schedule import TemperatureTargetWindow
from home_optimizer.features.backtest.models import (
    MpcBacktestResult,
    MpcBacktestPvDiagnostics,
    MpcBacktestSummary,
)
from home_optimizer.features.backtest.runner import SpaceHeatingMpcBacktestRunner
from home_optimizer.features.backtest.service import SpaceHeatingMpcBacktestService
from home_optimizer.features.dataset.models import MpcDataset, MpcDatasetRow
from home_optimizer.features.modeling import (
    RoomRcConfig,
    RoomRcModel,
    StoredModelVersion,
)
from home_optimizer.features.mpc import MpcInitialState
from home_optimizer.features.mpc import Rc2StateMpcInitialState
from home_optimizer.features.mpc import MpcObjectiveBreakdown
from home_optimizer.features.mpc import MpcPlan, MpcPlanStep


class _UnusedSamplesReader:
    def read_forecast_values(self, **kwargs):
        raise NotImplementedError

    def read_electricity_price_intervals(self, **kwargs):
        raise NotImplementedError


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
            exogenous_mode="perfect_foresight",
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
            pv_diagnostics=MpcBacktestPvDiagnostics(),
            mpc_objective_breakdown=MpcObjectiveBreakdown(),
            solver_objective_breakdown=MpcObjectiveBreakdown(),
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
        model_type="room_2r2c",
        created_at_utc=start_time,
        is_active=True,
        model=RoomRcModel(
            trained_from_utc=start_time,
            trained_to_utc=end_time,
            interval_minutes=10,
            config=RoomRcConfig(),
            params={},
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
        lambda *args, **kwargs: Rc2StateMpcInitialState(room_temp_c=20.0, mass_temp_c=20.0),
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


def test_backtest_service_perfect_foresight_sets_forecast_and_realized_exogenous_values_equal(
    monkeypatch,
) -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    end_time = datetime(2026, 1, 1, 12, 10, tzinfo=timezone.utc)
    captured_timeline = []

    class _CapturingRunner(_StubRunner):
        def run(self, **kwargs) -> MpcBacktestResult:
            captured_timeline.extend(kwargs["timeline"])
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
        model_id="room-model-active",
        model_type="room_2r2c",
        created_at_utc=start_time,
        is_active=True,
        model=RoomRcModel(
            trained_from_utc=start_time,
            trained_to_utc=end_time,
            interval_minutes=10,
            config=RoomRcConfig(),
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
            solar_irradiance_w_m2=300.0,
            solar_gain_proxy_w_m2=150.0,
            pv_output_power_kw=2.5,
            net_power_kw=1.0,
            hp_electric_power_kw=0.5,
            occupied_flag=0,
        ),
        MpcDatasetRow(
            timestamp_utc=end_time,
            room_temperature_c=20.1,
            outdoor_temperature_c=5.0,
            room_target_min_temperature_c=19.0,
            room_target_max_temperature_c=21.0,
            solar_irradiance_w_m2=320.0,
            solar_gain_proxy_w_m2=160.0,
            pv_output_power_kw=2.8,
            net_power_kw=1.2,
            hp_electric_power_kw=0.6,
            occupied_flag=0,
        ),
    ]

    monkeypatch.setattr(service.preparation, "resolve_room_model_version", lambda model_id: version)
    monkeypatch.setattr(service.preparation, "load_initial_rows", lambda **kwargs: rows[:1])
    monkeypatch.setattr(
        service.preparation,
        "initial_state_from_rows",
        lambda *args, **kwargs: Rc2StateMpcInitialState(room_temp_c=20.0, mass_temp_c=20.0),
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

    result = service.run(
        start_time_utc=start_time,
        end_time_utc=end_time,
        horizon_steps=2,
    )

    assert result.exogenous_mode == "perfect_foresight"
    assert len(captured_timeline) == 2
    assert captured_timeline[0].pv_available_power_forecast_kw == captured_timeline[0].pv_available_power_realized_kw
    assert captured_timeline[0].solar_irradiance_forecast_w_m2 == captured_timeline[0].solar_irradiance_realized_w_m2
    assert captured_timeline[0].base_load_power_forecast_kw == captured_timeline[0].base_load_power_realized_kw


def test_backtest_service_forecast_replay_uses_archived_forecast_values(monkeypatch) -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    middle_time = datetime(2026, 1, 1, 12, 10, tzinfo=timezone.utc)
    end_time = datetime(2026, 1, 1, 12, 20, tzinfo=timezone.utc)
    captured_horizons = []

    class _ForecastSamplesReader(_UnusedSamplesReader):
        def read_forecast_values(self, **kwargs):
            return __import__("pandas").DataFrame(
                [
                    {
                        "created_at_utc": start_time - timedelta(minutes=30),
                        "forecast_time_utc": start_time,
                        "name": "temperature",
                        "source": "forecast",
                        "unit": "C",
                        "value": 3.0,
                    },
                    {
                        "created_at_utc": start_time - timedelta(minutes=30),
                        "forecast_time_utc": start_time,
                        "name": "gti_living_room_windows_adjusted",
                        "source": "forecast",
                        "unit": "W/m2",
                        "value": 500.0,
                    },
                    {
                        "created_at_utc": start_time - timedelta(minutes=30),
                        "forecast_time_utc": start_time,
                        "name": "gti_pv",
                        "source": "forecast",
                        "unit": "W/m2",
                        "value": 400.0,
                    },
                    {
                        "created_at_utc": start_time - timedelta(minutes=30),
                        "forecast_time_utc": middle_time,
                        "name": "temperature",
                        "source": "forecast",
                        "unit": "C",
                        "value": 4.0,
                    },
                    {
                        "created_at_utc": start_time - timedelta(minutes=30),
                        "forecast_time_utc": middle_time,
                        "name": "gti_living_room_windows_adjusted",
                        "source": "forecast",
                        "unit": "W/m2",
                        "value": 300.0,
                    },
                    {
                        "created_at_utc": start_time - timedelta(minutes=30),
                        "forecast_time_utc": middle_time,
                        "name": "gti_pv",
                        "source": "forecast",
                        "unit": "W/m2",
                        "value": 200.0,
                    },
                    {
                        "created_at_utc": start_time - timedelta(minutes=30),
                        "forecast_time_utc": end_time,
                        "name": "temperature",
                        "source": "forecast",
                        "unit": "C",
                        "value": 5.0,
                    },
                    {
                        "created_at_utc": start_time - timedelta(minutes=30),
                        "forecast_time_utc": end_time,
                        "name": "gti_living_room_windows_adjusted",
                        "source": "forecast",
                        "unit": "W/m2",
                        "value": 150.0,
                    },
                    {
                        "created_at_utc": start_time - timedelta(minutes=30),
                        "forecast_time_utc": end_time,
                        "name": "gti_pv",
                        "source": "forecast",
                        "unit": "W/m2",
                        "value": 100.0,
                    },
                ]
            )

        def read_electricity_price_intervals(self, **kwargs):
            return __import__("pandas").DataFrame(
                [
                    {
                        "start_time_utc": start_time,
                        "end_time_utc": end_time + timedelta(minutes=10),
                        "source": "price",
                        "name": "electricity_price",
                        "unit": "EUR/kWh",
                        "value": 0.25,
                    }
                ]
            )

    class _CapturingController:
        def plan(self, request, **kwargs) -> MpcPlan:
            captured_horizons.append(request.horizon)
            step = request.horizon[0]
            return MpcPlan(
                status="ok",
                termination_condition="optimal",
                feasible=True,
                objective_breakdown=MpcObjectiveBreakdown(),
                steps=[
                    MpcPlanStep(
                        timestamp_utc=step.timestamp_utc,
                        hp_on=False,
                        start=False,
                        stop=False,
                        predicted_room_temp_c=20.0,
                        temp_min_c=step.temp_min_c,
                        temp_max_c=step.temp_max_c,
                        slack_low_c=0.0,
                        slack_high_c=0.0,
                        effective_heating_kw=0.0,
                        price_eur_kwh=step.import_price_eur_kwh,
                        estimated_energy_cost_eur=0.0,
                    )
                ],
            )

    service = SpaceHeatingMpcBacktestService(
        samples_reader=_ForecastSamplesReader(),
        active_room_model_reader=_UnusedModelReader(),
        target_schedule=[
            TemperatureTargetWindow(time=time(0, 0), target=20.0, low_margin=0.5, high_margin=1.0)
        ],
        default_interval_minutes=10,
        runner=SpaceHeatingMpcBacktestRunner(controller=_CapturingController()),
    )
    version = StoredModelVersion(
        model_id="room-model-active",
        model_type="room_2r2c",
        created_at_utc=start_time,
        is_active=True,
        model=RoomRcModel(
            trained_from_utc=start_time,
            trained_to_utc=end_time,
            interval_minutes=10,
            config=RoomRcConfig(),
            params={},
            sample_count=10,
        ),
    )
    rows = [
        MpcDatasetRow(
            timestamp_utc=start_time,
            room_temperature_c=20.0,
            outdoor_temperature_c=10.0,
            room_target_min_temperature_c=19.0,
            room_target_max_temperature_c=21.0,
            solar_irradiance_w_m2=100.0,
            solar_gain_proxy_w_m2=100.0,
            pv_output_power_kw=1.0,
            net_power_kw=0.8,
            hp_electric_power_kw=0.5,
            occupied_flag=0,
            price_import_eur_kwh=0.25,
        ),
        MpcDatasetRow(
            timestamp_utc=middle_time,
            room_temperature_c=20.1,
            outdoor_temperature_c=11.0,
            room_target_min_temperature_c=19.0,
            room_target_max_temperature_c=21.0,
            solar_irradiance_w_m2=120.0,
            solar_gain_proxy_w_m2=120.0,
            pv_output_power_kw=1.1,
            net_power_kw=0.9,
            hp_electric_power_kw=0.5,
            occupied_flag=0,
            price_import_eur_kwh=0.25,
        ),
        MpcDatasetRow(
            timestamp_utc=end_time,
            room_temperature_c=20.2,
            outdoor_temperature_c=12.0,
            room_target_min_temperature_c=19.0,
            room_target_max_temperature_c=21.0,
            solar_irradiance_w_m2=140.0,
            solar_gain_proxy_w_m2=140.0,
            pv_output_power_kw=1.2,
            net_power_kw=1.0,
            hp_electric_power_kw=0.5,
            occupied_flag=0,
            price_import_eur_kwh=0.25,
        ),
    ]

    monkeypatch.setattr(service.preparation, "resolve_room_model_version", lambda model_id: version)
    monkeypatch.setattr(service.preparation, "load_initial_rows", lambda **kwargs: rows[:1])
    monkeypatch.setattr(
        service.preparation,
        "initial_state_from_rows",
        lambda *args, **kwargs: Rc2StateMpcInitialState(room_temp_c=20.0, mass_temp_c=20.0),
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

    result = service.run(
        start_time_utc=start_time,
        end_time_utc=end_time,
        horizon_steps=2,
        exogenous_mode="forecast_replay",
    )

    assert result.exogenous_mode == "forecast_replay"
    assert result.forecast_coverage_ratio == pytest.approx(1.0)
    assert result.missing_forecast_count == 0
    assert len(captured_horizons) == 2
    assert captured_horizons[0][0].outdoor_temp_c == 3.0
    assert captured_horizons[0][1].pv_available_power_forecast_kw == pytest.approx(0.5)
    assert captured_horizons[0][1].pv_available_power_forecast_kw != captured_horizons[0][1].pv_available_power_realized_kw
    assert captured_horizons[0][0].solar_irradiance_forecast_w_m2 == 500.0
    assert captured_horizons[0][0].solar_irradiance_realized_w_m2 == 100.0
    assert result.step_results[0].pv_forecast_kw == pytest.approx(1.0)
    assert result.step_results[0].pv_realized_kw == pytest.approx(1.0)
    assert result.step_results[1].pv_forecast_kw != pytest.approx(result.step_results[1].pv_realized_kw)
    assert result.step_results[1].pv_realized_kw == pytest.approx(1.1)


def test_backtest_service_forecast_replay_derives_adjusted_solar_from_raw_window_forecast(
    monkeypatch,
) -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    end_time = datetime(2026, 1, 1, 12, 10, tzinfo=timezone.utc)
    initial_time = start_time - timedelta(minutes=10)
    captured_horizons = []

    class _ForecastSamplesReader(_UnusedSamplesReader):
        def read_forecast_values(self, **kwargs):
            return __import__("pandas").DataFrame(
                [
                    {
                        "created_at_utc": start_time - timedelta(minutes=30),
                        "forecast_time_utc": start_time,
                        "name": "temperature",
                        "source": "forecast",
                        "unit": "C",
                        "value": 3.0,
                    },
                    {
                        "created_at_utc": start_time - timedelta(minutes=30),
                        "forecast_time_utc": start_time,
                        "name": "gti_living_room_windows",
                        "source": "forecast",
                        "unit": "W/m2",
                        "value": 500.0,
                    },
                    {
                        "created_at_utc": start_time - timedelta(minutes=30),
                        "forecast_time_utc": start_time,
                        "name": "gti_pv",
                        "source": "forecast",
                        "unit": "W/m2",
                        "value": 400.0,
                    },
                    {
                        "created_at_utc": start_time - timedelta(minutes=30),
                        "forecast_time_utc": end_time,
                        "name": "temperature",
                        "source": "forecast",
                        "unit": "C",
                        "value": 4.0,
                    },
                    {
                        "created_at_utc": start_time - timedelta(minutes=30),
                        "forecast_time_utc": end_time,
                        "name": "gti_living_room_windows",
                        "source": "forecast",
                        "unit": "W/m2",
                        "value": 300.0,
                    },
                    {
                        "created_at_utc": start_time - timedelta(minutes=30),
                        "forecast_time_utc": end_time,
                        "name": "gti_pv",
                        "source": "forecast",
                        "unit": "W/m2",
                        "value": 200.0,
                    },
                ]
            )

        def read_electricity_price_intervals(self, **kwargs):
            return __import__("pandas").DataFrame(
                [
                    {
                        "start_time_utc": start_time,
                        "end_time_utc": end_time + timedelta(minutes=10),
                        "source": "price",
                        "name": "electricity_price",
                        "unit": "EUR/kWh",
                        "value": 0.25,
                    }
                ]
            )

    class _CapturingController:
        def plan(self, request, **kwargs) -> MpcPlan:
            captured_horizons.append(request.horizon)
            step = request.horizon[0]
            return MpcPlan(
                status="ok",
                termination_condition="optimal",
                feasible=True,
                objective_breakdown=MpcObjectiveBreakdown(),
                steps=[
                    MpcPlanStep(
                        timestamp_utc=step.timestamp_utc,
                        hp_on=False,
                        start=False,
                        stop=False,
                        predicted_room_temp_c=20.0,
                        temp_min_c=step.temp_min_c,
                        temp_max_c=step.temp_max_c,
                        slack_low_c=0.0,
                        slack_high_c=0.0,
                        effective_heating_kw=0.0,
                        price_eur_kwh=step.import_price_eur_kwh,
                        estimated_energy_cost_eur=0.0,
                    )
                ],
            )

    service = SpaceHeatingMpcBacktestService(
        samples_reader=_ForecastSamplesReader(),
        active_room_model_reader=_UnusedModelReader(),
        target_schedule=[
            TemperatureTargetWindow(time=time(0, 0), target=20.0, low_margin=0.5, high_margin=1.0)
        ],
        default_interval_minutes=10,
        runner=SpaceHeatingMpcBacktestRunner(controller=_CapturingController()),
    )
    version = StoredModelVersion(
        model_id="room-model-active",
        model_type="room_2r2c",
        created_at_utc=start_time,
        is_active=True,
        model=RoomRcModel(
            trained_from_utc=start_time,
            trained_to_utc=end_time,
            interval_minutes=10,
            config=RoomRcConfig(),
            params={},
            sample_count=10,
        ),
    )
    rows = [
        MpcDatasetRow(
            timestamp_utc=initial_time,
            room_temperature_c=19.9,
            outdoor_temperature_c=9.0,
            room_target_min_temperature_c=19.0,
            room_target_max_temperature_c=21.0,
            shutter_position_pct=50.0,
            occupied_flag=0,
            price_import_eur_kwh=0.25,
        ),
        MpcDatasetRow(
            timestamp_utc=start_time,
            room_temperature_c=20.0,
            outdoor_temperature_c=10.0,
            room_target_min_temperature_c=19.0,
            room_target_max_temperature_c=21.0,
            solar_irradiance_w_m2=100.0,
            solar_gain_proxy_w_m2=100.0,
            pv_output_power_kw=1.0,
            net_power_kw=0.8,
            hp_electric_power_kw=0.5,
            occupied_flag=0,
            price_import_eur_kwh=0.25,
            shutter_position_pct=50.0,
        ),
        MpcDatasetRow(
            timestamp_utc=end_time,
            room_temperature_c=20.1,
            outdoor_temperature_c=11.0,
            room_target_min_temperature_c=19.0,
            room_target_max_temperature_c=21.0,
            solar_irradiance_w_m2=120.0,
            solar_gain_proxy_w_m2=120.0,
            pv_output_power_kw=1.1,
            net_power_kw=0.9,
            hp_electric_power_kw=0.5,
            occupied_flag=0,
            price_import_eur_kwh=0.25,
            shutter_position_pct=50.0,
        ),
    ]

    monkeypatch.setattr(service.preparation, "resolve_room_model_version", lambda model_id: version)
    monkeypatch.setattr(service.preparation, "load_initial_rows", lambda **kwargs: rows[:1])
    monkeypatch.setattr(
        service.preparation,
        "initial_state_from_rows",
        lambda *args, **kwargs: Rc2StateMpcInitialState(room_temp_c=20.0, mass_temp_c=20.0),
    )
    monkeypatch.setattr(
        service.preparation,
        "build_dataset",
        lambda **kwargs: MpcDataset(
            interval_minutes=10,
            start_time_utc=start_time,
            end_time_utc=end_time,
            rows=rows[1:],
        ),
    )
    monkeypatch.setattr(service.preparation, "resolve_effective_heating_kw", lambda rows, *, fallback_kw: 3.0)
    monkeypatch.setattr(service.preparation, "resolve_hp_electric_power_kw", lambda rows, *, fallback_kw: 1.5)
    monkeypatch.setattr(service.preparation, "resolve_export_price_eur_kwh", lambda rows: 0.0)
    monkeypatch.setattr(service.preparation, "row_hp_on", lambda row: False)

    result = service.run(
        start_time_utc=start_time,
        end_time_utc=end_time,
        horizon_steps=2,
        exogenous_mode="forecast_replay",
    )

    assert result.exogenous_mode == "forecast_replay"
    assert result.forecast_coverage_ratio == pytest.approx(1.0)
    assert result.missing_forecast_count == 0
    assert len(captured_horizons) == 1
    assert captured_horizons[0][0].solar_irradiance_forecast_w_m2 == 250.0
    assert captured_horizons[0][1].solar_irradiance_forecast_w_m2 == 150.0


def test_backtest_service_forecast_replay_rejects_missing_archived_forecast(monkeypatch) -> None:
    start_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    end_time = datetime(2026, 1, 1, 12, 10, tzinfo=timezone.utc)

    class _MissingForecastSamplesReader(_UnusedSamplesReader):
        def read_forecast_values(self, **kwargs):
            return __import__("pandas").DataFrame([])

        def read_electricity_price_intervals(self, **kwargs):
            return __import__("pandas").DataFrame([])

    service = SpaceHeatingMpcBacktestService(
        samples_reader=_MissingForecastSamplesReader(),
        active_room_model_reader=_UnusedModelReader(),
        target_schedule=[
            TemperatureTargetWindow(time=time(0, 0), target=20.0, low_margin=0.5, high_margin=1.0)
        ],
        default_interval_minutes=10,
        runner=_StubRunner(),
    )
    version = StoredModelVersion(
        model_id="room-model-active",
        model_type="room_2r2c",
        created_at_utc=start_time,
        is_active=True,
        model=RoomRcModel(
            trained_from_utc=start_time,
            trained_to_utc=end_time,
            interval_minutes=10,
            config=RoomRcConfig(),
            params={},
            sample_count=10,
        ),
    )
    rows = [
        MpcDatasetRow(
            timestamp_utc=start_time,
            room_temperature_c=20.0,
            outdoor_temperature_c=10.0,
            room_target_min_temperature_c=19.0,
            room_target_max_temperature_c=21.0,
            occupied_flag=0,
        ),
        MpcDatasetRow(
            timestamp_utc=end_time,
            room_temperature_c=20.1,
            outdoor_temperature_c=11.0,
            room_target_min_temperature_c=19.0,
            room_target_max_temperature_c=21.0,
            occupied_flag=0,
        ),
    ]

    monkeypatch.setattr(service.preparation, "resolve_room_model_version", lambda model_id: version)
    monkeypatch.setattr(service.preparation, "load_initial_rows", lambda **kwargs: rows[:1])
    monkeypatch.setattr(
        service.preparation,
        "initial_state_from_rows",
        lambda *args, **kwargs: Rc2StateMpcInitialState(room_temp_c=20.0, mass_temp_c=20.0),
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

    class _NoopController:
        def plan(self, request, **kwargs) -> MpcPlan:
            step = request.horizon[0]
            return MpcPlan(
                status="ok",
                termination_condition="optimal",
                feasible=True,
                objective_breakdown=MpcObjectiveBreakdown(),
                steps=[
                    MpcPlanStep(
                        timestamp_utc=step.timestamp_utc,
                        hp_on=False,
                        start=False,
                        stop=False,
                        predicted_room_temp_c=20.0,
                        temp_min_c=step.temp_min_c,
                        temp_max_c=step.temp_max_c,
                        slack_low_c=0.0,
                        slack_high_c=0.0,
                        effective_heating_kw=0.0,
                        price_eur_kwh=step.import_price_eur_kwh,
                        estimated_energy_cost_eur=0.0,
                    )
                ],
            )

    service.runner = SpaceHeatingMpcBacktestRunner(controller=_NoopController())

    with pytest.raises(ValueError, match="missing_forecast") as error_info:
        service.run(
            start_time_utc=start_time,
            end_time_utc=end_time,
            horizon_steps=2,
            exogenous_mode="forecast_replay",
        )
    message = str(error_info.value)
    assert "issue_time=" in message
    assert "coverage_ratio=" in message
    assert "latest_created_at_utc=none" in message
    assert "temperature@" in message
