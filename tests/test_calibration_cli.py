"""CLI coverage for offline calibration reporting."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone

from home_optimizer.calibration import (
    COPCalibrationDiagnostics,
    COPCalibrationDataset,
    COPCalibrationResult,
    COPCalibrationSample,
    COPCalibrationSegmentQuality,
)
from home_optimizer.calibration import __main__ as calibration_main
from home_optimizer.cop_model import HeatPumpCOPParameters


def _build_cop_cli_dataset() -> COPCalibrationDataset:
    """Return a tiny COP dataset with retained and dropped segment diagnostics."""
    start = datetime(2026, 4, 18, 8, 0, tzinfo=timezone.utc)
    return COPCalibrationDataset(
        samples=(
            COPCalibrationSample(
                bucket_start_utc=start,
                bucket_end_utc=start + timedelta(minutes=5),
                dt_hours=5.0 / 60.0,
                mode_name="ufh",
                outdoor_temperature_mean_c=5.0,
                supply_target_temperature_mean_c=36.0,
                supply_temperature_mean_c=35.6,
                thermal_energy_kwh=0.30,
                electric_energy_kwh=0.10,
            ),
            COPCalibrationSample(
                bucket_start_utc=start + timedelta(minutes=10),
                bucket_end_utc=start + timedelta(minutes=15),
                dt_hours=5.0 / 60.0,
                mode_name="dhw",
                outdoor_temperature_mean_c=6.0,
                supply_target_temperature_mean_c=55.0,
                supply_temperature_mean_c=54.4,
                thermal_energy_kwh=0.28,
                electric_energy_kwh=0.11,
            ),
        ),
        segment_qualities=(
            COPCalibrationSegmentQuality(
                raw_segment_index=0,
                mode_name="ufh",
                selected=True,
                sample_count=2,
                duration_hours=10.0 / 60.0,
                thermal_energy_kwh=0.60,
                electric_energy_kwh=0.20,
                outdoor_temperature_span_c=1.5,
                supply_target_temperature_span_c=2.0,
                actual_cop_span=0.25,
                supply_tracking_rmse_c=0.4,
                score=4.0,
            ),
            COPCalibrationSegmentQuality(
                raw_segment_index=1,
                mode_name="dhw",
                selected=False,
                sample_count=2,
                duration_hours=10.0 / 60.0,
                thermal_energy_kwh=0.56,
                electric_energy_kwh=0.22,
                outdoor_temperature_span_c=0.2,
                supply_target_temperature_span_c=0.0,
                actual_cop_span=0.05,
                supply_tracking_rmse_c=1.8,
                score=0.2,
            ),
        ),
    )


def _build_cop_cli_result(dataset: COPCalibrationDataset) -> COPCalibrationResult:
    """Return a deterministic COP fit result for CLI assertions."""
    return COPCalibrationResult(
        fitted_parameters=HeatPumpCOPParameters(
            eta_carnot=0.47,
            delta_T_cond=5.0,
            delta_T_evap=5.0,
            T_supply_min=27.5,
            T_ref_outdoor=18.0,
            heating_curve_slope=0.9,
            cop_min=1.5,
            cop_max=7.0,
        ),
        t_ref_outdoor_was_fitted=True,
        rmse_supply_temperature_c=0.25,
        rmse_electric_energy_kwh=0.03,
        rmse_actual_cop=0.12,
        ufh_rmse_electric_energy_kwh=0.02,
        dhw_rmse_electric_energy_kwh=0.05,
        ufh_rmse_actual_cop=0.11,
        dhw_rmse_actual_cop=0.14,
        ufh_bias_actual_cop=-0.02,
        dhw_bias_actual_cop=0.03,
        diagnostic_eta_carnot_ufh=0.46,
        diagnostic_eta_carnot_dhw=0.49,
        sample_count=dataset.sample_count,
        ufh_sample_count=dataset.ufh_sample_count,
        dhw_sample_count=dataset.dhw_sample_count,
        dataset_start_utc=dataset.start_utc,
        dataset_end_utc=dataset.end_utc,
        heating_curve_optimizer_status="success",
        eta_optimizer_status="success",
        heating_curve_optimizer_cost=0.01,
        eta_optimizer_cost=0.02,
    )


def _build_cop_cli_diagnostics() -> COPCalibrationDiagnostics:
    """Return a deterministic COP diagnostics object for CLI assertions."""
    dataset = _build_cop_cli_dataset()
    return COPCalibrationDiagnostics(
        raw_row_count=8,
        mode_accepted_count=5,
        defrost_accepted_count=5,
        booster_accepted_count=5,
        dt_accepted_count=5,
        thermal_energy_accepted_count=4,
        electric_energy_accepted_count=3,
        finite_supply_accepted_count=3,
        cop_accepted_count=2,
        raw_segment_count=2,
        selected_segment_count=1,
        selected_sample_count=1,
        selected_ufh_sample_count=1,
        selected_dhw_sample_count=0,
        bucket_rejection_counts=(("electric_energy_below_min", 1), ("mode_not_ufh_or_dhw", 3)),
        segment_failure_counts=(("actual_cop_span", 1), ("sample_count", 1)),
        segment_qualities=dataset.segment_qualities,
    )


def test_calibration_cli_cop_json_reports_segment_and_mode_diagnostics(
    monkeypatch,
    capsys,
) -> None:
    """The COP CLI JSON output must expose richer segment-selection and per-mode fit diagnostics."""
    dataset = _build_cop_cli_dataset()
    result = _build_cop_cli_result(dataset)
    monkeypatch.setattr(
        calibration_main,
        "build_cop_dataset_from_repository",
        lambda repository, settings: dataset,
    )
    monkeypatch.setattr(
        calibration_main,
        "calibrate_cop_from_repository",
        lambda repository, settings: result,
    )
    monkeypatch.setattr(sys, "argv", ["home-optimizer-calibration", "--stage", "cop", "--json"])

    calibration_main.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["dataset"]["selected_segment_count"] == 1
    assert payload["dataset"]["selected_ufh_segment_count"] == 1
    assert payload["dataset"]["selected_dhw_segment_count"] == 0
    assert payload["dataset"]["raw_segment_count"] == 2
    assert payload["dataset"]["dropped_segment_count"] == 1
    assert payload["fit"]["ufh_rmse_electric_energy_kwh"] == result.ufh_rmse_electric_energy_kwh
    assert payload["fit"]["dhw_rmse_actual_cop"] == result.dhw_rmse_actual_cop
    assert payload["fit"]["diagnostic_eta_carnot_dhw"] == result.diagnostic_eta_carnot_dhw
    assert payload["fit"]["t_ref_outdoor_was_fitted"] is True


def test_calibration_cli_cop_text_reports_mode_metrics_and_segment_counts(
    monkeypatch,
    capsys,
) -> None:
    """The human-readable COP CLI output must show segment counts and per-mode diagnostics."""
    dataset = _build_cop_cli_dataset()
    result = _build_cop_cli_result(dataset)
    monkeypatch.setattr(
        calibration_main,
        "build_cop_dataset_from_repository",
        lambda repository, settings: dataset,
    )
    monkeypatch.setattr(
        calibration_main,
        "calibrate_cop_from_repository",
        lambda repository, settings: result,
    )
    monkeypatch.setattr(sys, "argv", ["home-optimizer-calibration", "--stage", "cop"])

    calibration_main.main()

    output = capsys.readouterr().out
    assert "Selected segments    : 1" in output
    assert "Raw segments         : 2" in output
    assert "Dropped segments     : 1" in output
    assert "UFH RMSE(E_elec)" in output
    assert "DHW RMSE(COP)" in output
    assert "UFH diagnostic eta" in output
    assert "T_ref_outdoor" in output
    assert "T_ref fitted         : True" in output
    assert "Heating-curve loss   : soft_l1" in output


def test_calibration_cli_cop_diagnostics_json_reports_rejection_counts(
    monkeypatch,
    capsys,
) -> None:
    """The COP diagnostics CLI must expose bucket and segment rejection counts in JSON."""
    diagnostics = _build_cop_cli_diagnostics()
    monkeypatch.setattr(
        calibration_main,
        "diagnose_cop_dataset_from_repository",
        lambda repository, settings: diagnostics,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["home-optimizer-calibration", "--stage", "cop", "--cop-diagnostics", "--json"],
    )

    calibration_main.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["diagnostics"]["raw_row_count"] == diagnostics.raw_row_count
    assert payload["diagnostics"]["selected_segment_count"] == diagnostics.selected_segment_count
    assert payload["diagnostics"]["bucket_rejection_counts"] == [
        ["electric_energy_below_min", 1],
        ["mode_not_ufh_or_dhw", 3],
    ]
    assert payload["diagnostics"]["segment_failure_counts"] == [
        ["actual_cop_span", 1],
        ["sample_count", 1],
    ]


