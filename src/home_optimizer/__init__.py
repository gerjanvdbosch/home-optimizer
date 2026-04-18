"""Home Optimizer – UFH + DHW thermal model with Kalman filter and combined MPC.

Physical model assumptions
--------------------------
* UFH (§3–§6): 2-state grey-box model [T_r, T_b]; heat transport via Newton's
  law; solar irradiance split by fraction α; forward-Euler discretisation.
* DHW (§7–§12): 2-node stratification tank [T_top, T_bot]; time-varying LTV
  state-space (depends on V_tap[k]); λ = 1.1628 kWh/(m³·K); tap-stream split
  correctly between layers.
* Combined MPC (§13–§14): block-diagonal state-space; shared heat-pump power
  budget; legionella constraint as stiff soft constraint.
* All energy quantities use kW (power) and kWh (energy); temperatures in °C.
* Discretisation: forward-Euler with time step dt_hours.
"""

from .cop_model import T_CELSIUS_TO_KELVIN, HeatPumpCOPModel, HeatPumpCOPParameters
from .calibration import (
    DHWActiveCalibrationDataset,
    DHWActiveCalibrationResult,
    DHWActiveCalibrationSample,
    DHWActiveCalibrationSettings,
    build_dhw_active_calibration_dataset,
    build_dhw_active_dataset_from_repository,
    calibrate_dhw_active_from_repository,
    calibrate_dhw_active_stratification,
    DHWStandbyCalibrationDataset,
    DHWStandbyCalibrationResult,
    DHWStandbyCalibrationSample,
    DHWStandbyCalibrationSettings,
    build_dhw_standby_calibration_dataset,
    build_dhw_standby_dataset_from_repository,
    calibrate_dhw_standby_from_repository,
    calibrate_dhw_standby_loss,
    UFHActiveCalibrationDataset,
    UFHActiveCalibrationResult,
    UFHActiveCalibrationSegmentQuality,
    UFHActiveCalibrationSample,
    UFHActiveCalibrationSettings,
    UFHCalibrationDataset,
    UFHCalibrationSample,
    UFHOffCalibrationResult,
    UFHOffCalibrationSettings,
    build_ufh_active_calibration_dataset,
    calibrate_ufh_active_from_repository,
    calibrate_ufh_active_rc,
    build_ufh_off_calibration_dataset,
    calibrate_ufh_off_envelope,
    calibrate_ufh_off_from_repository,
)
from .dhw_model import MEASUREMENT_MATRIX_DHW, DHWModel
from .kalman import (
    DHWKalmanFilter,
    ExtendedKalmanFilter,
    KalmanEstimate,
    LinearKalmanFilter,
    UFHKalmanFilter,
)
from .mpc import MPCController, MPCSolution
from .telemetry import (
    BufferedTelemetryCollector,
    TelemetryAggregate,
    TelemetryCollectorSettings,
    TelemetryRepository,
    aggregate_readings,
)
from .thermal_model import MEASUREMENT_MATRIX, ThermalModel, solar_gain_kw
from .types import (
    W_PER_KW,
    CombinedMPCParameters,
    DHWForecastHorizon,
    DHWMPCParameters,
    DHWParameters,
    ForecastHorizon,
    KalmanNoiseParameters,
    MPCParameters,
    ThermalParameters,
)

__all__ = [
    "CombinedMPCParameters",
    "DHWActiveCalibrationDataset",
    "DHWActiveCalibrationResult",
    "DHWActiveCalibrationSample",
    "DHWActiveCalibrationSettings",
    "DHWStandbyCalibrationDataset",
    "DHWStandbyCalibrationResult",
    "DHWStandbyCalibrationSample",
    "DHWStandbyCalibrationSettings",
    "DHWForecastHorizon",
    "DHWKalmanFilter",
    "DHWMPCParameters",
    "DHWModel",
    "DHWParameters",
    "ExtendedKalmanFilter",
    "ForecastHorizon",
    "HeatPumpCOPModel",
    "HeatPumpCOPParameters",
    "KalmanEstimate",
    "KalmanNoiseParameters",
    "LinearKalmanFilter",
    "MEASUREMENT_MATRIX",
    "MEASUREMENT_MATRIX_DHW",
    "MPCParameters",
    "MPCController",
    "MPCSolution",
    "T_CELSIUS_TO_KELVIN",
    "TelemetryAggregate",
    "TelemetryCollectorSettings",
    "TelemetryRepository",
    "ThermalModel",
    "ThermalParameters",
    "UFHActiveCalibrationDataset",
    "UFHActiveCalibrationResult",
    "UFHActiveCalibrationSegmentQuality",
    "UFHActiveCalibrationSample",
    "UFHActiveCalibrationSettings",
    "UFHCalibrationDataset",
    "UFHCalibrationSample",
    "UFHOffCalibrationResult",
    "UFHOffCalibrationSettings",
    "BufferedTelemetryCollector",
    "UFHKalmanFilter",
    "W_PER_KW",
    "aggregate_readings",
    "build_dhw_active_calibration_dataset",
    "build_dhw_active_dataset_from_repository",
    "build_dhw_standby_calibration_dataset",
    "build_dhw_standby_dataset_from_repository",
    "build_ufh_active_calibration_dataset",
    "build_ufh_off_calibration_dataset",
    "calibrate_dhw_active_from_repository",
    "calibrate_dhw_active_stratification",
    "calibrate_dhw_standby_from_repository",
    "calibrate_dhw_standby_loss",
    "calibrate_ufh_active_from_repository",
    "calibrate_ufh_active_rc",
    "calibrate_ufh_off_envelope",
    "calibrate_ufh_off_from_repository",
    "solar_gain_kw",
]
