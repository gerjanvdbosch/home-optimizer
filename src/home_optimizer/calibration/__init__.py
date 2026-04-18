"""Offline calibration utilities for learning thermal parameters from telemetry."""

from .dataset import (
    build_dhw_active_calibration_dataset,
    build_dhw_standby_calibration_dataset,
    build_ufh_active_calibration_dataset,
    build_ufh_off_calibration_dataset,
)
from .dhw_active import calibrate_dhw_active_stratification
from .dhw_standby import calibrate_dhw_standby_loss
from .models import (
    DHWActiveCalibrationDataset,
    DHWActiveCalibrationResult,
    DHWActiveCalibrationSample,
    DHWActiveCalibrationSettings,
    DHWStandbyCalibrationDataset,
    DHWStandbyCalibrationResult,
    DHWStandbyCalibrationSample,
    DHWStandbyCalibrationSettings,
    UFHActiveCalibrationDataset,
    UFHActiveCalibrationResult,
    UFHActiveCalibrationSegmentQuality,
    UFHActiveCalibrationSample,
    UFHActiveCalibrationSettings,
    UFHCalibrationDataset,
    UFHCalibrationSample,
    UFHOffCalibrationResult,
    UFHOffCalibrationSettings,
)
from .service import (
    build_dhw_active_dataset_from_repository,
    build_dhw_standby_dataset_from_repository,
    build_ufh_active_dataset_from_repository,
    build_ufh_off_dataset_from_repository,
    calibrate_dhw_active_from_repository,
    calibrate_dhw_standby_from_repository,
    calibrate_ufh_active_from_repository,
    calibrate_ufh_off_from_repository,
)
from .ufh_active import calibrate_ufh_active_rc
from .ufh_offline import calibrate_ufh_off_envelope

__all__ = [
    "DHWActiveCalibrationDataset",
    "DHWActiveCalibrationResult",
    "DHWActiveCalibrationSample",
    "DHWActiveCalibrationSettings",
    "UFHActiveCalibrationDataset",
    "UFHActiveCalibrationResult",
    "UFHActiveCalibrationSegmentQuality",
    "UFHActiveCalibrationSample",
    "UFHActiveCalibrationSettings",
    "DHWStandbyCalibrationDataset",
    "DHWStandbyCalibrationResult",
    "DHWStandbyCalibrationSample",
    "DHWStandbyCalibrationSettings",
    "UFHCalibrationDataset",
    "UFHCalibrationSample",
    "UFHOffCalibrationResult",
    "UFHOffCalibrationSettings",
    "build_dhw_active_calibration_dataset",
    "build_dhw_active_dataset_from_repository",
    "build_dhw_standby_calibration_dataset",
    "build_dhw_standby_dataset_from_repository",
    "build_ufh_active_calibration_dataset",
    "build_ufh_active_dataset_from_repository",
    "build_ufh_off_calibration_dataset",
    "build_ufh_off_dataset_from_repository",
    "calibrate_dhw_active_from_repository",
    "calibrate_dhw_active_stratification",
    "calibrate_dhw_standby_from_repository",
    "calibrate_dhw_standby_loss",
    "calibrate_ufh_active_from_repository",
    "calibrate_ufh_active_rc",
    "calibrate_ufh_off_envelope",
    "calibrate_ufh_off_from_repository",
]

