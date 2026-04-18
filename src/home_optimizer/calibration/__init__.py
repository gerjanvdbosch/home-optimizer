"""Offline calibration utilities for learning thermal parameters from telemetry."""

from .dataset import build_ufh_off_calibration_dataset
from .models import (
    UFHCalibrationDataset,
    UFHCalibrationSample,
    UFHOffCalibrationResult,
    UFHOffCalibrationSettings,
)
from .service import build_ufh_off_dataset_from_repository, calibrate_ufh_off_from_repository
from .ufh_offline import calibrate_ufh_off_envelope

__all__ = [
    "UFHCalibrationDataset",
    "UFHCalibrationSample",
    "UFHOffCalibrationResult",
    "UFHOffCalibrationSettings",
    "build_ufh_off_calibration_dataset",
    "build_ufh_off_dataset_from_repository",
    "calibrate_ufh_off_envelope",
    "calibrate_ufh_off_from_repository",
]

