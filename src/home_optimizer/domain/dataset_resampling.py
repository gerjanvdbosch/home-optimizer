from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .names import (
    BOOSTER_HEATER_ACTIVE,
    DEFROST_ACTIVE,
    DHW_BOTTOM_TEMPERATURE,
    DHW_TOP_TEMPERATURE,
    GTI_LIVING_ROOM_WINDOWS,
    HP_ELECTRIC_POWER,
    HP_FLOW,
    HP_MODE,
    HP_RETURN_TEMPERATURE,
    HP_SUPPLY_TEMPERATURE,
    OUTDOOR_TEMPERATURE,
    P1_NET_POWER,
    PV_OUTPUT_POWER,
    ROOM_TEMPERATURE,
    SHUTTER_LIVING_ROOM,
    THERMOSTAT_SETPOINT,
)

DatasetNumericResampleMethod = Literal["sample", "mean", "window_flag"]
DatasetTextResampleMethod = Literal["sample"]
DatasetSeriesSource = Literal["measurement", "forecast"]


@dataclass(frozen=True)
class DatasetNumericSignalSpec:
    name: str
    source: DatasetSeriesSource
    resample_method: DatasetNumericResampleMethod


@dataclass(frozen=True)
class DatasetTextSignalSpec:
    name: str
    source: DatasetSeriesSource
    resample_method: DatasetTextResampleMethod


DATASET_NUMERIC_SIGNAL_SPECS = (
    DatasetNumericSignalSpec(
        name=ROOM_TEMPERATURE,
        source="measurement",
        resample_method="sample",
    ),
    DatasetNumericSignalSpec(
        name=OUTDOOR_TEMPERATURE,
        source="measurement",
        resample_method="sample",
    ),
    DatasetNumericSignalSpec(
        name=DHW_TOP_TEMPERATURE,
        source="measurement",
        resample_method="sample",
    ),
    DatasetNumericSignalSpec(
        name=DHW_BOTTOM_TEMPERATURE,
        source="measurement",
        resample_method="sample",
    ),
    DatasetNumericSignalSpec(
        name=SHUTTER_LIVING_ROOM,
        source="measurement",
        resample_method="sample",
    ),
    DatasetNumericSignalSpec(
        name=THERMOSTAT_SETPOINT,
        source="measurement",
        resample_method="sample",
    ),
    DatasetNumericSignalSpec(
        name=HP_ELECTRIC_POWER,
        source="measurement",
        resample_method="mean",
    ),
    DatasetNumericSignalSpec(
        name=PV_OUTPUT_POWER,
        source="measurement",
        resample_method="mean",
    ),
    DatasetNumericSignalSpec(
        name=P1_NET_POWER,
        source="measurement",
        resample_method="mean",
    ),
    DatasetNumericSignalSpec(
        name=HP_SUPPLY_TEMPERATURE,
        source="measurement",
        resample_method="mean",
    ),
    DatasetNumericSignalSpec(
        name=HP_RETURN_TEMPERATURE,
        source="measurement",
        resample_method="mean",
    ),
    DatasetNumericSignalSpec(
        name=HP_FLOW,
        source="measurement",
        resample_method="mean",
    ),
    DatasetNumericSignalSpec(
        name=GTI_LIVING_ROOM_WINDOWS,
        source="forecast",
        resample_method="mean",
    ),
    DatasetNumericSignalSpec(
        name=DEFROST_ACTIVE,
        source="measurement",
        resample_method="window_flag",
    ),
    DatasetNumericSignalSpec(
        name=BOOSTER_HEATER_ACTIVE,
        source="measurement",
        resample_method="window_flag",
    ),
)

DATASET_TEXT_SIGNAL_SPECS = (
    DatasetTextSignalSpec(
        name=HP_MODE,
        source="measurement",
        resample_method="sample",
    ),
)

_NUMERIC_SPEC_BY_NAME = {spec.name: spec for spec in DATASET_NUMERIC_SIGNAL_SPECS}
_TEXT_SPEC_BY_NAME = {spec.name: spec for spec in DATASET_TEXT_SIGNAL_SPECS}


def dataset_numeric_signal_names(
    *,
    source: DatasetSeriesSource | None = None,
) -> list[str]:
    return [
        spec.name
        for spec in DATASET_NUMERIC_SIGNAL_SPECS
        if source is None or spec.source == source
    ]


def dataset_text_signal_names(
    *,
    source: DatasetSeriesSource | None = None,
) -> list[str]:
    return [
        spec.name
        for spec in DATASET_TEXT_SIGNAL_SPECS
        if source is None or spec.source == source
    ]


def dataset_numeric_signal_spec(name: str) -> DatasetNumericSignalSpec:
    spec = _NUMERIC_SPEC_BY_NAME.get(name)
    if spec is None:
        raise ValueError(f"Unknown dataset numeric signal: {name}")
    return spec


def dataset_text_signal_spec(name: str) -> DatasetTextSignalSpec:
    spec = _TEXT_SPEC_BY_NAME.get(name)
    if spec is None:
        raise ValueError(f"Unknown dataset text signal: {name}")
    return spec
