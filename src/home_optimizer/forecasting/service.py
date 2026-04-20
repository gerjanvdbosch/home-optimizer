"""Runtime machine-learning forecast service.

The service enriches optimizer requests with optional horizon arrays that are not
available from Open-Meteo directly. Each provider is independent and only fills a
field when the caller has not already supplied it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from datetime import datetime
import logging
from typing import TYPE_CHECKING, Any, cast

from .baseload import BaseloadForecaster
from .dhw_tap import DHWTapForecaster
from .models import ForecastServiceSettings
from .shutter import ShutterForecaster

if TYPE_CHECKING:
    from ..telemetry.repository import TelemetryRepository


log = logging.getLogger("home_optimizer.forecasting.service")


class ForecastProvider(ABC):
    """Abstract provider for one or more runtime request forecast overrides."""

    @abstractmethod
    def build_overrides(
        self,
        *,
        request_data: Mapping[str, object],
        repository: "TelemetryRepository",
        weather_rows: Sequence[Any],
        current_overrides: Mapping[str, object],
    ) -> dict[str, object]:
        """Return zero or more additional request overrides."""

    @abstractmethod
    def train_and_persist(
        self,
        *,
        repository: "TelemetryRepository",
        base_request_data: Mapping[str, object] | None = None,
    ) -> object | None:
        """Train the provider model on repository history and persist it to disk."""


class ShutterForecastProvider(ForecastProvider):
    """Predict ``shutter_forecast`` with a scikit-learn autoregressive regressor."""

    def __init__(self, forecaster: ShutterForecaster | None = None) -> None:
        self._forecaster = forecaster or ShutterForecaster()

    def build_overrides(
        self,
        *,
        request_data: Mapping[str, object],
        repository: "TelemetryRepository",
        weather_rows: Sequence[Any],
        current_overrides: Mapping[str, object],
    ) -> dict[str, object]:
        explicit_request_forecast = request_data.get("shutter_forecast")
        if explicit_request_forecast is not None or "shutter_forecast" in current_overrides:
            return {}

        horizon_raw = request_data.get("horizon_hours")
        if horizon_raw is None:
            raise ValueError("request_data must contain horizon_hours for shutter forecasting.")
        if not isinstance(horizon_raw, (int, float)):
            raise ValueError("horizon_hours must be numeric for shutter forecasting.")
        horizon_steps = int(cast(int | float, horizon_raw))

        shutter_raw = current_overrides.get("shutter_living_room_pct")
        if shutter_raw is None:
            shutter_raw = request_data.get("shutter_living_room_pct", 100.0)
        if not isinstance(shutter_raw, (int, float)):
            raise ValueError("shutter_living_room_pct must be numeric for shutter forecasting.")
        initial_shutter_pct = float(cast(int | float, shutter_raw))
        predicted = self._forecaster.predict_from_repository(
            repository=repository,
            weather_rows=weather_rows,
            horizon_steps=horizon_steps,
            initial_shutter_pct=initial_shutter_pct,
        )
        if predicted is None:
            return {}
        return {"shutter_forecast": predicted.tolist()}

    def train_and_persist(
        self,
        *,
        repository: "TelemetryRepository",
        base_request_data: Mapping[str, object] | None = None,
    ) -> object | None:
        """Train and persist the disk-backed shutter model artifact."""

        _ = base_request_data
        return self._forecaster.train_and_persist_from_repository(repository=repository)


class BaseloadForecastProvider(ForecastProvider):
    """Predict a persisted household baseload forecast."""

    def __init__(self, forecaster: BaseloadForecaster | None = None) -> None:
        self._forecaster = forecaster or BaseloadForecaster()

    def build_overrides(
        self,
        *,
        request_data: Mapping[str, object],
        repository: "TelemetryRepository",
        weather_rows: Sequence[Any],
        current_overrides: Mapping[str, object],
    ) -> dict[str, object]:
        explicit_baseload = request_data.get("baseload_forecast")
        horizon_raw = request_data.get("horizon_hours")
        if horizon_raw is None:
            raise ValueError("request_data must contain horizon_hours for baseload forecasting.")
        if not isinstance(horizon_raw, (int, float)):
            raise ValueError("horizon_hours must be numeric for baseload forecasting.")
        horizon_steps = int(cast(int | float, horizon_raw))
        if explicit_baseload is not None or "baseload_forecast" in current_overrides:
            baseload_forecast = current_overrides.get("baseload_forecast", explicit_baseload)
        else:
            baseload_prediction = self._forecaster.predict_from_repository(
                repository=repository,
                weather_rows=weather_rows,
                horizon_steps=horizon_steps,
            )
            if baseload_prediction is None:
                return {}
            baseload_forecast = baseload_prediction.tolist()

        return {"baseload_forecast": baseload_forecast}

    def train_and_persist(
        self,
        *,
        repository: "TelemetryRepository",
        base_request_data: Mapping[str, object] | None = None,
    ) -> object | None:
        """Train and persist the disk-backed baseload model artifact."""

        _ = base_request_data
        return self._forecaster.train_and_persist_from_repository(repository=repository)


class DHWTapForecastProvider(ForecastProvider):
    """Build and train recurring hour-of-day DHW tap-flow forecasts from telemetry."""

    def __init__(self, forecaster: DHWTapForecaster | None = None) -> None:
        self._forecaster = forecaster or DHWTapForecaster()

    @staticmethod
    def _resolve_numeric_field(
        *,
        field_name: str,
        calibration_value: object | None,
        base_request_data: Mapping[str, object] | None,
    ) -> float | None:
        """Return one numeric DHW field from calibration overrides or the base request."""

        if calibration_value is not None:
            if not isinstance(calibration_value, (int, float)):
                raise ValueError(f"{field_name} calibration override must be numeric.")
            return float(cast(int | float, calibration_value))
        if base_request_data is None:
            return None
        raw_value = base_request_data.get(field_name)
        if raw_value is None:
            return None
        if not isinstance(raw_value, (int, float)):
            raise ValueError(f"{field_name} in base_request_data must be numeric.")
        return float(cast(int | float, raw_value))

    def build_overrides(
        self,
        *,
        request_data: Mapping[str, object],
        repository: "TelemetryRepository",
        weather_rows: Sequence[Any],
        current_overrides: Mapping[str, object],
    ) -> dict[str, object]:
        explicit_request_forecast = request_data.get("dhw_v_tap_forecast")
        if explicit_request_forecast is not None or "dhw_v_tap_forecast" in current_overrides:
            return {}
        if not bool(request_data.get("dhw_enabled", True)):
            return {}
        if not weather_rows:
            return {}

        c_top_raw = current_overrides.get("dhw_C_top", request_data.get("dhw_C_top"))
        c_bot_raw = current_overrides.get("dhw_C_bot", request_data.get("dhw_C_bot"))
        r_loss_raw = current_overrides.get("dhw_R_loss", request_data.get("dhw_R_loss"))
        lambda_raw = current_overrides.get(
            "dhw_lambda_water_kwh_per_m3k",
            request_data.get("dhw_lambda_water_kwh_per_m3k"),
        )
        top_bias_raw = current_overrides.get(
            "dhw_top_temperature_bias_c",
            request_data.get("dhw_top_temperature_bias_c"),
        )
        bottom_bias_raw = current_overrides.get(
            "dhw_bottom_temperature_bias_c",
            request_data.get("dhw_bottom_temperature_bias_c"),
        )
        ambient_bias_raw = current_overrides.get(
            "dhw_boiler_ambient_bias_c",
            request_data.get("dhw_boiler_ambient_bias_c"),
        )
        for field_name, raw_value in (
            ("dhw_C_top", c_top_raw),
            ("dhw_C_bot", c_bot_raw),
            ("dhw_R_loss", r_loss_raw),
            ("dhw_lambda_water_kwh_per_m3k", lambda_raw),
            ("dhw_top_temperature_bias_c", top_bias_raw),
            ("dhw_bottom_temperature_bias_c", bottom_bias_raw),
            ("dhw_boiler_ambient_bias_c", ambient_bias_raw),
        ):
            if not isinstance(raw_value, (int, float)):
                return {}

        horizon_valid_at_utc: list[datetime] = []
        for row in weather_rows:
            valid_at_utc = getattr(row, "valid_at_utc", None)
            if not isinstance(valid_at_utc, datetime):
                raise ValueError("weather_rows must expose datetime valid_at_utc values for DHW tap forecasting.")
            horizon_valid_at_utc.append(valid_at_utc)

        predicted = self._forecaster.predict_from_repository(
            repository=repository,
            horizon_valid_at_utc=horizon_valid_at_utc,
            c_top_kwh_per_k=float(cast(int | float, c_top_raw)),
            c_bot_kwh_per_k=float(cast(int | float, c_bot_raw)),
            r_loss_k_per_kw=float(cast(int | float, r_loss_raw)),
            lambda_water_kwh_per_m3_k=float(cast(int | float, lambda_raw)),
            top_temperature_bias_c=float(cast(int | float, top_bias_raw)),
            bottom_temperature_bias_c=float(cast(int | float, bottom_bias_raw)),
            boiler_ambient_bias_c=float(cast(int | float, ambient_bias_raw)),
        )
        if predicted is None:
            return {}
        return {"dhw_v_tap_forecast": predicted.tolist()}

    def train_and_persist(
        self,
        *,
        repository: "TelemetryRepository",
        base_request_data: Mapping[str, object] | None = None,
    ) -> object | None:
        """Train and persist the DHW tap profile using the effective DHW tuple.

        The inferred tap-flow profile depends on the physical DHW tank parameters
        and the active sensor-bias corrections. The effective training tuple is the
        runtime base request with any persisted calibration overrides layered on top.
        This preserves the configured tank volume even when the latest calibration
        snapshot only updates ``R_loss`` or sensor biases.
        """

        calibration_snapshot = repository.get_latest_calibration_snapshot()
        effective_parameters = calibration_snapshot.effective_parameters if calibration_snapshot is not None else None
        c_top = self._resolve_numeric_field(
            field_name="dhw_C_top",
            calibration_value=None if effective_parameters is None else effective_parameters.dhw_C_top,
            base_request_data=base_request_data,
        )
        c_bot = self._resolve_numeric_field(
            field_name="dhw_C_bot",
            calibration_value=None if effective_parameters is None else effective_parameters.dhw_C_bot,
            base_request_data=base_request_data,
        )
        r_loss = self._resolve_numeric_field(
            field_name="dhw_R_loss",
            calibration_value=None if effective_parameters is None else effective_parameters.dhw_R_loss,
            base_request_data=base_request_data,
        )
        lambda_water = self._resolve_numeric_field(
            field_name="dhw_lambda_water_kwh_per_m3k",
            calibration_value=None,
            base_request_data=base_request_data,
        )
        top_bias = self._resolve_numeric_field(
            field_name="dhw_top_temperature_bias_c",
            calibration_value=None if effective_parameters is None else effective_parameters.dhw_top_temperature_bias_c,
            base_request_data=base_request_data,
        )
        bottom_bias = self._resolve_numeric_field(
            field_name="dhw_bottom_temperature_bias_c",
            calibration_value=None if effective_parameters is None else effective_parameters.dhw_bottom_temperature_bias_c,
            base_request_data=base_request_data,
        )
        ambient_bias = self._resolve_numeric_field(
            field_name="dhw_boiler_ambient_bias_c",
            calibration_value=None if effective_parameters is None else effective_parameters.dhw_boiler_ambient_bias_c,
            base_request_data=base_request_data,
        )
        if (
            c_top is None
            or c_bot is None
            or r_loss is None
            or lambda_water is None
            or top_bias is None
            or bottom_bias is None
            or ambient_bias is None
        ):
            log.info(
                "Skipping DHW tap-profile training: the effective DHW runtime tuple is incomplete. "
                "Provide base_request_data and/or calibration overrides for dhw_C_top, dhw_C_bot, dhw_R_loss, "
                "dhw_lambda_water_kwh_per_m3k, dhw_top_temperature_bias_c, dhw_bottom_temperature_bias_c, "
                "and dhw_boiler_ambient_bias_c."
            )
            return None

        return self._forecaster.train_and_persist_from_repository(
            repository=repository,
            c_top_kwh_per_k=float(c_top),
            c_bot_kwh_per_k=float(c_bot),
            r_loss_k_per_kw=float(r_loss),
            lambda_water_kwh_per_m3_k=float(lambda_water),
            top_temperature_bias_c=float(top_bias),
            bottom_temperature_bias_c=float(bottom_bias),
            boiler_ambient_bias_c=float(ambient_bias),
        )


class ForecastService:
    """Compose multiple ML forecast providers into one runtime enrichment step.

    Providers currently cover DHW tap-flow, household baseload, and living-room
    shutters. The provider-based design keeps the API stable when more forecast
    enrichments are added.
    """

    def __init__(self, settings: ForecastServiceSettings | None = None) -> None:
        effective_settings = settings or ForecastServiceSettings()
        self._providers: tuple[tuple[str, ForecastProvider], ...] = (
            ("dhw_v_tap_forecast", DHWTapForecastProvider(DHWTapForecaster(effective_settings.dhw_tap))),
            ("baseload_forecast", BaseloadForecastProvider(BaseloadForecaster(effective_settings.baseload))),
            ("shutter_forecast", ShutterForecastProvider(ShutterForecaster(effective_settings.shutter))),
        )

    def build_missing_overrides(
        self,
        *,
        request_data: Mapping[str, object],
        repository: "TelemetryRepository",
        weather_rows: Sequence[Any],
        current_overrides: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Return ML-generated forecast overrides that are still missing.

        Args:
            request_data: Materialized request-like mapping with scalar runtime
                settings and any explicit user-supplied forecasts.
            repository: Telemetry repository used as the training source.
            weather_rows: Future hourly forecast rows for the current horizon.
            current_overrides: Overrides already assembled by the caller.

        Returns:
            Dict containing only newly predicted fields.
        """

        incoming_overrides = dict(current_overrides or {})
        produced_overrides: dict[str, object] = {}
        effective_overrides = dict(incoming_overrides)
        for _field_name, provider in self._providers:
            produced = provider.build_overrides(
                request_data=request_data,
                repository=repository,
                weather_rows=weather_rows,
                current_overrides=effective_overrides,
            )
            effective_overrides.update(produced)
            produced_overrides.update(produced)
        return produced_overrides

    def train_and_persist_models(
        self,
        *,
        repository: "TelemetryRepository",
        base_request_data: Mapping[str, object] | None = None,
    ) -> dict[str, object | None]:
        """Train every configured forecast provider and persist its model artifact.

        Returns:
            Mapping from provider field name to training metadata, or ``None`` when
            a provider skipped training because insufficient history was available.
        """

        return {
            field_name: provider.train_and_persist(repository=repository, base_request_data=base_request_data)
            for field_name, provider in self._providers
        }

