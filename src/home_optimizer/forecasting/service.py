"""Runtime machine-learning forecast service.

The service enriches optimizer requests with optional horizon arrays that are not
available from Open-Meteo directly. Each provider is independent and only fills a
field when the caller has not already supplied it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from .baseload import BaseloadForecaster
from .dhw_tap import DHWTapForecaster
from .models import ForecastServiceSettings
from .shutter import ShutterForecaster

if TYPE_CHECKING:
    from ..telemetry.repository import TelemetryRepository


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
    def train_and_persist(self, *, repository: "TelemetryRepository") -> object | None:
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

    def train_and_persist(self, *, repository: "TelemetryRepository") -> object | None:
        """Train and persist the disk-backed shutter model artifact."""

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

    def train_and_persist(self, *, repository: "TelemetryRepository") -> object | None:
        """Train and persist the disk-backed baseload model artifact."""

        return self._forecaster.train_and_persist_from_repository(repository=repository)


class DHWTapForecastProvider(ForecastProvider):
    """Build a recurring hour-of-day DHW tap-flow forecast from persisted telemetry."""

    def __init__(self, forecaster: DHWTapForecaster | None = None) -> None:
        self._forecaster = forecaster or DHWTapForecaster()

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
        for field_name, raw_value in (
            ("dhw_C_top", c_top_raw),
            ("dhw_C_bot", c_bot_raw),
            ("dhw_R_loss", r_loss_raw),
            ("dhw_lambda_water_kwh_per_m3k", lambda_raw),
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
        )
        if predicted is None:
            return {}
        return {"dhw_v_tap_forecast": predicted.tolist()}

    def train_and_persist(self, *, repository: "TelemetryRepository") -> object | None:
        """No-op: the DHW tap forecast is derived directly from telemetry at runtime."""
        _ = repository
        return None


class ForecastService:
    """Compose multiple ML forecast providers into one runtime enrichment step.

    The current service only predicts ``shutter_forecast``. The provider-based
    design keeps the API stable when a later baseload forecaster is added.
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

    def train_and_persist_models(self, *, repository: "TelemetryRepository") -> dict[str, object | None]:
        """Train every configured forecast provider and persist its model artifact.

        Returns:
            Mapping from provider field name to training metadata, or ``None`` when
            a provider skipped training because insufficient history was available.
        """

        return {
            field_name: provider.train_and_persist(repository=repository)
            for field_name, provider in self._providers
        }

