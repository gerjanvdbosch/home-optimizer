"""Runtime machine-learning forecast service.

The service enriches optimizer requests with optional horizon arrays that are not
available from Open-Meteo directly. Each provider is independent and only fills a
field when the caller has not already supplied it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

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
        horizon_steps = int(horizon_raw)

        shutter_raw = current_overrides.get("shutter_living_room_pct")
        if shutter_raw is None:
            shutter_raw = request_data.get("shutter_living_room_pct", 100.0)
        initial_shutter_pct = float(shutter_raw)
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


class ForecastService:
    """Compose multiple ML forecast providers into one runtime enrichment step.

    The current service only predicts ``shutter_forecast``. The provider-based
    design keeps the API stable when a later baseload forecaster is added.
    """

    def __init__(self, settings: ForecastServiceSettings | None = None) -> None:
        effective_settings = settings or ForecastServiceSettings()
        self._providers: tuple[tuple[str, ForecastProvider], ...] = (
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

