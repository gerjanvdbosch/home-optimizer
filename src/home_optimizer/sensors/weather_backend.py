"""Weather-augmented sensor backend.

Wraps any :class:`SensorBackend` and injects a seasonal T_mains estimate into
every :class:`LiveReadings` snapshot.

Architecture
------------
* The *inner* backend (e.g. :class:`HomeAssistantBackend`) handles all HA sensor
  readings but produces a placeholder value (``0.0``) for ``t_mains_estimated_c``.
* :class:`WeatherAugmentedBackend` evaluates the :class:`SeasonalMainsModel`
  (a pure date function, no network I/O) on every ``read_all()`` call and uses
  ``dataclasses.replace()`` to overwrite the field before returning the snapshot.
* Forecast data (GTI for windows/PV, T_out forecast) lives exclusively in the
  ``forecast_snapshots`` table and is managed by
  :class:`~home_optimizer.telemetry.ForecastPersister`, which owns its own
  :class:`~home_optimizer.sensors.OpenMeteoClient` instance.
  This keeps update cadences cleanly separated: T_mains is a daily-changing
  physical DHW parameter; forecasts change hourly.

Typical production setup::

    from home_optimizer.sensors import (
        HomeAssistantBackend, HAEntityConfig,
        SeasonalMainsModel, WeatherAugmentedBackend,
    )

    ha_backend = HomeAssistantBackend(...)
    mains_model = SeasonalMainsModel.for_netherlands()
    backend = WeatherAugmentedBackend(ha_backend, mains_model)
    readings = backend.read_all()
    # readings.t_mains_estimated_c  ← seasonal cold mains estimate [°C]

Units
-----
T_mains : °C
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

from .base import LiveReadings, SensorBackend
from .open_meteo import SeasonalMainsModel


class WeatherAugmentedBackend(SensorBackend):
    """SensorBackend wrapper that injects a seasonal T_mains estimate into readings.

    Replaces the ``t_mains_estimated_c`` placeholder from the inner backend with
    a date-derived value from the :class:`SeasonalMainsModel` on every
    ``read_all()`` call.  No network I/O; no caching required.

    Parameters
    ----------
    inner:
        Any :class:`SensorBackend` that populates all non-weather
        :class:`LiveReadings` fields.  Its ``t_mains_estimated_c`` placeholder
        (typically ``0.0``) is discarded and replaced.
    mains_model:
        :class:`SeasonalMainsModel` instance.  Use
        ``SeasonalMainsModel.for_netherlands()`` for Dutch sites.
    """

    def __init__(
        self,
        inner: SensorBackend,
        mains_model: SeasonalMainsModel,
    ) -> None:
        if not isinstance(inner, SensorBackend):
            raise TypeError(
                f"inner must be a SensorBackend instance, got {type(inner).__name__}."
            )
        self._inner = inner
        self._mains = mains_model

    # ------------------------------------------------------------------
    # SensorBackend interface
    # ------------------------------------------------------------------

    def read_all(self) -> LiveReadings:
        """Read sensor snapshot from the inner backend and inject T_mains.

        Sequence:
        1. Call ``inner.read_all()`` for all HA/local sensor values.
        2. Compute T_mains from the seasonal model (pure date function, no I/O).
        3. Replace ``t_mains_estimated_c`` using ``dataclasses.replace()``.

        Returns
        -------
        LiveReadings
            Fully populated snapshot with seasonal T_mains injected.
        """
        reading = self._inner.read_all()
        t_mains = self._mains.estimate(datetime.now(tz=timezone.utc))
        # dataclasses.replace() works on frozen dataclasses with __slots__:
        # it creates a new instance with the specified field overwritten.
        return dataclasses.replace(reading, t_mains_estimated_c=t_mains)

    def close(self) -> None:
        """Close the inner backend's resources.

        The seasonal model is stateless; only the inner backend may hold connections.
        """
        self._inner.close()
