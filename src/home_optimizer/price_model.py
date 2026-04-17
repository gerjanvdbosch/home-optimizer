"""Electricity price models for the MPC cost function.

Supports three price modes (§14.2):

1. ``flat``      — single flat rate [€/kWh] for all hours.
2. ``dual``      — high/low tariff with a feed-in compensation rate for
                   net-export hours (terugleververgoeding).
3. ``nordpool``  — real day-ahead hourly prices fetched from the Nordpool
                   Data Portal via :class:`~home_optimizer.sensors.nordpool.NordpoolClient`.

Architecture
------------
All modes are implemented as subclasses of the abstract :class:`BasePriceModel`.
The factory function :func:`build_price_model` constructs the right subclass
from a :class:`PriceConfig` Pydantic model, which is validated fail-fast before
reaching the MPC.

Units
-----
All prices returned by :meth:`BasePriceModel.prices` are in **€/kWh**.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field, model_validator

from .sensors.nordpool import NordpoolClient

log = logging.getLogger("home_optimizer.price_model")


# ---------------------------------------------------------------------------
# Enum: price mode
# ---------------------------------------------------------------------------


class PriceMode(str, Enum):
    """Electricity price mode selection.

    Values
    ------
    flat
        Single flat rate for all hours. Requires ``flat_rate_eur_per_kwh``.
    dual
        High / low tariff (piek/dal) with optional terugleververgoeding.
        Requires ``high_rate_eur_per_kwh``, ``low_rate_eur_per_kwh``,
        ``feed_in_rate_eur_per_kwh``.  Low tariff applies to the hours listed
        in ``low_tariff_hours`` (configurable).
    nordpool
        Real day-ahead hourly prices from the Nordpool Data Portal API.
        Requires ``nordpool_country_code`` and ``nordpool_vat_factor``.
        Raises on network failure — no silent fallback.
    """

    flat = "flat"
    dual = "dual"
    nordpool = "nordpool"


# ---------------------------------------------------------------------------
# Pydantic config model
# ---------------------------------------------------------------------------


class PriceConfig(BaseModel):
    """Validated electricity price configuration.

    All monetary parameters are in **€/kWh**.

    Attributes
    ----------
    mode:
        Price mode (flat / dual / nordpool).
    flat_rate_eur_per_kwh:
        Flat import rate [€/kWh].  Required for ``flat`` mode.
    high_rate_eur_per_kwh:
        Peak (high) import tariff [€/kWh].  Required for ``dual`` mode.
    low_rate_eur_per_kwh:
        Off-peak (low) import tariff [€/kWh].  Required for ``dual`` mode.
    feed_in_rate_eur_per_kwh:
        Feed-in / terugleververgoeding rate for net export [€/kWh].
        Use 0.0 to disable feed-in compensation.  Required for ``dual`` mode.
    low_tariff_hours:
        List of hour-of-day integers (0–23) that qualify as off-peak.
        Defaults to 23:00–06:00 (hours 23, 0, 1, 2, 3, 4, 5, 6).
    nordpool_country_code:
        Nordpool bidding-zone / delivery-area code, e.g. ``"NL"`` or ``"DE-LU"``.
        Required for ``nordpool`` mode.
    nordpool_vat_factor:
        VAT + surcharge multiplier applied to the raw day-ahead price.
        E.g. 1.21 for 21 % Dutch BTW.  Must be ≥ 1.0.
    """

    mode: PriceMode = Field(PriceMode.flat, description="Price mode: flat | dual | nordpool")

    # ── flat ─────────────────────────────────────────────────────────────────
    flat_rate_eur_per_kwh: float = Field(
        0.25, ge=0.0, le=5.0, description="Flat electricity import rate [€/kWh]"
    )

    # ── dual ─────────────────────────────────────────────────────────────────
    high_rate_eur_per_kwh: float = Field(
        0.36, ge=0.0, le=5.0, description="Peak (high) import tariff [€/kWh]"
    )
    low_rate_eur_per_kwh: float = Field(
        0.21, ge=0.0, le=5.0, description="Off-peak (low) import tariff [€/kWh]"
    )
    feed_in_rate_eur_per_kwh: float = Field(
        0.09,
        ge=0.0,
        le=5.0,
        description="Feed-in (terugleververgoeding) rate for net export [€/kWh]",
    )
    low_tariff_hours: list[int] = Field(
        default_factory=lambda: [23, 0, 1, 2, 3, 4, 5, 6],
        description=(
            "Hours (0–23) that qualify as off-peak in dual-tariff mode.  "
            "Default: 23:00–06:00 inclusive."
        ),
    )

    # ── nordpool ─────────────────────────────────────────────────────────────
    nordpool_country_code: str = Field(
        "NL",
        min_length=2,
        description="Nordpool delivery-area code, e.g. 'NL' or 'DE-LU'",
    )
    nordpool_vat_factor: float = Field(
        1.21,
        ge=1.0,
        le=2.0,
        description="VAT + surcharge multiplier applied to raw Nordpool price (e.g. 1.21 = 21% BTW)",
    )

    @model_validator(mode="after")
    def _validate_mode_requirements(self) -> "PriceConfig":
        """Fail-fast: verify that required fields are present for the chosen mode."""
        if self.mode == PriceMode.dual:
            if self.high_rate_eur_per_kwh <= 0.0:
                raise ValueError("high_rate_eur_per_kwh must be > 0 for dual-tariff mode.")
            if self.low_rate_eur_per_kwh <= 0.0:
                raise ValueError("low_rate_eur_per_kwh must be > 0 for dual-tariff mode.")
            invalid_hours = [h for h in self.low_tariff_hours if not 0 <= h <= 23]
            if invalid_hours:
                raise ValueError(f"low_tariff_hours contains invalid values: {invalid_hours}")
        return self


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BasePriceModel(ABC):
    """Abstract base class for all electricity price models.

    Subclasses implement :meth:`prices` to return an array of import prices
    over a requested horizon.  Feed-in compensation is exposed via
    :meth:`feed_in_prices` (default: zero for all hours).
    """

    @abstractmethod
    def prices(self, start_hour: int, n_steps: int) -> np.ndarray:
        """Return import electricity prices [€/kWh] for ``n_steps`` hours.

        Args:
            start_hour: UTC hour of the first step (0–23).
            n_steps:    Number of steps (one per hour).

        Returns:
            1-D ``np.ndarray`` of length ``n_steps``, all values ≥ 0.
        """

    def feed_in_prices(self, start_hour: int, n_steps: int) -> np.ndarray:
        """Return feed-in (export) prices [€/kWh] for ``n_steps`` hours.

        The default implementation returns zeros (no compensation).
        Override in subclasses that support feed-in (e.g. :class:`DualTariffPriceModel`).

        Args:
            start_hour: UTC hour of the first step (0–23).
            n_steps:    Number of steps.

        Returns:
            1-D ``np.ndarray`` of length ``n_steps``, all values ≥ 0.
        """
        return np.zeros(n_steps)


# ---------------------------------------------------------------------------
# Flat-rate model
# ---------------------------------------------------------------------------


class FlatPriceModel(BasePriceModel):
    """Flat (constant) electricity import price.

    Args:
        rate_eur_per_kwh: Import tariff [€/kWh].  Must be ≥ 0.
    """

    def __init__(self, rate_eur_per_kwh: float) -> None:
        if rate_eur_per_kwh < 0.0:
            raise ValueError(f"rate_eur_per_kwh must be ≥ 0, got {rate_eur_per_kwh}.")
        self._rate = rate_eur_per_kwh

    def prices(self, start_hour: int, n_steps: int) -> np.ndarray:
        """Return a constant price array of length ``n_steps``.

        Args:
            start_hour: Ignored for flat pricing.
            n_steps:    Number of steps.

        Returns:
            Array filled with ``self._rate`` [€/kWh].
        """
        return np.full(n_steps, self._rate)


# ---------------------------------------------------------------------------
# Dual-tariff model (piek/dal + terugleververgoeding)
# ---------------------------------------------------------------------------


class DualTariffPriceModel(BasePriceModel):
    """High/low (piek/dal) tariff with optional feed-in compensation.

    Off-peak hours are specified via ``low_tariff_hours`` (list of hour
    integers 0–23).  All other hours use the high (peak) tariff.

    Args:
        config: Validated :class:`PriceConfig` with dual-tariff fields set.
    """

    def __init__(self, config: PriceConfig) -> None:
        self._high = config.high_rate_eur_per_kwh
        self._low = config.low_rate_eur_per_kwh
        self._feed_in = config.feed_in_rate_eur_per_kwh
        self._low_hours: frozenset[int] = frozenset(config.low_tariff_hours)

    def prices(self, start_hour: int, n_steps: int) -> np.ndarray:
        """Return import tariff per step: high rate outside off-peak, low rate in off-peak.

        Args:
            start_hour: UTC hour of the first step (0–23).
            n_steps:    Number of steps.

        Returns:
            1-D array [€/kWh], length ``n_steps``.
        """
        hours = [(start_hour + k) % 24 for k in range(n_steps)]
        return np.array(
            [self._low if h in self._low_hours else self._high for h in hours],
            dtype=float,
        )

    def feed_in_prices(self, start_hour: int, n_steps: int) -> np.ndarray:
        """Return the constant feed-in (terugleververgoeding) rate for all steps.

        Args:
            start_hour: Ignored — feed-in rate is time-invariant in this model.
            n_steps:    Number of steps.

        Returns:
            1-D array filled with ``feed_in_rate_eur_per_kwh`` [€/kWh].
        """
        return np.full(n_steps, self._feed_in)


# ---------------------------------------------------------------------------
# Nordpool day-ahead model
# ---------------------------------------------------------------------------

#: Number of hours in a 48-hour window used for horizon spanning midnight.
_WINDOW_HOURS: int = 48


class NordpoolPriceModel(BasePriceModel):
    """Day-ahead electricity prices from the Nordpool Data Portal API.

    Delegates HTTP fetching to :class:`~home_optimizer.sensors.nordpool.NordpoolClient`.
    Caches per calendar date so repeated calls within the same day cost no
    additional HTTP requests.

    Horizon spanning midnight is handled by fetching both today's and
    tomorrow's prices into a 48-hour window.

    Raises on network or parse failure — **no silent fallback**.  The
    operator must ensure the Nordpool API is reachable when this mode is
    configured.

    Args:
        config: Validated :class:`PriceConfig` with ``nordpool_*`` fields set.
    """

    def __init__(self, config: PriceConfig) -> None:
        self._client = NordpoolClient(
            delivery_area=config.nordpool_country_code,
            vat_factor=config.nordpool_vat_factor,
        )
        # Cache: maps date-string → 24-element price array [€/kWh].
        self._cache: dict[str, np.ndarray] = {}

    def _prices_for_date(self, target_date: "datetime.date") -> np.ndarray:  # type: ignore[name-defined]
        """Return (possibly cached) 24-hour price array for ``target_date``.

        Args:
            target_date: Calendar date (UTC).

        Returns:
            24-element array [€/kWh].

        Raises:
            httpx.HTTPStatusError: Non-200 API response.
            httpx.RequestError: Network failure.
            ValueError: Unparseable API payload.
        """
        key = target_date.strftime("%Y-%m-%d")
        if key not in self._cache:
            result = self._client.fetch_day_ahead(target_date)
            self._cache[key] = result.prices_eur_per_kwh
        return self._cache[key]

    def prices(self, start_hour: int, n_steps: int) -> np.ndarray:
        """Return day-ahead prices for ``n_steps`` hours starting at ``start_hour``.

        Builds a 48-hour window (today + tomorrow UTC) and slices the
        requested horizon.  Raises if either day's prices cannot be fetched.

        Args:
            start_hour: UTC hour of the first step (0–23).
            n_steps:    Number of steps.

        Returns:
            1-D array [€/kWh], length ``n_steps``.

        Raises:
            httpx.HTTPStatusError: Non-200 API response for today or tomorrow.
            httpx.RequestError: Network failure.
            ValueError: Unparseable API payload or missing area in response.
        """
        now_utc = datetime.now(tz=timezone.utc)
        today = now_utc.date()
        tomorrow = (now_utc + timedelta(days=1)).date()

        today_prices = self._prices_for_date(today)
        tomorrow_prices = self._prices_for_date(tomorrow)
        full_48h = np.concatenate([today_prices, tomorrow_prices])

        # Slice the requested horizon from start_hour, wrapping within 48 h.
        indices = np.array([(start_hour + k) % _WINDOW_HOURS for k in range(n_steps)])
        return full_48h[indices]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_price_model(config: PriceConfig) -> BasePriceModel:
    """Factory: construct the appropriate :class:`BasePriceModel` subclass.

    Args:
        config: Validated :class:`PriceConfig`.

    Returns:
        :class:`FlatPriceModel`, :class:`DualTariffPriceModel`, or
        :class:`NordpoolPriceModel` depending on ``config.mode``.

    Raises:
        ValueError: If an unknown mode is encountered (guards against future
            enum extensions that are not yet handled here).
    """
    if config.mode == PriceMode.flat:
        return FlatPriceModel(config.flat_rate_eur_per_kwh)
    if config.mode == PriceMode.dual:
        return DualTariffPriceModel(config)
    if config.mode == PriceMode.nordpool:
        return NordpoolPriceModel(config)
    raise ValueError(f"Unknown PriceMode: {config.mode!r}")  # pragma: no cover

