"""Nordpool day-ahead electricity price client.

Uses the public Nordpool Data Portal REST API (no authentication required):

    GET https://dataportal-api.nordpoolgroup.com/api/DayAheadPrices
        ?date=<YYYY-MM-DD>&market=DayAhead&deliveryArea=<AREA>&currency=<CCY>

The response contains one price entry per Market Time Unit (MTU) — typically
one hour for most European bidding zones.  Prices are returned in the
requested currency per MWh and are converted to €/kWh here.

Architecture
------------
* :class:`NordpoolClient` is a **stateless** HTTP client.  Each call to
  :meth:`fetch_day_ahead` makes one HTTP request and returns the parsed result.
* The caller (:class:`~home_optimizer.price_model.NordpoolPriceModel`) is
  responsible for caching and fallback logic.
* All numeric constants (unit conversion factor, timeout) are named module-level
  constants — no magic numbers inline.

Units
-----
Nordpool API → €/MWh.  Output of :meth:`fetch_day_ahead` → €/kWh (after
``vat_factor`` is applied).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import httpx
import numpy as np

log = logging.getLogger("home_optimizer.sensors.nordpool")

# ---------------------------------------------------------------------------
# Named constants — no magic numbers
# ---------------------------------------------------------------------------

#: Nordpool Data Portal base URL for day-ahead prices.
_NORDPOOL_BASE_URL: str = (
    "https://dataportal-api.nordpoolgroup.com/api/DayAheadPrices"
)

#: Market identifier for day-ahead products on the Nordpool portal.
_MARKET: str = "DayAhead"

#: Nordpool returns prices in €/MWh; this factor converts to €/kWh.
_MWH_TO_KWH: float = 1_000.0

#: Expected number of hourly MTUs per day.  Most EU zones use 1-hour MTUs.
_HOURS_PER_DAY: int = 24

#: HTTP request timeout [s].  Nordpool's CDN is fast; 10 s is generous.
_REQUEST_TIMEOUT_S: float = 10.0

#: HTTP header identifying this client to the Nordpool portal.
_USER_AGENT: str = "None"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DayAheadPrices:
    """Parsed day-ahead price result from the Nordpool API.

    Attributes
    ----------
    delivery_date:
        The calendar date for which prices apply (UTC).
    delivery_area:
        Nordpool bidding-zone code (e.g. ``"NL"``).
    currency:
        Currency of the prices (e.g. ``"EUR"``).
    prices_eur_per_kwh:
        Hourly prices [€/kWh] after VAT/surcharge multiplication,
        one entry per hour (0–23).  Length is always 24.
    """

    delivery_date: date
    delivery_area: str
    currency: str
    prices_eur_per_kwh: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        if len(self.prices_eur_per_kwh) != _HOURS_PER_DAY:
            raise ValueError(
                f"prices_eur_per_kwh must have exactly {_HOURS_PER_DAY} entries, "
                f"got {len(self.prices_eur_per_kwh)}."
            )
        if np.any(self.prices_eur_per_kwh < 0.0):
            # Negative prices are physically valid (surplus generation), but
            # values below −1 €/kWh are almost certainly a parse error.
            log.warning(
                "Day-ahead prices for %s contain negative values (min=%.4f €/kWh).  "
                "Negative prices are possible during high-surplus periods.",
                self.delivery_date,
                float(self.prices_eur_per_kwh.min()),
            )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


@dataclass
class NordpoolClient:
    """HTTP client for the Nordpool Data Portal day-ahead price API.

    Parameters
    ----------
    delivery_area:
        Nordpool bidding-zone code.  Common values:

        * ``"NL"``    — Netherlands (APX/EPEX area)
        * ``"DE-LU"`` — Germany-Luxembourg
        * ``"BE"``    — Belgium
        * ``"FR"``    — France
        * ``"DK1"``   — Denmark West
        * ``"DK2"``   — Denmark East
        * ``"NO1"``   — Norway South-East
        * ``"SE3"``   — Sweden Central
        * ``"FI"``    — Finland
    currency:
        Three-letter ISO 4217 currency code for the returned prices.
        Defaults to ``"EUR"``.
    vat_factor:
        Multiplier applied to the raw Nordpool price (which is the wholesale
        spot price **excluding** taxes).  Set to e.g. ``1.21`` for 21 % Dutch
        BTW + energy surcharges.  Must be ≥ 1.0.

    Examples
    --------
    >>> client = NordpoolClient(delivery_area="NL", vat_factor=1.21)
    >>> prices = client.fetch_day_ahead(date(2026, 4, 18))
    >>> prices.prices_eur_per_kwh[8]   # hour 08:00 price [€/kWh]
    """

    delivery_area: str
    currency: str = "EUR"
    vat_factor: float = 1.21

    def __post_init__(self) -> None:
        if len(self.delivery_area) < 2:
            raise ValueError(
                f"delivery_area must be at least 2 characters, got {self.delivery_area!r}."
            )
        if self.vat_factor < 1.0:
            raise ValueError(
                f"vat_factor must be ≥ 1.0 (raw Nordpool price is excl. taxes), "
                f"got {self.vat_factor}."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_day_ahead(self, delivery_date: date | None = None) -> DayAheadPrices:
        """Fetch hourly day-ahead prices for ``delivery_date``.

        Makes one synchronous HTTP GET request to the Nordpool Data Portal.
        Raises on network failure or non-200 HTTP status; the caller
        (:class:`~home_optimizer.price_model.NordpoolPriceModel`) handles
        the fallback to the Dutch proxy pattern.

        Parameters
        ----------
        delivery_date:
            Calendar date for which to fetch prices.  Defaults to today (UTC).

        Returns
        -------
        DayAheadPrices
            Parsed result with 24 hourly prices [€/kWh] including VAT.

        Raises
        ------
        httpx.HTTPStatusError
            When the API returns a non-200 HTTP status code.
        httpx.RequestError
            On network-level failures (timeout, DNS error, etc.).
        ValueError
            When the API response cannot be parsed or contains fewer than
            24 hourly entries.
        """
        if delivery_date is None:
            delivery_date = datetime.now(tz=timezone.utc).date()

        url = self._build_url(delivery_date)
        log.debug("Fetching Nordpool day-ahead prices: GET %s", url)

        with httpx.Client(timeout=_REQUEST_TIMEOUT_S, headers={"User-Agent": _USER_AGENT}) as client:
            response = client.get(url)
            response.raise_for_status()

        payload: dict = response.json()
        return self._parse_response(payload, delivery_date)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_url(self, delivery_date: date) -> str:
        """Construct the Nordpool API URL for ``delivery_date``.

        Parameters
        ----------
        delivery_date: Target delivery date.

        Returns
        -------
        str
            Fully-formed URL including all query parameters.
        """
        date_str = delivery_date.strftime("%Y-%m-%d")
        return (
            f"{_NORDPOOL_BASE_URL}"
            f"?date={date_str}"
            f"&market={_MARKET}"
            f"&deliveryArea={self.delivery_area}"
            f"&currency={self.currency}"
        )

    def _parse_response(self, payload: dict, delivery_date: date) -> DayAheadPrices:
        """Parse the Nordpool JSON payload into a :class:`DayAheadPrices`.

        The Nordpool Data Portal response has the structure::

            {
              "multiAreaEntries": [
                {
                  "deliveryStart": "2026-04-18T00:00:00Z",
                  "deliveryEnd":   "2026-04-18T01:00:00Z",
                  "entryPerArea": {"NL": 85.42}
                },
                ...
              ]
            }

        Each entry covers one MTU (typically 1 hour).  Prices are in the
        requested currency per MWh and are converted to €/kWh here.

        Parameters
        ----------
        payload:      Parsed JSON dict from the API response.
        delivery_date: Expected delivery date (used for validation).

        Returns
        -------
        DayAheadPrices

        Raises
        ------
        ValueError
            When ``multiAreaEntries`` is missing, the area key is not found,
            or fewer than 24 entries are present.
        """
        entries: list[dict] = payload.get("multiAreaEntries", [])
        if not entries:
            raise ValueError(
                f"Nordpool API returned no 'multiAreaEntries' for "
                f"{self.delivery_area} on {delivery_date}."
            )

        raw_prices: list[float] = []
        for entry in entries:
            area_prices: dict = entry.get("entryPerArea", {})
            if self.delivery_area not in area_prices:
                raise ValueError(
                    f"Delivery area '{self.delivery_area}' not found in Nordpool response entry.  "
                    f"Available areas: {list(area_prices.keys())}"
                )
            raw_prices.append(float(area_prices[self.delivery_area]))

        if len(raw_prices) < _HOURS_PER_DAY:
            raise ValueError(
                f"Expected at least {_HOURS_PER_DAY} hourly entries for "
                f"{self.delivery_area} on {delivery_date}, got {len(raw_prices)}.  "
                "This may occur when prices are not yet published (typically available "
                "from ~13:00 CET the day before delivery)."
            )

        # Take exactly 24 entries (guard against DST days with 23/25 hours).
        prices_mwh = np.array(raw_prices[:_HOURS_PER_DAY], dtype=float)
        # Convert €/MWh → €/kWh and apply VAT/surcharge factor.
        prices_kwh = (prices_mwh / _MWH_TO_KWH) * self.vat_factor

        log.info(
            "Nordpool day-ahead prices parsed for %s (area=%s, vat=%.2f): "
            "min=%.4f max=%.4f avg=%.4f €/kWh",
            delivery_date,
            self.delivery_area,
            self.vat_factor,
            float(prices_kwh.min()),
            float(prices_kwh.max()),
            float(prices_kwh.mean()),
        )

        return DayAheadPrices(
            delivery_date=delivery_date,
            delivery_area=self.delivery_area,
            currency=self.currency,
            prices_eur_per_kwh=prices_kwh,
        )

