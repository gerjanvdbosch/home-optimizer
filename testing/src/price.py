import logging
import requests
import pandas as pd

from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PriceClient:
    def __init__(self, area="NL", currency="EUR"):
        self.base_url = "https://dataportal-api.nordpoolgroup.com/api/DayAheadPrices"
        self.area = area
        self.currency = currency

    def fetch_prices(self, date):
        """Haalt prijzen op voor een specifieke datum."""
        params = {
            "deliveryArea": self.area,
            "market": "DayAhead",
            "currency": self.currency,
            "date": date.strftime("%Y-%m-%d"),
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            entries = []
            # Nord Pool API structuur: multiAreaEntries bevat de prijzen per gebied
            for entry in data.get("multiAreaEntries", []):
                start_time = datetime.fromisoformat(
                    entry["deliveryStart"].replace("Z", "+00:00")
                )
                # Pak de prijs voor het gevraagde gebied (NL)
                price = entry["entryPerArea"].get(self.area)
                if price is not None:
                    # Nord Pool levert prijzen in MWh, we converteren naar kWh
                    price_kwh = price / 1000.0
                    entries.append({"timestamp": start_time, "price": price_kwh})

            return pd.DataFrame(entries)
        except Exception as e:
            logger.error(f"[PriceClient] Fout bij ophalen {params['date']}: {e}")
            return pd.DataFrame()

    def get_forecast(self, now):
        """Haalt vandaag en morgen op en combineert ze tot één kwartier-reeks."""
        today = now.date()
        tomorrow = today + timedelta(days=1)

        # Haal data op voor beide dagen
        df_today = self.fetch_prices(today)
        df_tomorrow = self.fetch_prices(tomorrow)

        # Combineer, verwijder dubbelen (bijv. overlap op middernacht) en sorteer
        df = pd.concat([df_today, df_tomorrow])

        if df.empty:
            logger.warning("[PriceClient] Geen prijsdata ontvangen van Nord Pool")
            return pd.DataFrame()

        # Schoonmaken en sorteren
        df = df.drop_duplicates("timestamp").sort_values("timestamp")

        return df.reset_index(drop=True)
