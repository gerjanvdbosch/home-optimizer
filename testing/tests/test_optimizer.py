import pandas as pd
import numpy as np
import logging
import os
import sys
from unittest.mock import MagicMock

from datetime import datetime, timedelta
from pathlib import Path

# Zorg dat de import paden kloppen
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from optimizer import Optimizer  # noqa: E402

# Logging instellen
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =========================================================
# 1. MOCK CLASSES
# =========================================================
class MockConfig:
    def __init__(self):
        self.hp_model_path = "test_perf_map.joblib"
        self.rc_model_path = "test_rc_model.joblib"
        self.ufh_model_path = "test_ufh_res.joblib"
        self.dhw_model_path = "test_dhw_res.joblib"


class MockDatabase:
    def get_history(self, days):
        return pd.DataFrame()


# =========================================================
# 2. DATA GENERATIE
# =========================================================
def create_test_context():
    horizon = 48  # 12 uur (48 * 15 min)
    now = datetime.now().replace(minute=0, second=0, microsecond=0)

    # Tijdstempels
    timestamps = [now + timedelta(minutes=15 * i) for i in range(horizon)]

    # Weersvoorspelling
    temps = 5 + 3 * np.sin(np.linspace(-np.pi, np.pi, horizon))
    wind = [4.0] * horizon
    solar_rad = np.maximum(0, 400 * np.sin(np.linspace(-np.pi, np.pi, horizon)))

    # Prijzen
    prices = [0.30] * horizon
    prices[10:20] = [0.10] * 10

    solar_forecast = np.zeros(horizon)
    solar_forecast[16:32] = [2.0 * np.sin(x) for x in np.linspace(0, np.pi, 16)]

    forecast_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temp": temps,
            "solar": solar_rad,
            "wind": wind,
            "price": prices,
            "solar_forecast": solar_forecast,
        }
    )

    class Context:
        def __init__(self, df):
            self.forecast_df = df
            self.room_temp = 19.8
            self.dhw_top = 12.0
            self.dhw_bottom = 10.0

    return Context(forecast_df)


# =========================================================
# 3. HET TEST RUNNEN
# =========================================================
def run_test():
    logger.info("--- Start Optimizer Test ---")

    # Initialiseer
    config = MockConfig()
    db = MockDatabase()
    optimizer = Optimizer(config, db)

    # 1. Forceer modelwaarden
    optimizer.ident.R = 15.0
    optimizer.ident.C = 30.0
    optimizer.ident.K_emit = 0.15
    optimizer.ident.K_tank = 0.25
    optimizer.ident.K_loss_dhw = 0.20

    # 2. Mock de Performance Map
    optimizer.perf_map.is_fitted = True
    optimizer.perf_map.cop_model = MagicMock()
    optimizer.perf_map.cop_model.predict.return_value = np.array([3.5])
    optimizer.perf_map.power_model = MagicMock()
    optimizer.perf_map.power_model.predict.return_value = np.array([1.4])

    optimizer.perf_map.max_freq_model = MagicMock()
    # Simuleer: dynamische limiet rond de 55-60Hz voor dit weer
    optimizer.perf_map.max_freq_model.predict.side_effect = lambda x: [
        70.0 - (x[0][0] + 10) * 1.0
    ]

    # Maak test data
    context = create_test_context()

    logger.info(
        f"Huidige staat: Kamer={context.room_temp}C, Boiler Gemiddeld={(context.dhw_top+context.dhw_bottom)/2}C"
    )

    # Run de optimizer
    result = optimizer.resolve(context)

    if result:
        logger.info("--- Resultaat ---")
        logger.info(f"Gekozen Modus: {result['mode']}")
        logger.info(f"Frequentie: {result['freq']} Hz")

        if optimizer.mpc.f_ufh.value is not None:
            logger.info("\nEerste uren van het plan:")
            f_u = optimizer.mpc.f_ufh.value
            f_d = optimizer.mpc.f_dhw.value
            t_r = optimizer.mpc.t_room.value
            t_d = optimizer.mpc.t_dhw.value  # De voorspelde DHW temp

            # Print tabelletje van de eerste 8 kwartieren
            plan_data = []
            for i in range(48):
                mode = "UFH" if f_u[i] > 5 else "DHW" if f_d[i] > 5 else "OFF"
                plan_data.append(
                    {
                        "Tijd": context.forecast_df["timestamp"]
                        .iloc[i]
                        .strftime("%H:%M"),
                        "Mode": mode,
                        "Freq": round(max(f_u[i], f_d[i]), 1),
                        "T_room": round(t_r[i], 2),
                        "T_dhw": round(t_d[i], 2),  # NIEUW: DHW temperatuur in de tabel
                    }
                )

            print(pd.DataFrame(plan_data).to_string(index=False))

        assert "mode" in result
        assert "freq" in result
        logger.info(
            "\n✅ Test geslaagd: De solver heeft een geldig resultaat gegenereerd."
        )
    else:
        logger.error("❌ Test gefaald: Geen resultaat.")


if __name__ == "__main__":
    paths = [
        "test_perf_map.joblib",
        "test_rc_model.joblib",
        "test_ufh_res.joblib",
        "test_dhw_res.joblib",
    ]
    for p in paths:
        if Path(p).exists():
            Path(p).unlink()

    try:
        run_test()
    except Exception as e:
        logger.exception(f"Fout tijdens test: {e}")
    finally:
        for p in paths:
            if Path(p).exists():
                Path(p).unlink()
