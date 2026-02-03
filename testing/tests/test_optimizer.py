import pandas as pd
import numpy as np
import logging
import os
import sys

from datetime import datetime, timedelta
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "..", "src"))

if src_path not in sys.path:
    sys.path.insert(0, src_path)


# Importeer je eigen classes (zorg dat de bestandsnaam klopt, ik ga uit van optimizer.py)
from optimizer import Optimizer, Context

# Logging aanzetten zodat we zien wat de solver doet
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# =========================================================
# 1. MOCK CLASSES (Om de dependencies te simuleren)
# =========================================================
class MockConfig:
    def __init__(self):
        self.rc_model_path = "test_rc_model.joblib"
        self.ufh_model_path = "test_ufh_residual.joblib"
        self.dhw_model_path = "test_dhw_residual.joblib"

class MockDatabase:
    def get_history(self, cutoff_date):
        # We geven een lege DF terug voor dit test-scenario
        return pd.DataFrame()

# =========================================================
# 2. DATA GENERATIE
# =========================================================
def create_test_context():
    horizon = 48  # 12 uur aan data (15-min intervals)
    now = datetime.now().replace(minute=0, second=0, microsecond=0)

    # 1. Tijdstempels
    timestamps = [now + timedelta(minutes=15*i) for i in range(horizon)]

    # 2. Weersvoorspelling: koud in de ochtend, warmer in de middag
    # Sinus golf tussen 2 en 8 graden
    temps = 5 + 3 * np.sin(np.linspace(-np.pi, np.pi, horizon))

    # 3. Zonnepanelen: bult in het midden van de dag
    solar = np.zeros(horizon)
    solar[16:32] = [2.0 * np.sin(x) for x in np.linspace(0, np.pi, 16)]

    # 4. Prijzen: duur in de ochtend/avond, goedkoop overdag
    prices = [0.30] * horizon
    prices[10:20] = [0.10] * 10 # Goedkope middag

    forecast_df = pd.DataFrame({
        "timestamp": timestamps,
        "temp": temps,
        "power_corrected": solar,    # Solar forecast
        "load_corrected": [0.3] * horizon # Basisverbruik huis
    })

    # Maak de context object
    class TestContext:
        def __init__(self, df, pr):
            self.forecast_df = df
            self.prices = pr
            self.room_temp = 19.2    # Huidige binnentemperatuur
            self.dhw_top = 42.0     # Bovenkant boiler
            self.dhw_bottom = 35.0  # Onderkant boiler

    return TestContext(forecast_df, prices)

# =========================================================
# 3. HET TEST RUNNEN
# =========================================================
def run_test():
    logger.info("--- Start Optimizer Test ---")

    # Initialiseer
    config = MockConfig()
    db = MockDatabase()
    optimizer = Optimizer(config, db)

    # Forceer RC waarden (zodat we niet hoeven te trainen)
    optimizer.ident.R = 15.0
    optimizer.ident.C = 40.0
    optimizer.ident.is_fitted = True

    # Maak test data
    context = create_test_context()

    logger.info(f"Huidige staat: Kamer={context.room_temp}C, Boiler Top={context.dhw_top}C")

    # Run de optimizer
    result = optimizer.resolve(context)

    if result:
        logger.info("--- Resultaat ---")
        logger.info(f"Status: {result['status']}")
        logger.info(f"Gekozen Modus: {result['mode']}")
        logger.info(f"Target Vermogen: {result['target_power']} kW")
        logger.info(f"Boiler Energy: {result['dhw_energy_kwh']:.2f} kWh")

        # Laat het verloop van de eerste 4 uur zien
        logger.info("\nPlan voor de komende uren:")
        plan_df = pd.DataFrame({
            "Tijd": context.forecast_df["timestamp"][:16],
            "Prijs": context.prices[:16],
            "Temp_Out": context.forecast_df["temp"][:16].round(1),
            "Plan_Room": [round(x, 2) for x in result['planned_room'][:16]],
            "Plan_DHW": [round(x, 2) for x in result['planned_dhw'][:16]]
        })
        print(plan_df.to_string(index=False))

        # Check of de resultaten zinnig zijn
        assert "mode" in result
        assert result["target_power"] >= 0
        logger.info("\n✅ Test geslaagd: De solver heeft een geldig plan gegenereerd.")
    else:
        logger.error("❌ Test gefaald: Geen resultaat van de optimizer.")

if __name__ == "__main__":
    # Verwijder oude test files als die er zijn
    for f in ["test_rc_model.joblib", "test_ufh_residual.joblib", "test_dhw_residual.joblib"]:
        if Path(f).exists(): Path(f).unlink()

    try:
        run_test()
    finally:
        # Opruimen
        for f in ["test_rc_model.joblib", "test_ufh_residual.joblib", "test_dhw_residual.joblib"]:
            if Path(f).exists(): Path(f).unlink()