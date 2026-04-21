Kalibratie-handleiding voor de offline calibration-tool
=====================================================

Deze pagina beschrijft het gebruik van de offline calibration CLI van Home Optimizer.
De calibration-tool analyseert eerder gepersistente telemetrie (telemetry SQLite database)
en past fysische/gelijkwaardige parameters (UFH / DHW / COP) aan zodat het
thermische model beter overeenkomt met de historische data.

Doelgroep
---------
- Ingenieurs en ontwikkelaars die de thermische modelparameters willen afleiden uit
  gemeten geschiedenis.
- Gebruikers die automatische calibratie willen valideren voor productie of tests.

Belangrijke eigenschappen
------------------------
- Alle groottes en eenheden volgen het project-conventie: vermogen in kW, energie in kWh,
  temperatuur in °C, volumes in L of m³, tijd in uren (h).
- De CLI is fail-fast: ontbrekende referentieparameters voor actieve stages resulteren in
  directe fouten met duidelijke meldingen.
- De calibration-tool vertrouwt op gepersistente telemetry in een SQLAlchemy-compatibele
  database (meestal SQLite). De repository-API wordt direct gebruikt:
  ``TelemetryRepository(database_url=...)``.

Installeer / start
------------------
In een ontwikkelomgeving met de repository gecloned kun je de calibration-tool als volgt
direct uitvoeren:

```bash
# vanuit de projectroot
python -m home_optimizer.calibration
```

De CLI heeft ook een console-entrypoint `home-optimizer-calibration` wanneer het pakket
is geïnstalleerd via pip / build tooling.

Hoofdconcepten & stages
------------------------
De tool ondersteunt meerdere kalibratiestadia. Kies via de `--stage` parameter:

- off
  - Bepaal passieve (off-mode) dynamica van de woning (tau_house), nuttig om
    het huis-tijdconstante te schatten wanneer de UFH uit staat.

- active-ufh
  - Active UFH RC-fit: schat `C_r`, `C_b`, `R_br`, `R_ro` (en optioneel een
    initiele vloer-temperatuur offset). Vereist referentieparameters (--dt-hours,
    --c-r, --c-b, --r-br, --r-ro, --alpha, --eta, --a-glass) als uitgangspunt.

- dhw-standby
  - Schat DHW standby-verlies (R_loss, tau_standby) vanuit passieve tank-episodes.
    Vereist DHW tijdstap en capaciteiten (--dhw-dt-hours, --dhw-c-top, --dhw-c-bot).

- active-dhw
  - Actieve DHW (stratificatie) fit: schat `R_strat` met vooraf gegeven referentie
   waarden (dt, C_top, C_bot, R_loss, R_strat). Vereist meerdere actieve segments
    met voldoende temperatuurspreiding en geladen energie.

- cop
  - Offline COP-fit: combineert UFH- en DHW-segmentselectie, fitted Carnot-achtige
    parameters (`eta_carnot`, `T_supply_min`, `T_ref_outdoor`, `heating_curve_slope`)
    en valideert model-elektriciteitsbalansen. Veel handige selectie/cleaning flags
    zijn beschikbaar om ruwe telemetry te filteren.

Veelvoorkomende CLI-parameters
-----------------------------
- --database-url
  - SQLAlchemy URL naar telemetry DB (default `sqlite:///database.sqlite3`).

- UFH / algemene referenties (voor active-ufh):
  - --dt-hours, --c-r, --c-b, --r-br, --r-ro, --alpha, --eta, --a-glass

- DHW referenties (voor dhw-* stages):
  - --dhw-dt-hours, --dhw-c-top, --dhw-c-bot, --dhw-r-loss, --dhw-r-strat

- COP-tuning / robust-loss instellingen: zie `--cop-*` args in de CLI voor
  `min_segment_samples`, `reaggregate_min_electric_energy_kwh`, `cop-min`, `cop-max`,
  en robust loss-schaalwaarden.

- --json
  - Print machineleesbare JSON van de dataset statussen en fits in plaats van
    een korte menselijke samenvatting.

Voorbeelden
-----------

# 1) Passive UFH fit (off-mode) op een lokale dev DB
python -m home_optimizer.calibration --stage off --database-url sqlite:///dev_data/telemetry.db

# 2) Active UFH fit: vereist referentieparameters (fail-fast als ze missen)
python -m home_optimizer.calibration \
  --stage active-ufh \
  --database-url sqlite:///dev_data/telemetry.db \
  --dt-hours 1.0 --c-r 6.0 --c-b 10.0 --r-br 1.0 --r-ro 10.0 --alpha 0.25 --eta 0.55 --a-glass 7.5

# 3) DHW standby-loss fit (vereist DHW capacities en dt)
python -m home_optimizer.calibration --stage dhw-standby --database-url sqlite:///dev_data/telemetry.db \
  --dhw-dt-hours 1.0 --dhw-c-top 0.05 --dhw-c-bot 0.05

# 4) COP offline fit met diagnostics (inspecteer welke buckets/segmenten gekeurd worden)
python -m home_optimizer.calibration --stage cop --cop-diagnostics --database-url sqlite:///dev_data/telemetry.db

Interpretatie van output
------------------------
- Standaard wordt een compacte samenvatting naar stdout geschreven waarin sleutelwaarden
  (fit parameters, RMSE, dataset omvang, optimizer status) zichtbaar zijn.
- Met `--json` ontvang je een volledige JSON structuur met `dataset` en `fit` keys waarmee
  je eventuele CI-pijplijnen of automatische validatie kunt voeden.

Fail-fast & validatie
---------------------
- Voor actieve stages controleert de CLI expliciet dat de noodzakelijke referentieparameters
  opgegeven zijn. Wanneer parameters missen wordt het proces afgebroken met een
  duidelijke error (zie `_build_reference_parameters` en soortgelijke helper functies).
- Parameter-waarden (minima en maxima) en selectiecriteria zijn aanwezig als
  standaarddefinities in `src/home_optimizer/calibration/models.py` en verwante settings-factory.

Diagnose en debugging
---------------------
- Gebruik `--cop-diagnostics` om te zien hoeveel history-rows en segments er worden
  afgekeurd wegens gebrek aan energie, onvoldoende spreiding of andere kwalitatieve
  criteria. Dit helpt om te bepalen of meer historische data of andere selectie
  instellingen noodzakelijk zijn.
- Wanneer optimizer statuses aangeven dat een answer niet-optimaal is, kun je:
  - de selectiecriteria versoepelen (--min-samples, --min-segment-samples);
  - meer telemetry verzamelen (langere periode of hogere sampling);
  - de CLI in JSON-mode draaien en de output inspecteren via een script.

Tests
-----
- Er is een test-suite aanwezig die kalibratie-gerelateerde functionaliteit valideert
  (zie `tests/test_calibration.py` en `tests/test_calibration_cli.py`). Voer de tests
  lokaal uit met:

```bash
pytest tests/test_calibration.py -q
pytest tests/test_calibration_cli.py -q
```

Best practices
--------------
- Begin met `--stage off` en `--stage dhw-standby` om passieve parameters te schatten
  (minder afhankelijke referenties). Gebruik die waardes als referenties voor de
  actieve fits.
- Gebruik `--json` en sla resultaten op in versiebeheer (of als CI-artifact) zodat
  calibratie-snapshots traceerbaar worden (datum, dataset fingerprint, instellingen).
- Kalibreer `R_strat` empirisch: de parameter is grey-box en moet consistent getest
  worden op jouw specifieke installatie.

Waar vind je meer informatie?
-----------------------------
- Broncode calibration entrypoint: `src/home_optimizer/calibration/__main__.py`
- Calibratie-algoritmes en instellingen: `src/home_optimizer/calibration/service.py` en
  `src/home_optimizer/calibration/settings_factory.py` en `models.py`.
- Telemetry schema en repository helpers: `src/home_optimizer/telemetry/repository.py`.

