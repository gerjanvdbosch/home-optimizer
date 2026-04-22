Local runner — ontwikkel- en test-handleiding
============================================

Kort: deze pagina beschrijft hoe je de lokale runner (`local_runner`) gebruikt om
de volledige stack (ForecastPersister, Telemetry collector en FastAPI) lokaal uit
te voeren voor ontwikkeling en tests.

Wat ik ga/doe
-------------
- Beschrijf CLI-argumenten en environment-variabelen.
- Leg de interactie uit tussen repository (database), persisted ML-artifacts
  (`--models-dir`) en `sensors.json` als bron voor live initial conditions.
- Voorzie duidelijke copy-paste voorbeelden en test-commando's.

Overzicht
--------
`local_runner` combineert drie hoofdfuncties:

- ForecastPersister: haalt Open-Meteo weer- en GTI-data op en schrijft deze naar
  de telemetry-database (gebruikelijk één keer bij opstart en vervolgens elk uur).
- BufferedTelemetryCollector (optioneel): leest sensoren (via `sensors.json`) en
  schrijft geaggregeerde buckets naar de database — handig voor offline simulatie.
- FastAPI / Uvicorn: draait de HTTP API (`/api/...`) zodat de web UI en endpoints
  lokaal beschikbaar zijn.

Het doel van de runner is een realistische lokale omgeving te bieden zonder Home
Assistant.

Belangrijke CLI-argumenten
-------------------------

- `--database PATH`
  - Pad naar een SQLite-bestand. Dit heeft prioriteit boven `DATABASE_URL` env.
  - Voorbeeld: `--database ./dev_data/telemetry.db`.

- `--models-dir PATH` (nieuw)
  - Directory waarin gepersistenteerde ML forecast-artifacten (joblib) worden
    opgeslagen. Standaard `./models`. Als je deze flag weglaat, wordt het oude
    gedrag gebruikt: artifacts worden naast de SQLite database opgeslagen.
  - Voorbeelden:
    - `--models-dir ./models`
    - `--models-dir /config/models` (Home Assistant add-on map)

- `--sensors-json PATH`
  - Als aanwezig: de runner leest op elke MPC-run de initial conditions uit
    het JSON-bestand (LocalBackend). Dit is ideaal om reproductible MPC-runs te
    simuleren met vaste readings.

- `--horizon HOURS`
  - Forecast horizon die de ForecastPersister ophaalt en persist. Default 48.

- `--host`, `--port`, `--reload`
  - Uvicorn binding en auto-reload (voor template / API-ontwikkeling).

- `--mpc-interval`
  - Hoe vaak de MPC periodiek draait (seconden). 0 betekent geen periodieke MPC.

- `--calibration-interval`
  - Hoe vaak de automatische calibratie (offline) draait. 0 = uit.

Environment-variabelen
----------------------

- `DATABASE_URL`
  - Alternatief voor `--database`. Voorbeeld: `export DATABASE_URL=sqlite:///dev.db`.

- `MODELS_DIR`
  - Alternatief voor `--models-dir` wanneer je de runner via `from_env()` of
    een supervisor-like environment start. Dit wordt gebruikt wanneer geen
    `--models-dir` CLI-argument is opgegeven.

Start voorbeelden
-----------------

1) Installatie
----------

```bash
.venv/bin/pip install -e ".[dev]"
```

Controleer daarna de codekwaliteit en type-annotaties:

```bash
ruff check .
pyright
pytest -q
```

2) Snelle start (standaard modellen map `./models`):

```bash
python -m home_optimizer.local_runner
```

3) Start met expliciete models-dir en sensors.json (lokale telemetry collection):

```bash
python -m home_optimizer.local_runner \
  --database ./dev_data/telemetry.db \
  --models-dir ./models \
  --sensors-json ./sensors.json \
  --port 8099
```

4) Gebruik env-variabelen in plaats van CLI-args:

```bash
export DATABASE_URL=sqlite:///dev_data/telemetry.db
export MODELS_DIR=/config/models
python -m home_optimizer.local_runner
```

Uitleg van gedrag
-----------------

- Wanneer `--models-dir` is opgegeven (of `MODELS_DIR` env) worden persisted
  forecast artifacts (bijv. `baseload_model.joblib`, `shutter_model.joblib`) in
  die directory geplaatst. De runner zorgt voor het aanmaken van de directory.
- Wanneer geen models-dir is opgegeven worden artifacts naast het SQLite DB-bestand
  geplaatst met de conventie `dbbasename.<artifact>.joblib`.
- `--sensors-json` activeert een lokale BufferedTelemetryCollector die de
  readings in regelmatige buckets naar de database schrijft. Wanneer geen
  sensors-json is opgegeven dan wordt er geen lokale collector gestart en werkt
  de MPC met CLI-waarden als initial conditions.

Praktische tips
---------------

- Gebruik `--reload` alleen tijdens ontwikkeling van de API / templates — de
  reloader spawn subtiele subprocessen.
- `--models-dir` is handig wanneer je artifacts wilt delen tussen container
  runs of wanneer je ze onder `/config` in Home Assistant wilt hebben.
- Als je reproducible MPC-runs wilt testen, versieer dan je `sensors.json`
  en `models/` folder (artifacts zijn deterministisch gegeven dezelfde
  training-fingerprint).

Testing
-------

Run de tests:

```bash
pytest -q
```

Debug en logs
-------------

- De runner logt naar stdout. Voor extra details verhoog `logging.basicConfig`
  level in `local_runner.main()` of start met `DEBUG` in je environment.
- Wanneer de ForecastPersister of calibration job faalt, worden uitzonderingen
  gelogd maar starten ze de runner doorgaans niet af (initial fetchs worden
  gehanteerd met try/except zodat de API nog steeds beschikbaar komt).

Veelvoorkomende problemen
------------------------

- Geen schrijfrechten op `--models-dir` — je krijgt een duidelijke OSError bij
  poging tot persist. Controleer permissies en eigenaarschap.
- `sensors.json` bevat niet alle benodigde velden — LocalBackend zal een
  uitzondering werpen bij parsing; controleer `examples/sensors.json`.

Waar vind je meer?
------------------

- Runner broncode: `src/home_optimizer/local_runner.py`
- Telemetry & forecast persistence: `src/home_optimizer/telemetry/repository.py` en
  `src/home_optimizer/forecasting/*`.

  
