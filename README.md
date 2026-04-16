# Home Optimizer

Python-implementatie van een gecombineerd thermisch model voor **vloerverwarming (UFH)** en **tapwater (DHW)**, inclusief:

- fysisch consistente forward-Euler discretisatie (grey-box model);
- state-space representatie met matrices `A`, `B` en `E`;
- Kalman-filter (KF) voor UFH en Extended Kalman-filter (EKF) voor DHW;
- convex MPC-regelaar (CVXPY/OSQP) met dynamische stroomprijs, comfortgrenzen, ramp-rate constraints en gedeeld warmtepompvermogen;
- Carnot COP-model met stooklijn voor tijdsvariabele efficiëntie;
- **FastAPI webinterface** met operationeel dashboard (Open-Meteo forecast) en MPC-simulator;
- **telemetrylaag** (SQLite + SQLAlchemy + APScheduler) voor sensor- en forecastopslag.

---

## Installeren

```bash
python -m pip install -e '.[dev]'
```

---

## Lokale ontwikkeling starten

De lokale runner combineert de **ForecastPersister** (haalt Open-Meteo op, slaat op in SQLite) en de **FastAPI/Uvicorn** server in één proces:

```bash
python -m home_optimizer.local_runner
```

Of met opties:

```bash
python -m home_optimizer.local_runner \
    --lat 52.37 --lon 4.90 \
    --database ./dev_data/local.db \
    --port 8000 \
    --horizon 48 \
    --pv-tilt 35
```

Of via het geïnstalleerde script:

```bash
home-optimizer-local --database ./dev.db --pv-tilt 35
```

**Alle opties:**

| Optie | Standaard | Beschrijving |
|---|---|---|
| `--lat` | `52.37` | Breedtegraad site [°N] |
| `--lon` | `4.90` | Lengtegraad site [°E] |
| `--database` | zie DATABASE_URL | Pad naar SQLite-bestand, bijv. `./dev.db` |
| `--host` | `127.0.0.1` | Uvicorn bind-adres |
| `--port` | `8000` | Uvicorn poort |
| `--reload` | uit | Uvicorn auto-reload (handig bij template-ontwikkeling) |
| `--horizon` | `48` | Forecast horizon [h] |
| `--window-tilt` | `90` | Glas-oppervlak helling [°] (90 = verticaal) |
| `--pv-tilt` | uit | PV-paneel helling [°]; weglaten = geen PV GTI |
| `--pv-azimuth` | `0` | PV-azimut [°] (0 = Zuid) |

### Database-locatie configureren

Prioriteit: `--database` CLI-arg > `DATABASE_URL` env-var > standaard (`sqlite:///database.sqlite3` in CWD).

```bash
# Via env-var (zonder --database flag)
DATABASE_URL=sqlite:///mijn_lokale.db python -m home_optimizer.local_runner
```

---

## Web-interface

Open na het starten van de runner:

| URL | Beschrijving |
|---|---|
| **http://localhost:8000** | Operationeel dashboard — laadt forecast uit de database, toont temperatuur, GTI en verwarmingsbehoefte |
| **http://localhost:8000/simulator** | MPC-simulator — alle fysische parameters instelbaar, optimaliseert UFH + DHW + PV |
| **http://localhost:8000/docs** | Automatische API-documentatie (OpenAPI/Swagger) |

### Dashboard (`/`)

- Laadt standaard de **meest recente forecast uit de database** (geen live API-call)
- Vinkje **"Live Open-Meteo ophalen"** voor een directe API-call met instelbare locatie/horizon/PV
- Grafieken: buitentemperatuurverwachting, GTI (ramen + PV), graaduren verwarming
- KPI-cards: T nu, min/max, zonnepiek, geldigheid forecast

### Simulator (`/simulator`)

- Alle huisparameters instelbaar (C_r, C_b, R_br, R_ro, α, η, A_glass)
- UFH + DHW + PV self-consumption
- Carnot COP-model met stooklijn
- Resultaten: kamertemperatuur-traject, warmtepompvermogen, COP-profiel, DHW-tanktemperaturen

De standaardwaarden zijn afgestemd op een **redelijk goed geïsoleerde Nederlandse tussenwoning (ca. 2023)** met vloerverwarming, warmtepomp en ~7,5 m² zuidgericht glas.

---

## API-endpoints

| Methode | Pad | Beschrijving |
|---|---|---|
| `GET` | `/` | Operationeel dashboard (HTML) |
| `GET` | `/simulator` | MPC-simulator (HTML) |
| `GET` | `/api/defaults` | Standaard `RunRequest` als JSON |
| `GET` | `/api/forecast` | Live Open-Meteo forecast (query params: lat, lon, horizon, pv_tilt, …) |
| `GET` | `/api/forecast/latest` | Meest recente forecast uit de database |
| `POST` | `/api/optimize` | Voer één MPC-stap uit, retourneert grafieken + samenvattingen |

---

## Home Assistant Addon

In een HA-omgeving start de addon automatisch:

```bash
home-optimizer-addon
```

De addon leest `/data/options.json` (geschreven door de HA Supervisor), bouwt de HA-sensorbackend, start de telemetrycollector + ForecastPersister en draait Uvicorn. De `DATABASE_URL` wordt automatisch ingesteld vanuit `options.json`.

---

## Telemetry

De telemetrylaag samplet live sensoren vaker dan ze naar disk schrijft:

- **sample** standaard elke `10 s`
- **flush / opslag** standaard elke `300 s` (= 5 minuten)
- opslag in SQLite via SQLAlchemy ORM in tabel `telemetry_aggregates`
- bij een wijziging van `hp_mode` wordt de bucket direct afgesloten (UFH en DHW nooit gemengd)

**Forecast persistentie** (`forecast_snapshots`):

- de `ForecastPersister` haalt elk uur een nieuw Open-Meteo forecast op
- elke stap wordt als aparte rij opgeslagen (normaalvorm: één rij per stap per fetch)
- duplicaten worden stilzwijgend genegeerd (veilig bij herstart)
- het dashboard leest altijd de meest recente batch via `GET /api/forecast/latest`

Lokale demo (leest uit `sensors.json`):

```bash
python examples/telemetry_collection.py
```

Programmatisch gebruik:

```python
from home_optimizer.sensors import LocalBackend
from home_optimizer.telemetry import (
    BufferedTelemetryCollector,
    TelemetryCollectorSettings,
    TelemetryRepository,
)

settings = TelemetryCollectorSettings(database_url="sqlite:///database.sqlite3")
repository = TelemetryRepository(database_url=settings.database_url)
backend = LocalBackend.from_json_file("sensors.json")
collector = BufferedTelemetryCollector(backend=backend, repository=repository, settings=settings)

collector.start()
# ... service draait ...
collector.shutdown(flush=True)
```

---

## CLI demo

```bash
python -m home_optimizer
# of
python examples/minimal_run.py
```

---

## Tests draaien

```bash
python -m pytest
ruff check src tests examples
black --check src tests examples
```
