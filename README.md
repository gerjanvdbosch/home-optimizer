# Home Optimizer

Python-implementatie van het 2-state thermische model voor vloerverwarming (UFH), inclusief:
- fysisch consistente forward-Euler discretisatie;
- state-space representatie met matrices `A`, `B` en `E`;
- Kalman-filter voor schatting van ruimte- en vloertemperatuur;
- convex MPC-regelaar met dynamische stroomprijs, comfortgrenzen en ramp-rate constraints;
- **FastAPI webinterface met interactieve Plotly-grafieken**.

## Installeren

```bash
python -m pip install -e '.[dev]'
```

## Web-interface (FastAPI + Plotly)

```bash
uvicorn home_optimizer.api:app --reload
```

Open daarna **http://localhost:8000** in de browser. Je ziet:
- Live Plotly-grafiek van de voorspelde kamertemperatuur T_r
- Comfort-band (T_min / T_max) en setpoint T_ref
- UFH-vermogen P_UFH en dynamische stroomprijs
- Instelbaar: alle huisparameters, horizon, comfort-grenzen, weersvoorspelling

De standaardwaarden in de webinterface zijn afgestemd op een **redelijk goed geïsoleerde Nederlandse tussenwoning uit circa 2023** met:
- vloerverwarming en warmtepomp;
- ongeveer **7.5 m²** zuidgericht glas (bijv. openslaande deuren + zijlicht);
- typische interne warmtelast van **0.30 kW**.

> Let op: **zonnepanelen (bijv. 2 kWp PV)** worden op dit moment nog **niet** apart gemodelleerd in de energiekosten of thermische toestanden. Alleen **zonnewarmte door glas** (`A_glass`, `eta`, `alpha`) gaat het thermische model in.

API-docs: **http://localhost:8000/docs**

## CLI demo draaien

```bash
python -m home_optimizer
```

of:

```bash
python examples/minimal_run.py
```

## Tests draaien

```bash
python -m pytest
ruff check src tests examples
black --check src tests examples
```

