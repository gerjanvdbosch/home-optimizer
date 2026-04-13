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

