# Home Optimizer

Python-implementatie van het 2-state thermische model voor vloerverwarming (UFH), inclusief:
- fysisch consistente forward-Euler discretisatie;
- state-space representatie met matrices `A`, `B` en `E`;
- Kalman-filter voor schatting van ruimtetemperatuur en vloertemperatuur;
- convex MPC-regelaar met dynamische stroomprijs, comfortgrenzen en ramp-rate constraints.

## Installeren

```bash
python -m pip install -e '.[dev]'
```

## Demo draaien

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

