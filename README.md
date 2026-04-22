# Home Optimizer

Home Optimizer is een compacte, fysisch-gefundeerde regelingstack voor woningen met
een warmtepomp. De applicatie combineert een grey-box thermisch model (UFH + DHW),
online state-estimatie (Kalman / Extended Kalman), en een Model Predictive Controller
(MPC) om thermisch vermogen economisch en comfortabel te dispatchen onder
dynamische elektriciteitsprijzen. Telemetrie, forecast-modellen en calibratieflows
zijn ingebouwd zodat de oplossing end-to-end reproduceerbaar lokaal of als
Home Assistant addon draait.

Belangrijkste features:
 - Fysisch consistente forward-Euler discretisatie voor UFH en DHW
 - State-space representaties met generieke Kalman- en EKF-implementaties
 - Convexe MPC-formulering (CVXPY/OSQP) met comfort-, ramp-rate- en gedeelde
   warmtepomp-constraints
 - Carnot-gebaseerd COP-model en pre-calculatie voor elektrische kosten in de MPC
 - Persistente telemetry (SQLite + SQLAlchemy) en periodieke forecast-persistentie
 - ML-forecastlaag (scikit-learn) met atomic artifact I/O
 - Offline calibration tools voor het schatten van thermische parameters
 - FastAPI dashboard en simulator voor inspectie en handmatige runs

Zie ook: `docs/calibration.md` en `docs/local_runner.md`.

---

## Installeren

[![Open your Home Assistant instance and show the add add-on repository dialog with a specific repository URL pre-filled.](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2Fgerjanvdbosch%2Fhome-optimizer)

Voor lokale ontwikkeling met linting, type-checking en tests:

```bash
.venv/bin/pip install -e ".[dev]"
ruff check .
pyright
pytest -q
```
