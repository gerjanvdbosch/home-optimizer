# Home Optimizer

> Experimental — use in test environments only.

A Home Assistant add-on that uses Model Predictive Control (MPC) to optimally schedule a heat pump for underfloor heating (UFH) and domestic hot water (DHW). It learns your building's thermal behavior, your PV system's output, and your household's energy usage — then plans the next 24 hours to minimize electricity costs while keeping your home comfortable.

---

## How it works

Every 15 minutes the system:

1. **Collects** sensor data from Home Assistant (room temperature, DHW tank, heat pump state, PV output, grid power)
2. **Forecasts** solar production (Solcast + OpenMeteo + ML correction) and household base load
3. **Identifies** your building's thermal model — R/C values, floor lag, tank loss — directly from historical data
4. **Solves** a Mixed Integer Linear Program over 96 × 15-minute steps using CVXPY + HiGHS
5. **Outputs** the optimal mode (`UFH`, `DHW`, or `OFF`), target electrical power, and supply temperature

The optimizer uses Sequential Linear Programming (SLP) iterations to handle the nonlinear COP curve of the heat pump while keeping the core problem a fast-solving MILP. All models are retrained nightly.

---

## Features

### Optimization
- 24-hour MPC horizon with 15-minute resolution
- Maximizes PV self-consumption, minimizes grid import
- Comfort constraints with soft slack — never sacrifices safety for cost
- Adaptive switching cost based on building physics — prevents short ineffective runs
- Terminal value reward for stored thermal energy at end of horizon
- Separate handling of UFH floor lag and DHW tank dynamics

### Learning
- **Building model** — learns R (thermal resistance), C (thermal mass), floor lag, and tank loss coefficient from your own measurement history
- **Heat pump performance** — learns electrical power and COP per operating mode directly from measured supply/return temperatures
- **Hydraulic model** — learns supply temperature lift and slope for UFH and DHW circuits
- **Solar forecasting** — blends Solcast with a trained ML model, corrected in real-time via a nowcaster
- **Load forecasting** — predicts household base load with quantile regression (P75), corrected in real-time
- **Shutter prediction** — learns your blind/shutter behavior to account for solar gain through windows
- **DHW demand** — learns hot water usage patterns per time of day

### Real-time correction
Both solar and load forecasts use an exponentially decaying nowcaster that tracks the current measurement against the model prediction and projects the correction forward in time.

---

## Installation
[![Open your Home Assistant instance and show the add add-on repository dialog with a specific repository URL pre-filled.](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2Fgerjanvdbosch%2Fhome-optimizer)
