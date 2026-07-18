# Home Optimizer

A Home Assistant add-on that learns your home's thermal behavior and uses 
model predictive control to schedule your heat pump around solar 
production and electricity prices.


## Requirements

Home Optimizer requires the **InfluxDB add-on (v1)** for Home Assistant to 
store and retrieve historical sensor data used for training and optimization.

## Update API

The `/api/update` endpoint accepts a JSON payload describing the Home Assistant 
sensors. The payload should have the following format:

```json
{
  "solar_forecast": {
    "p10": ["sensor.solcast_pv_forecast", "pv_estimate10"],
    "p50": ["sensor.solcast_pv_forecast", "pv_estimate"],
    "p90": ["sensor.solcast_pv_forecast", "pv_estimate90"]
  }
}
```

Where:

* `p10` – Entity ID and attribute containing the 10th percentile forecast.
* `p50` – Entity ID and attribute containing the median (expected) forecast.
* `p90` – Entity ID and attribute containing the 90th percentile forecast.
