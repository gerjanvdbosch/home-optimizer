# Home Optimizer

A Home Assistant add-on that learns your home's thermal behavior and uses model
predictive control to schedule your heat pump around solar production and electricity
prices.

## Requirements

Home Optimizer requires the **InfluxDB add-on (v1)** for Home Assistant to store and
retrieve historical sensor data used for training and optimization.

See the [InfluxDB documentation](https://www.home-assistant.io/integrations/influxdb/)
for installation and configuration.

## Update API

The `/api/update` endpoint is called from a Home Assistant automation using a
`rest_command`. It registers the Home Assistant sensor mappings for later use by the
training and optimization endpoints and updates the current optimizer state.

Example automation action:

```yaml
actions:
  - action: rest_command.home_optimizer_api
    data:
      endpoint: update
      payload: |
        {{ {
          "solar": {
            "production": "sensor.pv_output",
            "capacity": 2
          },
          "heat_pump": {
            "state": "sensor.ecodan_heatpump_ca09ec_status_bedrijf",
            "supply_temperature": "sensor.ecodan_heatpump_ca09ec_aanvoer_temp",
            "return_temperature": "sensor.ecodan_heatpump_ca09ec_retour_temp",
            "compressor_frequency": "sensor.ecodan_heatpump_compressor_frequentie",
            "boiler": {
              "setpoint": "sensor.ecodan_heatpump_ca09ec_sww_setpoint_waarde",
              "top_temperature": "sensor.ecodan_heatpump_ca09ec_sww_2e_temp_sensor",
              "bottom_temperature": "sensor.ecodan_heatpump_ca09ec_sww_huidige_temp"
            }
          },
          "forecast": {
            "solcast": "sensor.solcast_pv_forecast",
            "open_meteo": "sensor.open_meteo_forecast"
          },
          "presence": [
            "device_tracker.iphone_gerjan",
            "device_tracker.phone_partner"
          ]
        } | to_json }}
```

## Fit API

The `/api/fit` endpoint is called from a Home Assistant automation using a
`rest_command`.

Example automation action:

```yaml
actions:
  - action: rest_command.home_optimizer_api
    data:
      endpoint: fit
      payload: |
        {{ {
          "days": 90
        } | to_json }}
```

## Backtest API

The `/api/backtest` endpoint is called from a Home Assistant automation using a
`rest_command`.

Example automation action:

```yaml
actions:
  - action: rest_command.home_optimizer_api
    data:
      endpoint: backtest
      payload: |
        {{ {
          "days": 90,
          "forecaster": "solar"
        } | to_json }}
```

## Tune API

The `/api/tune` endpoint is called from a Home Assistant automation using a
`rest_command`.

Example automation action:

```yaml
actions:
  - action: rest_command.home_optimizer_api
    data:
      endpoint: tune
      payload: |
        {{ {
          "days": 90,
          "forecaster": "solar",
          "trails": 5
        } | to_json }}
```

## Home Assistant Setup

Add the following configurations to your Home Assistant `configuration.yaml` to enable
communication and prepare the required forecast sensors.

### 1. REST Command

This command allows Home Assistant to send automations and payloads to the Home
Optimizer API.

```yaml
rest_command:
  home_optimizer_api:
    url: "http://127.0.0.1:8099/api/{{ endpoint }}"
    method: POST
    headers:
      content-type: "application/json"
    payload: "{{ payload }}"
```

### 2. Open-Meteo Forecast (REST Sensor)

This sensor fetches 15-minute interval weather metrics (irradiance, temperature, wind,
and clouds) from the Open-Meteo API.

```yaml
rest:
  - resource: "https://api.open-meteo.com/v1/forecast\
      ?latitude=YOUR_LATITUDE\
      &longitude=YOUR_LONGITUDE\
      &tilt=YOUR_TILT\
      &azimuth=YOUR_AZIMUTH\
      &minutely_15=\
        is_day,\
        temperature_2m,\
        relative_humidity_2m,\
        global_tilted_irradiance,\
        direct_radiation,\
        direct_normal_irradiance,\
        diffuse_radiation,\
        precipitation,\
        wind_speed_10m,\
        wind_direction_10m,\
        cloud_cover_low,\
        cloud_cover_mid,\
        cloud_cover_high\
      &forecast_days=2\
      &timezone=UTC"
    scan_interval: 1800
    sensor:
      - name: "Open-Meteo Forecast"
        unique_id: open_meteo_forecast
        value_template: "{{ now() }}"
        device_class: timestamp
        json_attributes_path: "$.minutely_15"
        json_attributes:
          - time
          - is_day
          - temperature_2m
          - relative_humidity_2m
          - global_tilted_irradiance
          - direct_radiation
          - direct_normal_irradiance
          - diffuse_radiation
          - precipitation
          - wind_speed_10m
          - wind_direction_10m
          - cloud_cover_low
          - cloud_cover_mid
          - cloud_cover_high
```

### 3. Solcast PV Forecast (Template Sensor)

This sensor aggregates the Solcast today and tomorrow forecasts.

```yaml
template:
  - sensor:
      - name: "Solcast PV Forecast"
        unique_id: solcast_pv_forecast
        device_class: timestamp
        state: "{{ states('sensor.solcast_pv_forecast_api_last_polled') }}"
        attributes:
          time: >
            {% set forecast =
              state_attr('sensor.solcast_pv_forecast_forecast_today', 'detailedForecast')
              + state_attr('sensor.solcast_pv_forecast_forecast_tomorrow', 'detailedForecast')
            %}
            {{ forecast 
              | map(attribute='period_start') 
              | map('as_timestamp') 
              | map('timestamp_custom', '%Y-%m-%dT%H:%M:%S%z')
              | list 
            }}
          pv_estimate: >
            {% set forecast =
              state_attr('sensor.solcast_pv_forecast_forecast_today', 'detailedForecast')
              + state_attr('sensor.solcast_pv_forecast_forecast_tomorrow', 'detailedForecast')
            %}
            {{ forecast 
              | map(attribute='pv_estimate') 
              | map('multiply', 1000.0) 
              | list 
            }}
          pv_estimate10: >
            {% set forecast =
              state_attr('sensor.solcast_pv_forecast_forecast_today', 'detailedForecast')
              + state_attr('sensor.solcast_pv_forecast_forecast_tomorrow', 'detailedForecast')
            %}
            {{ forecast 
              | map(attribute='pv_estimate10') 
              | map('multiply', 1000.0) 
              | list 
            }}
          pv_estimate90: >
            {% set forecast =
              state_attr('sensor.solcast_pv_forecast_forecast_today', 'detailedForecast')
              + state_attr('sensor.solcast_pv_forecast_forecast_tomorrow', 'detailedForecast')
            %}
            {{ forecast
              | map(attribute='pv_estimate90')
              | map('multiply', 1000.0) 
              | list 
            }}
```

### 4. InfluxDB Integration

Configure the InfluxDB integration in Home Assistant.

```yaml
influxdb:
  host: a0d7b954-influxdb
  port: 8086
  database: home_assistant
  username: YOUR_USERNAME
  password: YOUR_PASSWORD
  include:
    entities:
      - sensor.solcast_pv_forecast
      - sensor.open_meteo_forecast
      - ...
```

## Development

To run Home Optimizer locally:

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -e ".[dev]"
  ./run.sh
```