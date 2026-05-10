# Home Optimizer

This is a physical 2-state RC grey-box model for room temperature forecasting.

## States

- `T_air`: room air temperature
- `T_mass`: effective thermal mass temperature

## Physical parameters

- `R_air_out`
- `R_air_mass`
- `R_mass_out`
- `C_air`
- `C_mass`
- `eta_heat`
- `eta_solar_air`
- `eta_solar_mass`

The model fits explicit physical resistances, capacitances, and gains. It does not fit a free ARX model or free discrete-time `A/B` state-space model directly.

## Solar calculation

The direct solar gain through glazing is computed explicitly as:

```text
solar_glass_kw = irradiance_wm2 * 8.0 * g_glass * shutter_factor / 1000
```

By default:

- `glass_area_m2 = 8.0`
- `g_glass = 0.50`

## Shutter mode

Supported shutter interpretations:

- `open_percent`
- `closed_percent`

If `shutter_mode == "open_percent"`:

```text
shutter_factor = shutter_position / 100.0
```

If `shutter_mode == "closed_percent"`:

```text
shutter_factor = 1.0 - shutter_position / 100.0
```

The factor is always clamped to `[0, 1]`.

Important:

- If `100` means fully open, use `open_percent`.
- If `100` means fully closed, use `closed_percent`.

## Example

```python
from home_optimizer.features.modeling.room_rc import RoomRC2StatePhysicalModel, RoomRC2StateConfig

config = RoomRC2StateConfig(
    interval_minutes=10,
    glass_area_m2=8.0,
    g_glass=0.50,
    shutter_mode="open_percent",
    alpha_solar=0.85,
    alpha_heat=0.70,
)

model = RoomRC2StatePhysicalModel(config)

fit_report = model.fit(
    train_df,
    validation_df=valid_df,
    horizons=(1, 6, 36, 72),
)

metrics = model.evaluate(
    test_df,
    horizons=(1, 6, 36, 72, 144),
)

segment_metrics = model.evaluate_segments(
    test_df,
    horizons=(1, 6, 36, 72, 144),
)

forecast_df = model.forecast(
    future_df,
    horizon_steps=72,
    last_measurement_c=latest_room_temp_c,
)

model.save("room_rc_2state.json")
```

## Warning

Set `shutter_mode` correctly:

- if `100` means fully open: use `open_percent`
- if `100` means fully closed: use `closed_percent`
