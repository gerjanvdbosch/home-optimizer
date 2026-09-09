import numpy as np
import pytest

from domain.types import BoilerThermalModel, MPCConfig, MPCInput
from features.boiler import CP_WATER_J_PER_KG_K, RHO_WATER_KG_PER_L, discretize_zoh
from features.optimizer import MPCOptimizer, _lumped_state_space

THERMAL_MODEL = BoilerThermalModel(
    volume_l=200.0,
    ua_top_w_per_k=0.15,
    ua_bottom_w_per_k=0.20,
    ua_mix_idle_w_per_k=0.03,
    ua_mix_active_w_per_k=9638.6,
    q_in_nominal_w=3700.0,
)

# Solar rising to a midday peak then falling, 24 steps of 15 minutes (6 hours).
_SOLAR_MIDDAY_W = [500, 1000, 2000, 3000, 3500, 3000, 2000, 1000, 500, 0.0]
SOLAR_FORECAST_W = [0.0] * 8 + _SOLAR_MIDDAY_W + [0.0] * 6


def _make_input(**overrides) -> MPCInput:
    defaults = dict(
        solar_forecast_w=SOLAR_FORECAST_W,
        ambient_temperature=20.0,
        current_temp_top=30.0,
        current_temp_bottom=28.0,
        boiler_on_current=False,
        thermal_model=THERMAL_MODEL,
        target_temperature_top=(10.0,) * len(SOLAR_FORECAST_W),
    )
    defaults.update(overrides)
    return MPCInput(**defaults)


def test_schedules_heating_during_solar_peak_when_sufficient():
    """Regression test for the stated goal: with no draw-off requirement except a
    45 degC deadline that solar alone can satisfy, the optimizer must run the
    boiler during the solar peak (not before, not after) and must not run once
    the target is met - a real bug (see test_boiler_can_turn_off_after_target)
    once made it impossible to ever stop heating once started.
    """

    target = [10.0] * len(SOLAR_FORECAST_W)
    target[18] = 45.0

    data = _make_input(target_temperature_top=tuple(target))

    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig())
    result = optimizer.solve(data)

    on_steps = [k for k, v in enumerate(result.schedule) if v == 1]

    assert len(on_steps) > 0
    # The solar peak is at index 12; heating must overlap it, not happen only
    # long before or after it.
    assert min(on_steps) <= 14
    assert max(on_steps) < 18  # done heating before the deadline check

    # The requirement is actually met.
    assert result.temperatures[18] >= 45.0 - 1e-6

    # It must not keep heating past the point the target is satisfied and
    # exceeded (that would just waste money for no benefit).
    assert result.schedule[-1] == 0


def test_boiler_can_turn_off_after_reaching_target():
    """Regression test for a real bug: an equality startup constraint
    (boiler_start[k] == boiler_on[k] - boiler_on[k-1]) forces boiler_start to -1
    on every stop event, which is infeasible against its own binary domain -
    making any schedule that ever turns the boiler back off unsolvable, so the
    optimizer was forced to keep it on forever once started. A schedule that
    turns on, then off again, must be achievable.
    """

    target = [10.0] * len(SOLAR_FORECAST_W)
    target[10] = 40.0  # an early, modest requirement well before the horizon ends

    data = _make_input(target_temperature_top=tuple(target))

    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig())
    result = optimizer.solve(data)

    # Must turn off again at some point after turning on - not stay on for the
    # rest of the horizon once started.
    assert 0 in result.schedule
    assert 1 in result.schedule

    first_on = result.schedule.index(1)
    later_off = any(v == 0 for v in result.schedule[first_on:])
    assert later_off


def test_tap_forecast_adds_an_additional_heat_sink_to_the_temperature_trajectory():
    """The tap-draw forecast (see MPCInput.tap_forecast_w) must genuinely affect
    the predicted temperature trajectory as an additional heat sink on top of
    passive UA loss, not just be accepted and ignored.
    """

    low_target = (10.0,) * len(SOLAR_FORECAST_W)
    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig())

    baseline = optimizer.solve(_make_input(target_temperature_top=low_target))

    with_draws = optimizer.solve(
        _make_input(
            target_temperature_top=low_target,
            # Small enough that the resulting drop stays well above the 10 degC
            # floor (starting ~29 degC) - isolates the tap term's effect on the
            # trajectory without also triggering a heating-decision difference.
            tap_forecast_w=(300.0,) * len(SOLAR_FORECAST_W),
        )
    )

    # The target is already below the starting temperature, so the boiler need
    # not run at all either way - this isolates the tap term's effect on the
    # simulated trajectory from any heating-decision difference.
    assert baseline.schedule == with_draws.schedule
    assert all(v == 0 for v in baseline.schedule)
    assert all(
        with_draws.temperatures[k] < baseline.temperatures[k] - 1e-6
        for k in range(1, len(SOLAR_FORECAST_W))
    )


def test_tap_forecast_can_force_additional_heating_before_a_deadline():
    """A large forecasted draw between the last heating opportunity and a
    deadline must force the optimizer to heat more than it would without it -
    the entire point of making tap usage visible to planning.
    """

    target = [10.0] * len(SOLAR_FORECAST_W)
    target[18] = 45.0

    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig())
    baseline = optimizer.solve(_make_input(target_temperature_top=tuple(target)))

    tap_forecast = [0.0] * len(SOLAR_FORECAST_W)
    for k in range(13, 18):
        tap_forecast[k] = 5000.0

    with_draw = optimizer.solve(
        _make_input(
            target_temperature_top=tuple(target),
            tap_forecast_w=tuple(tap_forecast),
        )
    )

    assert with_draw.temperatures[18] >= 45.0 - 1e-6
    assert sum(with_draw.schedule) > sum(baseline.schedule)


def test_thermal_dynamics_matches_manual_discretization():
    """The pyomo thermal_dynamics constraints must reproduce exactly the same
    trajectory as directly calling discretize_zoh with the lumped state-space -
    verifies the MPC's own dynamics formulation against the shared physics
    utility, independent of the solver's optimization behavior.
    """

    data = _make_input(boiler_on_current=True)

    # Minimum runtime of 1 so an arbitrary short on/off pattern doesn't conflict
    # with the (unrelated) scheduling constraint this test isn't exercising.
    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig(boiler_min_runtime_steps=1))
    model = optimizer._build_model(data)

    # Force a known, arbitrary on/off pattern and re-derive T by hand.
    pattern = [1, 1, 0, 0, 1, 0] * 4
    pattern = pattern[: len(SOLAR_FORECAST_W)]

    for k, v in enumerate(pattern):
        model.boiler_on[k].fix(v)

    from pyomo.contrib.appsi.solvers.highs import Highs

    results = Highs().solve(model)

    from pyomo.contrib.appsi.base import TerminationCondition

    assert results.termination_condition == TerminationCondition.optimal

    import pyomo.environ as pyo

    solved_T = [pyo.value(model.T[k]) for k in range(len(pattern))]

    a, b = _lumped_state_space(
        THERMAL_MODEL.volume_l,
        THERMAL_MODEL.ua_top_w_per_k + THERMAL_MODEL.ua_bottom_w_per_k,
    )
    a_d, b_d = discretize_zoh(a, b, MPCConfig().step_hours * 3600.0)

    expected_T = [(data.current_temp_top + data.current_temp_bottom) / 2.0]
    for k in range(len(pattern) - 1):
        q_in = THERMAL_MODEL.q_in_nominal_w if pattern[k] else 0.0
        next_t = (
            a_d[0, 0] * expected_T[-1]
            + b_d[0, 0] * data.ambient_temperature
            + b_d[0, 1] * q_in
        )
        expected_T.append(float(next_t))

    assert np.allclose(solved_T, expected_T, atol=1e-6)


def test_minimum_runtime_is_respected():
    target = [10.0] * len(SOLAR_FORECAST_W)
    target[10] = 40.0

    config = MPCConfig(boiler_min_runtime_steps=4)
    data = _make_input(target_temperature_top=tuple(target))

    optimizer = MPCOptimizer(THERMAL_MODEL, config)
    result = optimizer.solve(data)

    schedule = result.schedule
    k = 0
    while k < len(schedule):
        if schedule[k] == 1 and (k == 0 or schedule[k - 1] == 0):
            run_length = 0
            j = k
            while j < len(schedule) and schedule[j] == 1:
                run_length += 1
                j += 1
            assert run_length >= config.boiler_min_runtime_steps or j == len(
                schedule
            )
            k = j
        else:
            k += 1


def test_validate_input_rejects_mismatched_target_length():
    data = _make_input(target_temperature_top=(10.0, 10.0))  # wrong length

    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig())

    with pytest.raises(ValueError):
        optimizer.solve(data)


def test_no_heating_scheduled_when_target_already_below_current():
    """If the whole horizon's requirement is already satisfied by the current
    temperature, the optimizer must not spend money heating anyway.
    """

    data = _make_input(
        current_temp_top=50.0,
        current_temp_bottom=50.0,
        target_temperature_top=(10.0,) * len(SOLAR_FORECAST_W),
    )

    optimizer = MPCOptimizer(THERMAL_MODEL, MPCConfig())
    result = optimizer.solve(data)

    assert all(v == 0 for v in result.schedule)
    assert result.objective_value == pytest.approx(0.0, abs=1e-9)


def test_lumped_state_space_matches_full_tank_capacity():
    c_expected = RHO_WATER_KG_PER_L * THERMAL_MODEL.volume_l * CP_WATER_J_PER_KG_K
    ua_total = THERMAL_MODEL.ua_top_w_per_k + THERMAL_MODEL.ua_bottom_w_per_k

    a, b = _lumped_state_space(THERMAL_MODEL.volume_l, ua_total)

    assert a.shape == (1, 1)
    assert b.shape == (1, 3)
    assert a[0, 0] == pytest.approx(-ua_total / c_expected)
    assert b[0, 0] == pytest.approx(ua_total / c_expected)
    assert b[0, 1] == pytest.approx(1.0 / c_expected)
    # Tap draw is a heat sink, opposite sign to Q_in - same magnitude since
    # both terms are a straight W/C_total conversion.
    assert b[0, 2] == pytest.approx(-1.0 / c_expected)
