from __future__ import annotations

from datetime import datetime, timezone

from home_optimizer.features.mpc import ThermostatSetpointCandidateGenerator


def test_generate_constant_candidates_builds_one_schedule_per_setpoint() -> None:
    generator = ThermostatSetpointCandidateGenerator()

    candidates = generator.generate_constant_candidates(
        start_time=datetime(2026, 5, 2, 10, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 5, 2, 11, 0, tzinfo=timezone.utc),
        interval_minutes=15,
        allowed_setpoints=[19.0, 20.0, 21.0],
    )

    assert len(candidates) == 3
    assert [candidate.schedule.points[0].value for candidate in candidates] == [19.0, 20.0, 21.0]
    assert all(len(candidate.schedule.points) == 5 for candidate in candidates)


def test_generate_single_switch_candidates_builds_switch_scenarios() -> None:
    generator = ThermostatSetpointCandidateGenerator()

    candidates = generator.generate_single_switch_candidates(
        start_time=datetime(2026, 5, 2, 10, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 5, 2, 11, 0, tzinfo=timezone.utc),
        interval_minutes=15,
        allowed_setpoints=[19.0, 20.0],
        switch_times=[
            datetime(2026, 5, 2, 10, 15, tzinfo=timezone.utc),
            datetime(2026, 5, 2, 10, 45, tzinfo=timezone.utc),
        ],
    )

    assert len(candidates) == 4
    first_points = candidates[0].schedule.points
    assert [point.value for point in first_points] == [19.0, 20.0, 20.0, 20.0, 20.0]
    last_points = candidates[-1].schedule.points
    assert [point.value for point in last_points] == [20.0, 20.0, 20.0, 19.0, 19.0]
