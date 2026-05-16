from __future__ import annotations

from home_optimizer.domain import weather_code_label


def test_weather_code_label_returns_short_dutch_labels() -> None:
    assert weather_code_label(0) == "helder"
    assert weather_code_label(1) == "vrij helder"
    assert weather_code_label(2) == "half bewolkt"
    assert weather_code_label(3) == "bewolkt"
    assert weather_code_label(45) == "mist"
    assert weather_code_label(48) == "rijpmist"
    assert weather_code_label(51) == "lichte motregen"
    assert weather_code_label(57) == "ijzel"
    assert weather_code_label(61) == "lichte regen"
    assert weather_code_label(65) == "zware regen"
    assert weather_code_label(71) == "lichte sneeuw"
    assert weather_code_label(75) == "zware sneeuw"
    assert weather_code_label(77) == "sneeuwvlokken"
    assert weather_code_label(80) == "lichte buien"
    assert weather_code_label(82) == "hevige buien"
    assert weather_code_label(85) == "lichte sneeuwbuien"
    assert weather_code_label(86) == "hevige sneeuwbuien"
    assert weather_code_label(95) == "onweer"
    assert weather_code_label(96) == "onweer met hagel"
    assert weather_code_label(99) == "zwaar onweer met hagel"


def test_weather_code_label_returns_onbekend_for_unknown_codes() -> None:
    assert weather_code_label(999) == "onbekend"
    assert weather_code_label(3.0) == "bewolkt"

