from __future__ import annotations

from home_optimizer.domain import weather_code_label


def test_weather_code_label_returns_short_dutch_labels() -> None:
    assert weather_code_label(0) == "Helder"
    assert weather_code_label(1) == "Vrij helder"
    assert weather_code_label(2) == "Half bewolkt"
    assert weather_code_label(3) == "Bewolkt"
    assert weather_code_label(45) == "Mist"
    assert weather_code_label(48) == "Rijpmist"
    assert weather_code_label(51) == "Lichte motregen"
    assert weather_code_label(57) == "Ijzel"
    assert weather_code_label(61) == "Lichte regen"
    assert weather_code_label(65) == "Zware regen"
    assert weather_code_label(71) == "Lichte sneeuw"
    assert weather_code_label(75) == "Zware sneeuw"
    assert weather_code_label(77) == "Sneeuwvlokken"
    assert weather_code_label(80) == "Lichte buien"
    assert weather_code_label(82) == "Hevige buien"
    assert weather_code_label(85) == "Lichte sneeuwbuien"
    assert weather_code_label(86) == "Hevige sneeuwbuien"
    assert weather_code_label(95) == "Onweer"
    assert weather_code_label(96) == "Onweer met hagel"
    assert weather_code_label(99) == "Zwaar onweer met hagel"


def test_weather_code_label_returns_onbekend_for_unknown_codes() -> None:
    assert weather_code_label(999) == "Onbekend"
    assert weather_code_label(3.0) == "Bewolkt"

