from __future__ import annotations

WMO_WEATHER_CODE_LABELS_NL = {
    0: "Helder",
    1: "Vrij helder",
    2: "Half bewolkt",
    3: "Bewolkt",
    45: "Mist",
    48: "Rijpmist",
    51: "Lichte motregen",
    53: "Motregen",
    55: "Zware motregen",
    56: "Lichte ijzel",
    57: "Ijzel",
    61: "Lichte regen",
    63: "Regen",
    65: "Zware regen",
    66: "Lichte ijsregen",
    67: "IJsregen",
    71: "Lichte sneeuw",
    73: "Sneeuw",
    75: "Zware sneeuw",
    77: "Sneeuwvlokken",
    80: "Lichte buien",
    81: "Buien",
    82: "Hevige buien",
    85: "Lichte sneeuwbuien",
    86: "Hevige sneeuwbuien",
    95: "Onweer",
    96: "Onweer met hagel",
    99: "Zwaar onweer met hagel",
}


def weather_code_label(code: int | float) -> str:
    return WMO_WEATHER_CODE_LABELS_NL.get(int(code), "Onbekend")


__all__ = ["WMO_WEATHER_CODE_LABELS_NL", "weather_code_label"]

