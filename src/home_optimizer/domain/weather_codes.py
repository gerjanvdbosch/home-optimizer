from __future__ import annotations

WMO_WEATHER_CODE_LABELS_NL = {
    0: "helder",
    1: "vrij helder",
    2: "half bewolkt",
    3: "bewolkt",
    45: "mist",
    48: "rijpmist",
    51: "lichte motregen",
    53: "motregen",
    55: "zware motregen",
    56: "lichte ijzel",
    57: "ijzel",
    61: "lichte regen",
    63: "regen",
    65: "zware regen",
    66: "lichte ijsregen",
    67: "ijsregen",
    71: "lichte sneeuw",
    73: "sneeuw",
    75: "zware sneeuw",
    77: "sneeuwvlokken",
    80: "lichte buien",
    81: "buien",
    82: "hevige buien",
    85: "lichte sneeuwbuien",
    86: "hevige sneeuwbuien",
    95: "onweer",
    96: "onweer met hagel",
    99: "zwaar onweer met hagel",
}


def weather_code_label(code: int | float) -> str:
    return WMO_WEATHER_CODE_LABELS_NL.get(int(code), "onbekend")


__all__ = ["WMO_WEATHER_CODE_LABELS_NL", "weather_code_label"]

