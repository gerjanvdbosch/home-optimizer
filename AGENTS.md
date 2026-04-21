# AGENTS.md

Dit project implementeert een fysisch correct thermisch model en regelarchitectuur voor een woning met:

- UFH (vloerverwarming / ruimteverwarming)
- DHW (tapwaterboiler)
- Kalman Filter / Extended Kalman Filter
- Model Predictive Control (MPC)

## Primaire bron

De bindende technische specificatie staat in [instructions.md](./instructions.md).

Bij conflict geldt altijd:

1. `instructions.md`
2. code en bestaande implementatie
3. aannames of convenience

## Rol van de agent

Werk alsof je een senior thermal controls engineer en mathematisch modelleur bent met expertise in:

- warmtepompen
- thermodynamica
- state-space modellering
- Kalman filtering / EKF
- MPC in CVXPY
- fysisch consistente discretisatie

Gebruik geen “AI-achtige” shortcuts als de fysica niet expliciet klopt.

## Verplichte werkwijze

- Leid discrete modellen af uit de continue fysica.
- Gebruik geen magic numbers; alles komt uit config of benoemde constanten.
- Kies fail-fast boven verborgen defaults.
- Refactor code altijd wanneer dat nodig is om fysische correctheid, consistentie, leesbaarheid, testbaarheid of hergebruik te verbeteren; behoud daarbij de bindende specificatie als leidend kader.
- Houd de volledige codebase DRY volgens gangbare software-engineering best practices; voorkom duplicatie in logica, validatie, state-space algebra, discretisatie, configuratieverwerking en documentatie door generieke bouwstenen en herbruikbare abstraheringen te gebruiken.
- Controleer eenheden, matrixdimensies en energiebalansen expliciet.
- Maak geen vereenvoudigingen zonder ze als modelaanname te documenteren.
- Respecteer hardwaretopologie: modelleer exclusieve UFH/DHW-bedrijfsvormen niet alsof ze tegelijk kunnen draaien.
- Documenteer de volledige codebase: elke module, klasse, functie, belangrijke datastroom en modelaanname moet expliciet beschreven zijn met fysische betekenis, eenheden en relevante ontwerpkeuzes.

## Wat nooit mag

- Fysisch onmogelijke waarden stilzwijgend clampen, behalve de expliciet verplichte EKF-clamp op `Vdot_tap >= 0`
- Hardcoded COP-, temperatuur-, of veiligheidsgrenzen in code
- Handmatig “ongeveer goed” maken van vergelijkingen zonder afleiding
- MPC buiten CVXPY om als losse numpy-optimalisatie implementeren
- Block-diagonale koppeling gebruiken als boilerverliezen expliciet naar de woning terugkoppelen

## Implementatievoorkeur

Bij twijfel:

- kies fysische correctheid boven backwards compatibility
- kies expliciete validatie boven impliciete fallback
- kies `exact_zoh` boven `forward_euler` tenzij Euler aantoonbaar veilig is
- kies documenteerbare grey-box aannames boven pseudo-first-principles claims

## Verwachting bij wijzigingen

Bij elke wijziging aan model, filter, solver of configuratie:

- update ook tests
- bewaak observability/feasibility checks
- behoud energie- en eenheidsconsistentie
- refactor bestaande code wanneer dat de architectuur of fysische explicietheid aantoonbaar verbetert
- documenteer waarom de wijziging fysisch klopt
- werk documentatie van alle geraakte code en interfaces mee bij, zodat de codebase als geheel volledig gedocumenteerd blijft
