# 2-State Thermisch Model voor Vloerverwarming (UFH)

## 1. Doel van het model

Grey-box thermisch model voor woning-MPC met:

- vloerverwarming (UFH)
- thermische massa (buffer)
- zoninstraling (GTI)
- interne warmtewinsten
- Kalman state estimation
- MPC optimalisatie

---

## 2. Modelstructuur

**States**
- T_r: gemeten ruimtetemperatuur [°C]
- T_b: niet-gemeten thermische massa [°C]

---

## 3. Continue fysica

**Buffer:**
```
C_b · dT_b/dt = P_UFH − (T_b − T_r)/R_br + (1 − α) · Q_solar
```

**Ruimte:**
```
C_r · dT_r/dt = (T_b − T_r)/R_br − (T_r − T_out)/R_ro + α · Q_solar + Q_int
```

---

## 4. Warmtestromen

- P_UFH → regelbare warmte-injectie via vloer
- (T_b − T_r)/R_br → interne koppeling vloer ↔ ruimte
- (T_r − T_out)/R_ro → warmteverlies naar buiten
- Q_solar → zoninstraling via glas, verdeeld over lucht (α) en massa (1−α)
- Q_int → interne warmtewinsten (personen, apparatuur)

---

## 5. Discrete vorm (Forward Euler)

```
T_b[k+1] = T_b[k] + (Δt / C_b) · (P_UFH[k] − (T_b[k] − T_r[k])/R_br + (1 − α) · Q_solar[k])

T_r[k+1] = T_r[k] + (Δt / C_r) · ((T_b[k] − T_r[k])/R_br − (T_r[k] − T_out[k])/R_ro + α · Q_solar[k] + Q_int[k])
```

---

## 6. Zoninstraling (GTI)

```
Q_solar [kW] = A_glass · GTI · η / 1000
```

- GTI: Global Tilted Irradiance [W/m²]
- A_glass: effectief glasoppervlak [m²]
- η: transmissiefactor [-]

---

## 7. Variabelen

**States**
- T_r [°C]
- T_b [°C]

**Inputs**
- P_UFH [kW]
- T_out [°C]
- Q_solar [kW]
- Q_int [kW]

---

## 8. Modelparameters

| Parameter | Eenheid | Betekenis |
|---|---|---|
| C_r | kWh/K | warmtecapaciteit lucht + lichte massa |
| C_b | kWh/K | warmtecapaciteit vloer + gebouwmassa |
| R_br | K/kW | thermische weerstand vloer ↔ ruimte |
| R_ro | K/kW | thermische weerstand ruimte ↔ buiten |
| α | − | fractie zon direct naar lucht |
| η | − | glastransmissie |
| A_glass | m² | effectief glasoppervlak |
| P_max | kW | maximaal UFH vermogen |
| T_min | °C | ondergrens comforttemperatuur |
| T_max | °C | bovengrens comforttemperatuur |
| ΔP_max | kW/stap | maximale rampsnelheid UFH |

---

## 9. MPC parameters

| Parameter | Eenheid | Betekenis |
|---|---|---|
| N | − | voorspelhorizon (typisch 12–48 stappen) |
| Q_c | K⁻² | comfortgewicht (straf op T_r afwijking) |
| R_c | kW⁻² | energiegewicht (straf op P_UFH) |
| Q_N | K⁻² | eindtermgewicht |

---

## 10. Kalman ruisparameters

| Parameter | Eenheid | Betekenis |
|---|---|---|
| Q_n | K² | procesnuis: diag(σ_b², σ_r²) |
| R_n | K² | meetruis T_r sensor: σ_y² |

---

## 11. Discretisatie

```
Δt [uur]
T[k+1] = T[k] + (Δt / C) · P
```

Eenhedenconsistentie: kW · uur = kWh, kWh / (kWh/K) = K ✓

---

## 12. DHW model

```
T_dhw[k+1] = T_dhw[k] + (Δt / C_dhw) · P_dhw − (Δt / (C_dhw · R_dhw)) · (T_dhw − T_room)
```

T_room is fysisch correcter dan T_out voor een boiler binnenshuis.

---

## 13. State-space model

```
x[k+1] = A x[k] + B u[k] + E d[k]

x = [T_r, T_b]ᵀ
u = P_UFH
d = [T_out, Q_solar, Q_int]ᵀ
```

---

## 14. Matrices

```
a_br = Δt / (C_r · R_br)
a_ro = Δt / (C_r · R_ro)
b_br = Δt / (C_b · R_br)

A = [ 1 − a_br − a_ro,   a_br   ]
    [ b_br,               1−b_br ]

B = [ 0        ]
    [ Δt / C_b ]

E = [ a_ro,   α·Δt/C_r,       Δt/C_r ]
    [ 0,      (1−α)·Δt/C_b,   0      ]
```

---

## 15. Observeerbaarheid

```
O = [  C  ] = [ 1,              0    ]
    [ CA  ]   [ 1−a_br−a_ro,   a_br  ]

rank(O) = 2  voor alle R_br < ∞  ✓
```

T_b is volledig reconstrueerbaar uit T_r metingen.

---

## 16. Regelbaarheid

```
Γ = [B, AB]

B   = [ 0,        Δt/C_b ]ᵀ
AB  = [ a_br·Δt/C_b,   (1−b_br)·Δt/C_b ]ᵀ

det(Γ) = −(Δt)² · a_br / C_b² ≠ 0  voor R_br < ∞

rank(Γ) = 2  ✓
```

T_b is volledig stuurbaar via P_UFH.

---

## 17. Kalman filter — procesmodel

```
x[k+1] = A x[k] + B u[k] + E d[k] + w[k]
y[k]   = C x[k] + v[k]

C = [1, 0]
w ~ N(0, Q_n)
v ~ N(0, R_n)
```

---

## 18. Kalman filter — recursie

**Predict:**
```
x̂⁻[k] = A x̂[k−1] + B u[k−1] + E d[k−1]
P⁻[k]  = A P[k−1] Aᵀ + Q_n
```

**Update:**
```
K[k]  = P⁻[k] Cᵀ · (C P⁻[k] Cᵀ + R_n)⁻¹
x̂[k] = x̂⁻[k] + K[k] · (y[k] − C x̂⁻[k])
P[k]  = (I − K[k] C) P⁻[k]
```

---

## 19. MPC kostfunctie

```
J = Σ_{k=0}^{N−1} [ Q_c · (T_r[k] − T_ref[k])² + R_c · P_UFH[k]² ]
  + Q_N · (T_r[N] − T_ref[N])²
```

De eindterm Q_N valt buiten de som en stabiliseert de voorspellende horizon.

---

## 20. Constraints

```
0       ≤  P_UFH[k]                   ≤  P_max
T_min   ≤  T_r[k]                     ≤  T_max
         |P_UFH[k] − P_UFH[k−1]|      ≤  ΔP_max
```

---

## 21. Stabiliteit

Formeel criterium:
```
|λ(A)| < 1
```

Alle eigenwaarden van A moeten binnen de eenheidscirkel liggen. Bij woningparameters is het systeem sterk gedempt en ruimschoots stabiel.

---

## 22. Interne warmtewinsten

```
Q_int ≈ 0.08–0.12 kW per persoon + achtergrondapparatuur
```

Typisch totaal in een woning: 0.2–0.8 kW.

---

## 23. Architectuur

```
                    ┌──────────────────────────┐
                    ↓                          │
d ──→ Kalman filter ──→ x̂ ──→ MPC optimizer ──→ P_UFH
         ↑                         ↓
      y (T_r)            Thermisch model
                                   ↑
                              P_UFH + d
```

---

## 24. Systeemstructuur

- 2 states: T_r, T_b
- 1 actuator: P_UFH
- 3 disturbances: T_out, Q_solar, Q_int
- 1 estimator: Kalman filter
- 1 optimizer: MPC (cvxpy of gelijkwaardig)

---

## 25. Kernresultaat

Volledig gesloten regelkring voor woningthermiek:

- fysisch RC-netwerk (grey-box)
- observeerbare en regelbare hidden state T_b
- Kalman reconstructie van thermische massa
- MPC optimalisatie onder vermogen- en comfortconstraints
- uitbreidbaar naar koeling (P_UFH < 0) en DHW