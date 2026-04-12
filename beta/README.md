# 2-State Thermisch Model voor Vloerverwarming (UFH)

## 1. Doel van het model

Dit model beschrijft de thermische dynamica van een woning met vloerverwarming en wordt gebruikt als basis voor Model Predictive Control (MPC). Het model bevat thermische opslag, vertraging en externe verstoringen (weer en zon).

---

## 2. Modelstructuur

Het systeem bestaat uit twee thermische states:

* **T_r (Room temperature)**
  Gemeten kamertemperatuur

* **T_b (Buffer temperature)**
  Effectieve temperatuur van vloer + gebouwmassa (niet gemeten)

---

## 3. Continue fysische vergelijking

### Buffer (vloer / massa)

C_b * dT_b/dt = P_UFH - (T_b - T_r) / R_br

### Ruimte (lucht + directe omgeving)

C_r * dT_r/dt = (T_b - T_r) / R_br - (T_r - T_out) / R_ro + α * Q_solar

---

## 4. Discrete vorm (implementatie)

Met tijdstap Δt:

T_b[k+1] = T_b[k] + (Δt / C_b) * (P_UFH[k] - (T_b[k] - T_r[k]) / R_br)

T_r[k+1] = T_r[k] + (Δt / C_r) * ((T_b[k] - T_r[k]) / R_br - (T_r[k] - T_out[k]) / R_ro + α * Q_solar[k])

---

## 5. Zoninstraling (Open-Meteo GTI)

Gebruik **Global Tilted Irradiance (GTI)** van Open-Meteo.

### Fysisch model:

Q_solar = A_glass * GTI(t) * η

waar:

* A_glass = effectief glasoppervlak [m²]
* GTI(t) = zoninstraling [W/m²]
* η = transmissiefactor (0.5–0.8)

---

### Invoer in model:

* Q_solar wordt toegevoegd aan T_r dynamica
* Optioneel deels naar T_b voor thermische opslag

---

## 6. Variabelen en betekenis

### States

* **T_r [°C]**: kamertemperatuur (gemeten)
* **T_b [°C]**: thermische massa (vloer/constructie)

---

### Inputs

* **P_UFH [kW]**
  Warmtevermogen vloerverwarming

  Berekening:
  P_UFH = flow_kg_s * 4.18 * (T_supply - T_return)

* **T_out [°C]**
  Buitentemperatuur

* **Q_solar [kW]**
  Zoninstraling via GTI

---

### Parameters

* **C_r [Wh/K]**
  Thermische capaciteit van de ruimte

* **C_b [Wh/K]**
  Thermische capaciteit van vloer/gebouwmassa

* **R_br [K/kW]**
  Warmteweerstand tussen vloer en ruimte

* **R_ro [K/kW]**
  Warmteweerstand tussen ruimte en buiten

* **α [-]**
  Fractie zonnewarmte direct naar ruimte (typisch 0.6–0.8)

* **η [-]**
  Glas transmissiefactor (0.5–0.8)

* **A_glass [m²]**
  Effectief glasoppervlak

---

### Constante

* **Δt [uur]**
  Tijdstap (bijv. 5 min = 1/12 uur)

---

## 7. Eenhedenconsistentie

Aanbevolen set voor Home Assistant:

* Vermogen: kW
* Tijd: uur
* Capaciteit: Wh/K

Dan geldt:

T[k+1] = T[k] + (Δt / C) * P

---

## 8. DHW model (apart systeem)

T_dhw[k+1] = T_dhw[k] + α_dhw * P_dhw - β_dhw * (T_dhw - T_out)

---

## 9. Interpretatie van het model

* T_b representeert thermische opslag (vloer + massa)
* UFH verwarmt eerst de buffer, daarna de ruimte
* Zoninstraling komt via fysiek gemeten GTI
* Warmteverlies gebeurt via R_ro

---

## 10. Gebruik in MPC

Het model wordt herschreven naar state-space:

x[k+1] = A x[k] + B u[k] + E d[k]

waar:

* x = [T_r, T_b]
* u = P_UFH
* d = [T_out, Q_solar]

---

## 11. Doel van MPC

Optimalisatie van:

* energiegebruik (PV benutting)
* grid import minimaliseren
* COP optimalisatie warmtepomp
* comfort (T_r binnen band)

---

## 12. Belangrijkste eigenschap

Dit model maakt het mogelijk om:

* thermische opslag van vloerverwarming te benutten
* zoninstraling fysisch correct mee te nemen
* voorspellend te sturen op basis van weerdata

Zonder tweede state (T_b) en GTI-invoer is dit gedrag niet reproduceerbaar.
