# 2-State Thermisch Model voor Vloerverwarming (UFH)

---

## 1. Doel van het model

Dit model beschrijft de thermische dynamica van een woning met vloerverwarming voor Model Predictive Control (MPC).

Het model bevat:

* thermische opslag (vloer + gebouwmassa)
* warmteverlies naar buiten
* zoninstraling (GTI)
* regelbare warmteinvoer (UFH)

---

## 2. Modelstructuur

### T_r (Room temperature)

Gemeten luchttemperatuur in de ruimte.

### T_b (Buffer temperature)

Niet gemeten thermische massa:

* betonvloer
* binnenmuren
* constructieve massa

---

## 3. Continue fysische vergelijking

### Buffer

C_b · dT_b/dt =
P_UFH − (T_b − T_r)/R_br + (1 − α) · Q_solar

### Ruimte

C_r · dT_r/dt =
(T_b − T_r)/R_br − (T_r − T_out)/R_ro + α · Q_solar

---

## 4. Interpretatie van warmtestromen

* P_UFH → actieve warmte-injectie via vloerverwarming
* (T_b − T_r)/R_br → interne warmte-uitwisseling
* (T_r − T_out)/R_ro → warmteverlies naar buiten
* Q_solar → zoninstraling via glas

Zon wordt verdeeld:

* α → directe luchtverwarming
* (1 − α) → opslag in thermische massa

---

## 5. Discrete vorm (implementatie)

T_b[k+1] =
T_b[k] + (Δt / C_b) · (P_UFH[k] − (T_b[k] − T_r[k])/R_br + (1 − α) · Q_solar[k])

T_r[k+1] =
T_r[k] + (Δt / C_r) · ((T_b[k] − T_r[k])/R_br − (T_r[k] − T_out[k])/R_ro + α · Q_solar[k])

---

## 6. Zoninstraling (Open-Meteo GTI)

Q_solar [kW] = A_glass · GTI · η / 1000

---

## 7. Variabelen

### States

* T_r [°C]
* T_b [°C]

### Inputs

* P_UFH [kW]
* T_out [°C]
* Q_solar [kW]

---

## 8. Parameters

* C_r [kWh/K]
* C_b [kWh/K]
* R_br [K/kW]
* R_ro [K/kW]
* α [-]
* η [-]
* A_glass [m²]

---

## 9. Constante

* Δt [uur]

---

## 10. Eenhedenconsistentie

kW, kWh/K, uur:

T[k+1] = T[k] + (Δt / C) · P

---

## 11. DHW model

T_dhw[k+1] =
T_dhw[k] + (Δt / C_dhw) · P_dhw − (Δt / (C_dhw · R_dhw)) · (T_dhw − T_room)

---

## 12. State-space vorm

x[k+1] = A x[k] + B u[k] + E d[k]

x = [T_r, T_b]
u = P_UFH
d = [T_out, Q_solar]

---

### Matrices

a_br = Δt / (C_r · R_br)
a_ro = Δt / (C_r · R_ro)
b_br = Δt / (C_b · R_br)

A =
[ 1 − a_br − a_ro,   a_br ]
[ b_br,              1 − b_br ]

B =
[ 0 ]
[ Δt / C_b ]

E =
[ a_ro,                 α · Δt / C_r ]
[ 0,        (1 − α) · Δt / C_b ]

---

## 13. Stabiliteit (correcte formulering)

De stabiliteit van het systeem wordt formeel bepaald door:

|λ(A)| < 1

waar λ(A) de eigenwaarden van de systeemmatrix A zijn.

Equivalent:

* alle dynamische modes moeten binnen de eenheidscirkel liggen

In praktijk voor woningparameters:

* koppelingstermen zijn klein
* eigenwaarden liggen ruim binnen stabiel gebied

---

## 14. Interpretatie van het systeem

* UFH laadt eerst de thermische buffer
* buffer voedt ruimte vertraagd
* zon is externe, voorspelbare verstoring
* buitenlucht bepaalt continue warmteverlies

---

## 15. MPC geschiktheid

Model is geschikt omdat:

* lineair tijd-invariant
* fysisch interpreteerbaar
* stabiel onder realistische parameters
* direct bruikbaar in cvxpy

---

## 16. Minimale systeemstructuur

* 2 states: T_r, T_b
* 1 input: P_UFH
* 2 disturbances: T_out, Q_solar

Minimale volledige woning-MPC representatie zonder overmodellering.
