# Home Optimizer: 2-State Thermisch Model (UFH & MPC)

Dit document bevat de volledige wiskundige en logische instructies voor het implementeren van de thermische regeling in het Home Optimizer project. Het model combineert een fysisch model (grey-box) met een Kalman-filter voor schattingen en een Model Predictive Controller (MPC) voor het optimaliseren van het energieverbruik.

> ### ⚠️ Kerneis: Fysische Correctheid
> **Elke vergelijking, matrix en parameter in dit document moet 100% fysisch en wiskundig correct zijn.** Alle eenheden moeten consistent zijn (SI-eenheden met kW en kWh als energiegrootheid). Elke discretisatie moet stabiel zijn voor de gekozen $\Delta t$. Elke kostterm moet aansluiten op de werkelijke fysische grootheid die geoptimaliseerd wordt. Bij twijfel: afleiden vanuit de continue fysica, niet aanpassen op basis van convenientie. Modelaannames (bijv. lineair warmtetransport, gemiddelde zoninstraling) moeten expliciet benoemd zijn.

---

## 1. Doel & Architectuur

Het doel van deze module is het slim aansturen van de vloerverwarming (UFH) door vooruit te kijken (MPC) en rekening te houden met weersverwachtingen, interne warmte en de traagheid van het huis.

**De architectuur (Closed-loop systeem):**
1. **Verstoringen (d):** Buitentemperatuur, zon, en apparaten/mensen.
2. **Sensoren (y):** De thermostaat in de woonkamer meet de luchttemperatuur.
3. **Kalman Filter:** Berekent hoe warm de betonvloer (buffer) is, omdat we daar geen sensor voor hebben.
4. **MPC Optimizer:** Berekent de optimale hoeveelheid warmte ($P_{UFH}$) voor de komende uren om het huis comfortabel te houden tegen minimale energiekosten.

---

## 2. Definities van de Variabelen

### 2.1 States (Toestandsgrootheden)
*   **$T_r$ [°C]:** Gemeten ruimtetemperatuur (lucht).
*   **$T_b$ [°C]:** Niet-gemeten temperatuur van de thermische massa (de betonvloer/buffer).

### 2.2 Inputs & Verstoringen
*   **$P_{UFH}$ [kW]:** Het vermogen dat de vloerverwarming in de vloer pompt (Actuator).
*   **$T_{out}$ [°C]:** Buitentemperatuur (Verstoring).
*   **$Q_{solar}$ [kW]:** Zoninstraling door de ramen (Verstoring).
*   **$Q_{int}$ [kW]:** Interne warmte van mensen en apparaten, typisch 0.2–0.8 kW per huis (Verstoring).
*   **$p[k]$ [€/kWh]:** Dynamisch elektriciteitstarief op tijdstip $k$ (voor kostenoptimalisatie).

---

## 3. Continue Fysica (Het Thermische Model)

Het huis wordt gemodelleerd als twee 'emmers' met warmte: de vloer (buffer) en de lucht (ruimte).

**Modelaanname:** Warmtetransport tussen zones is lineair (Newtons afkoelwet). Zoninstraling wordt uniform verdeeld over vloer en lucht via factor $\alpha$.

**De Vloer (Buffer):**
$$C_b \cdot \frac{dT_b}{dt} = P_{UFH} - \frac{T_b - T_r}{R_{br}} + (1 - \alpha) \cdot Q_{solar}$$

**De Ruimte (Lucht):**
$$C_r \cdot \frac{dT_r}{dt} = \frac{T_b - T_r}{R_{br}} - \frac{T_r - T_{out}}{R_{ro}} + \alpha \cdot Q_{solar} + Q_{int}$$

> *   $C_b$ en $C_r$ zijn de warmtecapaciteiten [kWh/K].
> *   Warmte stroomt van vloer naar kamer via weerstand $R_{br}$ [K/kW], gedreven door het temperatuurverschil $T_b - T_r$.
> *   Warmte lekt van kamer naar buiten via $R_{ro}$ [K/kW].
> *   De zon warmt voor fractie $\alpha$ direct de lucht op, en $(1-\alpha)$ gaat naar de vloer/meubels.

---

## 4. Discrete Vorm (Voor in de Code)

**Methode:** Forward Euler discretisatie met tijdstap $\Delta t$ [h].

**Stabiliteitseis:** De tijdstap $\Delta t$ moet voldoen aan:
$$\Delta t \ll \min\left(C_r \cdot R_{br},\ C_b \cdot R_{br},\ C_r \cdot R_{ro}\right)$$
Typisch is $\Delta t \in \{0.25, 0.5, 1.0\}$ uur acceptabel voor een woning. Bij twijfel: gebruik Zero-Order Hold (ZOH) discretisatie, die onvoorwaardelijk stabiel is.

$$T_b[k+1] = T_b[k] + \frac{\Delta t}{C_b} \cdot \left( P_{UFH}[k] - \frac{T_b[k] - T_r[k]}{R_{br}} + (1 - \alpha) \cdot Q_{solar}[k] \right)$$

$$T_r[k+1] = T_r[k] + \frac{\Delta t}{C_r} \cdot \left( \frac{T_b[k] - T_r[k]}{R_{br}} - \frac{T_r[k] - T_{out}[k]}{R_{ro}} + \alpha \cdot Q_{solar}[k] + Q_{int}[k] \right)$$

**Zoninstraling berekenen:**
$$Q_{solar} = \frac{A_{glass} \cdot GTI \cdot \eta}{1000}$$
*(GTI in W/m², $A_{glass}$ in m², resultaat in kW.)*

---

## 5. State-Space Representatie (Matrix Vorm)

$$x[k+1] = A x[k] + B u[k] + E d[k]$$

*   $x = [T_r, T_b]^T$
*   $u = P_{UFH}$
*   $d = [T_{out}, Q_{solar}, Q_{int}]^T$

**Hulpgrootheden (eenheidloos, afleidbaar uit parameters):**

$$a_{br} = \frac{\Delta t}{C_r \cdot R_{br}}, \quad a_{ro} = \frac{\Delta t}{C_r \cdot R_{ro}}, \quad b_{br} = \frac{\Delta t}{C_b \cdot R_{br}}$$

**Matrices:**

$$A = \begin{bmatrix} 1 - a_{br} - a_{ro} & a_{br} \\ b_{br} & 1 - b_{br} \end{bmatrix}$$

$$B = \begin{bmatrix} 0 \\ \frac{\Delta t}{C_b} \end{bmatrix}$$

$$E = \begin{bmatrix} a_{ro} & \alpha \cdot \frac{\Delta t}{C_r} & \frac{\Delta t}{C_r} \\ 0 & (1-\alpha) \cdot \frac{\Delta t}{C_b} & 0 \end{bmatrix}$$

> Het model is **observeerbaar** (vloertemperatuur is afleidbaar uit kamertemperatuur over tijd) en **regelbaar** (de warmtepomp kan de vloertemperatuur sturen). Observeerbaarheid moet gecontroleerd worden na parametrisatie via de observeerbaarheidsmatrix $\mathcal{O} = [C^T, (CA)^T]^T$ met rang 2.

---

## 6. Kalman Filter (Temperatuur Schatter)

**Predictie:**
$$\hat{x}^-[k] = A \hat{x}[k-1] + B u[k-1] + E d[k-1]$$
$$P^-[k] = A P[k-1] A^T + Q_n$$

**Update:**
$$K[k] = P^-[k] C^T \cdot (C P^-[k] C^T + R_n)^{-1}$$
$$\hat{x}[k] = \hat{x}^-[k] + K[k] \cdot (y[k] - C \hat{x}^-[k])$$
$$P[k] = (I - K[k] C) P^-[k]$$

Met meetmatrix $C = [1, 0]$ (alleen $T_r$ wordt gemeten).

---

## 7. Model Predictive Control (De Optimizer)

De MPC kijkt $N$ stappen in de toekomst en minimaliseert energiekosten terwijl comfort gewaarborgd blijft.

**Kostfunctie:**
$$J = \sum_{k=0}^{N-1} \left[ Q_c \cdot (T_r[k] - T_{ref}[k])^2 + p[k] \cdot P_{UFH}[k] \cdot \Delta t + R_c \cdot P_{UFH}[k]^2 \right] + Q_N \cdot (T_r[N] - T_{ref}[N])^2$$

> **Toelichting kostfunctie:**
> - $Q_c \cdot (T_r - T_{ref})^2$: comfortafwijking (kwadratisch, convex).
> - $p[k] \cdot P_{UFH}[k] \cdot \Delta t$: **werkelijke energiekosten** [€] op tijdstip $k$ — lineair met het dynamische tarief en het energieverbruik in kWh. Dit is de primaire optimalisatieterm.
> - $R_c \cdot P_{UFH}[k]^2$: regularisatieterm die abrupte vermogenspieken dempt (convexiteit, geen fysische kostenterm).
> - $Q_N$: eindgewicht voor stabiele afsluiting van de horizon.

**Constraints:**
1. $0 \leq P_{UFH}[k] \leq P_{max}$ — verwarming werkt niet negatief.
2. $T_{min} \leq T_r[k] \leq T_{max}$ — comfortbereik.
3. $|P_{UFH}[k] - P_{UFH}[k-1]| \leq \Delta P_{max}$ — ramp-rate beperking warmtepomp.

---

## 8. Parameter Woordenboek

### Systeem & Huis Parameters
| Parameter | Eenheid | Betekenis |
|---|---|---|
| $C_r$ | kWh/K | Warmtecapaciteit lucht + meubels. |
| $C_b$ | kWh/K | Warmtecapaciteit vloer/beton. |
| $R_{br}$ | K/kW | Thermische weerstand vloer → lucht. |
| $R_{ro}$ | K/kW | Thermische weerstand huis → buiten. |
| $\alpha$ | — | Fractie zonlicht direct naar lucht (0–1). |
| $\eta$ | — | Glasdoorlaatfactor (transmissie). |
| $A_{glass}$ | m² | Raamoppervlak op de zonkant. |

### MPC Instellingen
| Parameter | Eenheid | Betekenis |
|---|---|---|
| $N$ | — | Vooruitkijk-horizon (bijv. 24 of 48 stappen). |
| $Q_c$ | K⁻² | Comfortgewicht. |
| $R_c$ | kW⁻² | Regularisatiegewicht (demping pieken). |
| $Q_N$ | K⁻² | Eindgewicht horizon. |
| $p[k]$ | €/kWh | Dynamisch elektriciteitstarief op tijdstip $k$. |
| $P_{max}$ | kW | Maximaal vermogen warmtepomp. |
| $\Delta P_{max}$ | kW/stap | Maximale ramp-rate. |

### Kalman Filter Variabelen
| Parameter | Eenheid | Betekenis |
|---|---|---|
| $Q_n$ | K² | Procesruis covariantie (modelonzekerheid). |
| $R_n$ | K² | Meetruis covariantie (sensorafwijking). |
| $C$ | Matrix | `[1, 0]` — alleen $T_r$ wordt gemeten. |