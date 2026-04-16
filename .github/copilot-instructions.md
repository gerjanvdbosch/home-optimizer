# Home Optimizer: Thermisch Model (UFH & DHW & MPC)

Dit document bevat de volledige wiskundige en logische instructies voor het implementeren van de thermische regeling in het Home Optimizer project. Het model combineert een fysisch model (grey-box) met een Kalman-filter voor schattingen en een Model Predictive Controller (MPC) voor het optimaliseren van het energieverbruik van zowel ruimteverwarming (UFH) als tapwater (DHW).

> ### ⚠️ Kerneis: Fysische Correctheid
> **Elke vergelijking, matrix en parameter in dit document moet 100% fysisch en wiskundig correct zijn.** Alle eenheden zijn consistent: vermogen in **kW**, energie in **kWh**, temperatuur in **°C**, volume in **m³**, tijd in **h**. Elke discretisatie moet stabiel zijn voor de gekozen $\Delta t$. Elke kostterm moet aansluiten op de werkelijke fysische grootheid die geoptimaliseerd wordt. Bij twijfel: afleiden vanuit de continue fysica, niet aanpassen op basis van convenientie. Alle modelaannames zijn expliciet benoemd.

> ### 🚫 Anti-Patroon: Geen Magic Numbers
> **Het hardcoden van numerieke waarden in de berekeningen of logica is streng verboden.** Elke fysieke constante, tuning-parameter, temperatuurgrens of tijdstap moet via een configuratiebestand (bijv. JSON/YAML) of parameter-object in de code worden geïnjecteerd. Zelfs universele constanten (zoals $\lambda = 1.1628$) of legionella-eisen (60°C) moeten benoemde variabelen zijn. In de formules en MPC-constraints staan uitsluitend referentienamen, géén losse getallen (met uitzondering van wiskundige operatoren zoals `0` of `1`).

> ### ♻️ Architectuureis: DRY (Don't Repeat Yourself) / Geen Dubbele Code
> **Het dupliceren van code of logica is ten strengste verboden.** Hoewel UFH en DHW fysisch andere systemen zijn, delen ze exact dezelfde wiskundige fundamenten (State-Space representaties, Kalman Filters, Forward Euler discretisatie en MPC).
> Schrijf generieke, herbruikbare functies of klassen (bijv. een algemene `KalmanFilter`-klasse die $A, B, E, C, Q, R$ matrices accepteert en de Joseph-vorm update uitvoert, en een afgeleide `ExtendedKalmanFilter`-klasse die een Jacobiaan-callback accepteert) in plaats van aparte hardcoded implementaties. Gebruik object-oriëntatie, compositie of afgeleide klassen om specifieke eigenschappen (zoals de tijdsvariabele $A$-matrix voor DHW of de EKF-linearisatie) af te handelen zonder de kern-wiskunde te kopiëren.

> ### 🚀 Ontwerpeis: Geen Backwards Compatibility
> **De code hoeft nergens backwards compatible te zijn met eerdere versies.** Focus uitsluitend op het bouwen van de meest robuuste, wiskundig zuivere en efficiënte architectuur zoals beschreven in dit document. Oude API's, legacy datastructuren, of eerdere (suboptimale) implementaties mogen zonder pardon worden gebroken, overschreven of verwijderd. Sleep geen technische schuld of workarounds mee om oudere systemen in de lucht te houden: de fysica en de nieuwe DRY-architectuur krijgen absolute voorrang.

> ### 🛡️ Code-eis: Fail-Fast & Harde Assertions
> **De software moet de fysische theorie actief bewaken.**
> 1. **Geen verborgen fallbacks:** Ontbreekt er een variabele of parameter? Verzin géén default waarde (zoals `T = 20.0` of `V = 0`), maar laat de code direct crashen met een expliciete foutmelding (Fail-Fast).
> 2. **Assertions:** Gebruik validatie in de code om fysische onmogelijkheden (zoals negatieve warmtecapaciteit $C$, negatief volume $\dot{V}$, of temperatuur onder het absolute nulpunt) te blokkeren voordat ze de wiskundige solver bereiken.
> 3. **EKF clamp:** Na elke EKF-updatestap moet $\hat{\dot{V}}_{tap}[k]$ worden geclampt op $\max(0,\ \hat{\dot{V}}_{tap}[k])$. Een negatief debiet is fysisch onmogelijk en mag de solver nooit bereiken.

> ### 🧹 Kwaliteitseis: CI/CD Pipeline Compliance
> **Alle code moet 100% foutloos door de gedefinieerde GitHub Actions pipeline (`./.github/workflows/python-lint.yml`) komen.** Dit betekent onvoorwaardelijke naleving van strakke code-standaarden. De code wordt automatisch getoetst op syntax (`pyflakes`), best-practices en linting (`ruff`), strikte formattering (`black --check`), en het slagen van alle unit tests (`pytest`). Code die de pipeline breekt, is per definitie ongeldig.

> ### 📝 Documentatie-eis: Transparantie & Traceerbaarheid
> **Alle code (klassen, functies, variabelen) moet voorzien zijn van uitputtende, gestructureerde documentatie en type-hints.**
> 1. **Docstrings & Eenheden:** Gebruik een vaste docstring-standaard (bijv. Google of NumPy style) voor élke functie. Benoem in de docstring niet alleen de argumenten, maar expliciet de **fysische eenheid** en verwachte datatypes (via Type Hints in de code).
> 2. **Traceerbaarheid naar theorie:** Koppel wiskundige bewerkingen in de code direct terug aan dit document. Gebruik inline comments zoals: `# Implementeert EKF Predictie (Jacobiaan) uit sectie 12`.
> 3. **Het 'Waarom', niet het 'Wat':** Code vertelt *wat* de computer doet; comments vertellen *waarom*. Documenteer expliciet matrixdimensies, gemaakte aannames bij linearisatie, en afhandeling van edge-cases (bijv. deling door nul bij $R_{strat}$, of negatieve EKF-schatting van $\dot{V}_{tap}$).

---

## Inhoudsopgave

**Deel A — UFH (Ruimteverwarming)**
1. Doel & Architectuur
2. Variabelendefinities UFH
3. Continue Fysica UFH
4. Discrete Vorm UFH
5. State-Space Representatie UFH
6. Kalman Filter UFH

**Deel B — DHW (Tapwater)**
7. Modelkeuze & Aannames DHW
8. Variabelendefinities DHW
9. Continue Fysica DHW
10. Discrete Vorm DHW
11. State-Space Representatie DHW (MPC)
12. Extended Kalman Filter DHW (EKF met augmented state)

**Deel C — Gecombineerd Systeem & MPC**
13. Gecombineerd State-Vector
14. MPC Kostfunctie & Constraints
15. Parameter Woordenboek
16. Software Stack & Validatie-eisen

---

# Deel A — UFH (Ruimteverwarming)

---

## 1. Doel & Architectuur

Het doel van deze module is het slim aansturen van de vloerverwarming (UFH) door vooruit te kijken (MPC) en rekening te houden met weersverwachtingen, interne warmte en de traagheid van het huis.

**De architectuur (Closed-loop systeem):**
1. **Verstoringen (d):** Buitentemperatuur, zon, en apparaten/mensen.
2. **Sensoren (y):** De thermostaat in de woonkamer meet de luchttemperatuur.
3. **Kalman Filter:** Berekent hoe warm de betonvloer (buffer) is, omdat we daar geen sensor voor hebben.
4. **MPC Optimizer:** Berekent de optimale hoeveelheid warmte ($P_{UFH}$) voor de komende uren om het huis comfortabel te houden tegen minimale energiekosten.

---

## 2. Variabelendefinities UFH

### 2.1 States

| Variabele | Eenheid | Betekenis |
|---|---|---|
| $T_r$ | °C | Gemeten ruimtetemperatuur (lucht) |
| $T_b$ | °C | Niet-gemeten temperatuur thermische massa (betonvloer/buffer) |

### 2.2 Inputs & Verstoringen

| Variabele | Eenheid | Betekenis |
|---|---|---|
| $P_{UFH}$ | kW | Thermisch vermogen vloerverwarming (Actuator) |
| $T_{out}$ | °C | Buitentemperatuur (Verstoring) |
| $Q_{solar}$ | kW | Zoninstraling door de ramen (Verstoring) |
| $Q_{int}$ | kW | Interne warmte mensen + apparaten, typisch 0.2–0.8 kW (Verstoring) |
| $p[k]$ | €/kWh | Dynamisch elektriciteitstarief op tijdstip $k$ |

---

## 3. Continue Fysica UFH

Het huis wordt gemodelleerd als twee thermische knooppunten: de vloer (buffer) en de lucht (ruimte).

**Modelaanname:** Warmtetransport tussen zones is lineair (Newtons afkoelwet). Zoninstraling wordt uniform verdeeld over vloer en lucht via factor $\alpha$.

**De Vloer (Buffer):**
$$C_b \cdot \frac{dT_b}{dt} = P_{UFH} - \frac{T_b - T_r}{R_{br}} + (1 - \alpha) \cdot Q_{solar}$$

**De Ruimte (Lucht):**
$$C_r \cdot \frac{dT_r}{dt} = \frac{T_b - T_r}{R_{br}} - \frac{T_r - T_{out}}{R_{ro}} + \alpha \cdot Q_{solar} + Q_{int}$$

> - $C_b$ en $C_r$: warmtecapaciteiten [kWh/K].
> - Warmte stroomt van vloer naar kamer via $R_{br}$ [K/kW], gedreven door $T_b - T_r$.
> - Warmte lekt van kamer naar buiten via $R_{ro}$ [K/kW].
> - De zon warmt voor fractie $\alpha$ direct de lucht op; $(1-\alpha)$ gaat naar de vloer/meubels.

---

## 4. Discrete Vorm UFH

**Methode:** Forward Euler discretisatie met tijdstap $\Delta t$ [h].

**Stabiliteitseis:**
$$\Delta t \ll \min\left(C_r \cdot R_{br},\ C_b \cdot R_{br},\ C_r \cdot R_{ro}\right)$$

Typisch is $\Delta t \in \{0.25, 0.5, 1.0\}$ uur acceptabel voor een woning. Bij twijfel: gebruik Zero-Order Hold (ZOH) discretisatie, die onvoorwaardelijk stabiel is.

$$T_b[k+1] = T_b[k] + \frac{\Delta t}{C_b} \cdot \left( P_{UFH}[k] - \frac{T_b[k] - T_r[k]}{R_{br}} + (1 - \alpha) \cdot Q_{solar}[k] \right)$$

$$T_r[k+1] = T_r[k] + \frac{\Delta t}{C_r} \cdot \left( \frac{T_b[k] - T_r[k]}{R_{br}} - \frac{T_r[k] - T_{out}[k]}{R_{ro}} + \alpha \cdot Q_{solar}[k] + Q_{int}[k] \right)$$

**Zoninstraling berekenen:**
$$Q_{solar}[k] = \frac{A_{glass} \cdot GTI[k] \cdot \eta}{1000} \quad \text{(GTI in W/m², resultaat in kW)}$$

---

## 5. State-Space Representatie UFH

$$x_{UFH}[k+1] = A_{UFH}\, x_{UFH}[k] + B_{UFH}\, u_{UFH}[k] + E_{UFH}\, d_{UFH}[k]$$

- $x_{UFH} = [T_r,\ T_b]^T$
- $u_{UFH} = P_{UFH}$
- $d_{UFH} = [T_{out},\ Q_{solar},\ Q_{int}]^T$

**Hulpgrootheden:**
$$a_{br} = \frac{\Delta t}{C_r \cdot R_{br}}, \qquad a_{ro} = \frac{\Delta t}{C_r \cdot R_{ro}}, \qquad b_{br} = \frac{\Delta t}{C_b \cdot R_{br}}$$

**Matrices:**
$$A_{UFH} = \begin{bmatrix} 1 - a_{br} - a_{ro} & a_{br} \\ b_{br} & 1 - b_{br} \end{bmatrix}$$

$$B_{UFH} = \begin{bmatrix} 0 \\ \dfrac{\Delta t}{C_b} \end{bmatrix}$$

$$E_{UFH} = \begin{bmatrix} a_{ro} & \alpha \cdot \dfrac{\Delta t}{C_r} & \dfrac{\Delta t}{C_r} \\[6pt] 0 & (1-\alpha) \cdot \dfrac{\Delta t}{C_b} & 0 \end{bmatrix}$$

> Het systeem is **observeerbaar** en **regelbaar**. Controleer observeerbaarheid na parametrisatie via $\mathcal{O} = [C_{UFH}^T,\ (C_{UFH} A_{UFH})^T]^T$ met rang 2, waarbij $C_{UFH} = [1,\ 0]$.

---

## 6. Kalman Filter UFH

**Predictie:**
$$\hat{x}^-_{UFH}[k] = A_{UFH}\,\hat{x}_{UFH}[k-1] + B_{UFH}\,u_{UFH}[k-1] + E_{UFH}\,d_{UFH}[k-1]$$
$$P^-_{UFH}[k] = A_{UFH}\,P_{UFH}[k-1]\,A_{UFH}^T + Q_{n,UFH}$$

**Update (Joseph-vorm voor numerieke stabiliteit):**
$$K_{UFH}[k] = P^-_{UFH}[k]\,C_{UFH}^T \cdot \left(C_{UFH}\,P^-_{UFH}[k]\,C_{UFH}^T + R_{n,UFH}\right)^{-1}$$
$$\hat{x}_{UFH}[k] = \hat{x}^-_{UFH}[k] + K_{UFH}[k] \cdot \left(y_{UFH}[k] - C_{UFH}\,\hat{x}^-_{UFH}[k]\right)$$
$$P_{UFH}[k] = \left(I - K_{UFH}C_{UFH}\right)P^-_{UFH}[k]\left(I - K_{UFH}C_{UFH}\right)^T + R_{n,UFH}\,K_{UFH}K_{UFH}^T$$

Met meetmatrix $C_{UFH} = [1,\ 0]$ (alleen $T_r$ wordt gemeten).

---

# Deel B — DHW (Tapwater)

---

## 7. Modelkeuze & Aannames DHW

Het DHW-systeem wordt gemodelleerd als een **2-node stratificatietank** (boven- en onderlaag). Dit is de minimale representatie die stratificatie, koude inlaat en warme uitlaat fysisch correct vangt. Het DHW-subsysteem is thermisch ontkoppeld van de ruimte; de koppeling met UFH loopt uitsluitend via gedeeld elektrisch vermogen van de warmtepomp.

**Expliciete modelaannames:**

| # | Aanname | Consequentie |
|---|---|---|
| A1 | Elke laag is perfect gemengd (wel-roerd) | Binnen een laag is $T$ uniform |
| A2 | Warmtetransport tussen lagen alleen via interne conductie/menging | Geen convectief transport anders dan via tapstroom |
| A3 | Tapstroom is plug-flow: warm water verlaat bovenaan, koud water komt onderaan | Tapterm is bilineair (state × state); afgehandeld via EKF-linearisatie |
| A4 | Standby-verlies evenredig met $(T_{laag} - T_{amb})$ (Newton) | Verlies aanwezig zolang $T > T_{amb}$ |
| A5 | Warmtepomp-warmtewisselaar zit **onderin** de tank (industriestandaard) | $P_{dhw}$ gaat volledig naar de onderlaag |
| A6 | $T_{dhw}$ (gemiddelde tanktemperatuur) is een **afgeleid** signaal, geen extra state | Geen overspecificatie; $T_{dhw} = (C_{top}T_{top} + C_{bot}T_{bot})/(C_{top}+C_{bot})$ |
| A7 | **Er is geen flowmeter aanwezig.** $\dot{V}_{tap}[k]$ is een onbekende grootheid die online wordt geschat | $\dot{V}_{tap}$ wordt behandeld als **augmented state** in de EKF (zie §12). Beide temperatuursensoren ($T_{top}$ en $T_{bot}$) zijn beschikbaar als meting en maken de augmented state observeerbaar |

---

## 8. Variabelendefinities DHW

### 8.1 States

| Variabele | Eenheid | Herkomst | Betekenis |
|---|---|---|---|
| $T_{top}$ | °C | Sensor | Temperatuur bovenlaag (tapwater uitgang) |
| $T_{bot}$ | °C | Sensor | Temperatuur onderlaag (koude instroombuffer, WP-kant) |
| $\dot{V}_{tap}$ | m³/h | EKF (geschat) | Tapwaterdebiet — **geen flowmeter**, geschat als augmented state |

> **Architectuurscheiding:** De EKF werkt intern met de 3-dimensionale augmented state $x_{dhw,aug} = [T_{top},\ T_{bot},\ \dot{V}_{tap}]^T$. De MPC werkt met de 2-dimensionale state $x_{dhw} = [T_{top},\ T_{bot}]^T$, waarbij $\hat{\dot{V}}_{tap}[k]$ na de EKF-updatestap als **bekende verstoring** aan de MPC wordt doorgegeven. Zie §11 en §12 voor de respectievelijke formuleringen.

**Afgeleid (geen state):**
$$T_{dhw}[k] = \frac{C_{top}\,T_{top}[k] + C_{bot}\,T_{bot}[k]}{C_{top} + C_{bot}}$$

### 8.2 Actuator

| Variabele | Eenheid | Betekenis |
|---|---|---|
| $P_{dhw}$ | kW | Thermisch vermogen naar de onderlaag (warmtepompuitgang) |

### 8.3 Verstoringen (voor MPC, na EKF)

| Variabele | Eenheid | Herkomst | Betekenis |
|---|---|---|---|
| $\hat{\dot{V}}_{tap}[k]$ | m³/h | EKF-uitvoer (geclampt $\geq 0$) | Geschat tapwaterdebiet, doorgegeven aan MPC als bekende LTV-parameter |
| $T_{mains}[k]$ | °C | Extern (seizoensmodel of meting) | Temperatuur inkomend koud leidingwater |
| $T_{amb}[k]$ | °C | Sensor of schatting | Omgevingstemperatuur rond de boiler (bijv. meterkast) |

### 8.4 Eenheidsconversie (éénmalig vastgelegd)

$$\lambda = \rho \cdot c_p = 1000\,\frac{\text{kg}}{\text{m}^3} \times \frac{4186\,\text{J}}{\text{kg}\cdot\text{K}} \times \frac{1\,\text{kWh}}{3{,}600{,}000\,\text{J}} = 1.1628\,\frac{\text{kWh}}{\text{m}^3 \cdot \text{K}}$$

Gebruik **uitsluitend** $\lambda$ in alle formules. Gebruik $c_p$ in J/kgK nergens direct.

---

## 9. Continue Fysica DHW

### 9.1 Tapwater-energiestroom

Warm water verlaat de tank **bovenaan** met temperatuur $T_{top}$; koud leidingwater komt **onderaan** binnen met $T_{mains}$. De totale netto warmteonttrekking aan het systeem is:

$$\dot{Q}_{tap} = \lambda \cdot \dot{V}_{tap} \cdot (T_{top} - T_{mains}) \quad \text{[kW]}$$

Intern splitst deze stroom zich fysisch correct over beide lagen:

| Laag | Term | Betekenis |
|---|---|---|
| Bovenlaag verliest | $-\lambda \cdot \dot{V}_{tap} \cdot T_{top}$ | Heet water verlaat de tank |
| Onderlaag wint | $+\lambda \cdot \dot{V}_{tap} \cdot T_{mains}$ | Koud leidingwater stroomt in |

> Als $\dot{Q}_{tap}$ als één term in de bovenlaag staat, ontbreekt de mains-bijdrage in de onderlaag en sluit de energiebalans **niet**. De splitsing hierboven is de enige fysisch correcte aanpak.

### 9.2 Standby-verliezen

$$\dot{Q}_{loss,top} = \frac{T_{top} - T_{amb}}{R_{loss}}, \qquad \dot{Q}_{loss,bot} = \frac{T_{bot} - T_{amb}}{R_{loss}} \quad \text{[kW]}$$

### 9.3 Interne conductie (stratificatie)

$$\dot{Q}_{strat} = \frac{T_{top} - T_{bot}}{R_{strat}} \quad \text{[kW]}$$

Hogere $R_{strat}$ → minder menging → betere stratificatie.

### 9.4 Continue differentiaalvergelijkingen

**Bovenlaag:**
$$C_{top}\frac{dT_{top}}{dt} = -\dot{Q}_{strat} - \lambda\dot{V}_{tap}\,T_{top} - \dot{Q}_{loss,top}$$

**Onderlaag:**
$$C_{bot}\frac{dT_{bot}}{dt} = +\dot{Q}_{strat} + P_{dhw} + \lambda\dot{V}_{tap}\,T_{mains} - \dot{Q}_{loss,bot}$$

### 9.5 Verificatie energiebalans

Som van beide knooppunten (gewogen):

$$\frac{d}{dt}(C_{top}T_{top} + C_{bot}T_{bot}) = P_{dhw} \underbrace{-\lambda\dot{V}_{tap}(T_{top}-T_{mains})}_{=-\dot{Q}_{tap}} \underbrace{-\frac{T_{top}-T_{amb}}{R_{loss}} - \frac{T_{bot}-T_{amb}}{R_{loss}}}_{=-\dot{Q}_{loss}} \quad \checkmark$$

De $\dot{Q}_{strat}$-term valt weg (intern transport, netto nul). Dit is de eerste wet van de thermodynamica voor het gehele tankvolume.

---

## 10. Discrete Vorm DHW

### 10.1 Bilineariteit & Behandeling

De termen $\lambda\dot{V}_{tap}[k]\cdot T_{top}[k]$ en $\lambda\dot{V}_{tap}[k]\cdot T_{mains}[k]$ zijn bilineair. Omdat $\dot{V}_{tap}[k]$ niet direct gemeten wordt, kan de LTV-aanname (§10 origineel) niet rechtstreeks worden toegepast.

**Oplossing (tweestaps architectuur):**

1. **EKF-stap** (§12): Schat $\hat{\dot{V}}_{tap}[k]$ via linearisatie rondom de vorige schatting. De EKF werkt met de 3-dimensionale augmented state en beide temperatuurmetingen.
2. **MPC-stap** (§11 en §14): Behandel $\hat{\dot{V}}_{tap}[k]$ als **bekende LTV-parameter** voor de horizon. De MPC ontvangt een array $\hat{\dot{V}}_{tap}[k],\ k=0,\ldots,N-1$ en construeert daarmee de tijdsvariabele matrices $A_{dhw}[k]$ en $E_{dhw}[k]$ precies zoals in het originele LTV-schema.

Definieer (zoals voorheen, nu gebaseerd op EKF-schatting):
$$a_{tap}[k] = \frac{\Delta t}{C_{top}} \cdot \lambda \cdot \hat{\dot{V}}_{tap}[k], \qquad b_{tap}[k] = \frac{\Delta t}{C_{bot}} \cdot \lambda \cdot \hat{\dot{V}}_{tap}[k]$$

### 10.2 Stabiliteitseis

$$\Delta t \ll \min\!\left(C_{top}\cdot R_{strat},\ C_{bot}\cdot R_{strat},\ C_{top}\cdot R_{loss}\right)$$

### 10.3 Vergelijkingen

**Bovenlaag:**
$$\boxed{T_{top}[k+1] = T_{top}[k] + \frac{\Delta t}{C_{top}}\!\left( -\frac{T_{top}[k]-T_{bot}[k]}{R_{strat}} - \lambda\hat{\dot{V}}_{tap}[k]\,T_{top}[k] - \frac{T_{top}[k]-T_{amb}[k]}{R_{loss}} \right)}$$

**Onderlaag:**
$$\boxed{T_{bot}[k+1] = T_{bot}[k] + \frac{\Delta t}{C_{bot}}\!\left( \frac{T_{top}[k]-T_{bot}[k]}{R_{strat}} + P_{dhw}[k] + \lambda\hat{\dot{V}}_{tap}[k]\,T_{mains}[k] - \frac{T_{bot}[k]-T_{amb}[k]}{R_{loss}} \right)}$$

---

## 11. State-Space Representatie DHW (voor MPC)

> **Scope:** Deze sectie beschrijft de 2-dimensionale state-space die de MPC gebruikt. $\hat{\dot{V}}_{tap}[k]$ is op dit punt al beschikbaar als uitvoer van de EKF (zie §12) en wordt als bekende LTV-parameter behandeld.

$$x_{dhw}[k+1] = A_{dhw}[k]\,x_{dhw}[k] + B_{dhw}\,u_{dhw}[k] + E_{dhw}[k]\,d_{dhw}[k]$$

- $x_{dhw} = [T_{top},\ T_{bot}]^T$
- $u_{dhw} = P_{dhw}$
- $d_{dhw} = [T_{amb},\ T_{mains}]^T$

**Hulpgrootheden (constant):**
$$a_{strat} = \frac{\Delta t}{C_{top}\cdot R_{strat}}, \quad b_{strat} = \frac{\Delta t}{C_{bot}\cdot R_{strat}}, \quad a_{loss} = \frac{\Delta t}{C_{top}\cdot R_{loss}}, \quad b_{loss} = \frac{\Delta t}{C_{bot}\cdot R_{loss}}$$

**State-transitiematrix (tijdsvariabel via $\hat{\dot{V}}_{tap}[k]$ uit EKF):**
$$A_{dhw}[k] = \begin{bmatrix} 1 - a_{strat} - a_{loss} - a_{tap}[k] & a_{strat} \\ b_{strat} & 1 - b_{strat} - b_{loss} \end{bmatrix}$$

**Inputmatrix (constant):**
$$B_{dhw} = \begin{bmatrix} 0 \\ \dfrac{\Delta t}{C_{bot}} \end{bmatrix}$$

> $P_{dhw}$ gaat naar de onderlaag (aanname A5). Bij een elektrisch verwarmingselement bovenaan: vervang door $[\Delta t/C_{top},\ 0]^T$ en documenteer dit als configuratiekeuze.

**Verstoringenmatrix (tijdsvariabel via $\hat{\dot{V}}_{tap}[k]$):**
$$E_{dhw}[k] = \begin{bmatrix} a_{loss} & 0 \\ b_{loss} & b_{tap}[k] \end{bmatrix}$$

> Kolom 1 = $T_{amb}$ (standby-verlies beide lagen). Kolom 2 = $T_{mains}$ (koud instromend water, alleen onderlaag).

**Observeerbaarheid (MPC-model, 2 states, $C_{obs,dhw} = [1,\ 0]$):**
$$\mathcal{O}_{dhw} = \begin{bmatrix} 1 & 0 \\ 1 - a_{strat} - a_{loss} - a_{tap}[k] & a_{strat} \end{bmatrix}$$

$\text{rang}(\mathcal{O}_{dhw}) = 2 \iff a_{strat} \neq 0$, wat altijd geldt voor een reële tank met eindige $R_{strat}$.

---

## 12. Extended Kalman Filter DHW (EKF met augmented state)

### 12.1 Motivatie & Architectuur

Omdat er geen flowmeter aanwezig is (aanname A7), wordt $\dot{V}_{tap}$ geschat als **augmented state**. De combinatie van $T_{top}$-sensor én $T_{bot}$-sensor levert voldoende informatie om het 3-dimensionale systeem observeerbaar te maken.

De tapsysteemfysica is **nonlineair** in de augmented state (product $\dot{V}_{tap} \times T_{top}$ en $\dot{V}_{tap} \times T_{mains}$). Daarom wordt een **Extended Kalman Filter (EKF)** toegepast, dat bij elke tijdstap lineariseert rondom de vorige schatting.

**Architectuureis DRY:** De EKF is een afgeleide klasse van de generieke `KalmanFilter`-klasse, waarbij de vaste matrix $A$ vervangen wordt door een Jacobiaan-callback $F(x, d)$, en de propagatiestap vervangen wordt door de nonlineaire functie $f(x, u, d)$.

### 12.2 Augmented State & Meetmodel

**Augmented state (intern in EKF):**
$$x_{aug}[k] = \begin{bmatrix} T_{top}[k] \\ T_{bot}[k] \\ \dot{V}_{tap}[k] \end{bmatrix}$$

**Random walk model voor $\dot{V}_{tap}$:**
$$\dot{V}_{tap}[k+1] = \dot{V}_{tap}[k] + w_{\dot{V}}[k], \qquad w_{\dot{V}} \sim \mathcal{N}(0,\ Q_{n,\dot{V}})$$

> De procesruis $Q_{n,\dot{V}}$ bepaalt hoe snel de EKF kan reageren op veranderende tapgebeurtenissen. Hogere waarde → sneller aanpassen, meer ruis in schatting. Lagere waarde → trage respons, maar gladder signaal. Dit is een tuning-parameter in de configuratie.

**Meetmodel (beide sensoren beschikbaar):**
$$y_{dhw}[k] = \begin{bmatrix} T_{top}^{meas}[k] \\ T_{bot}^{meas}[k] \end{bmatrix} = C_{obs,dhw}\, x_{aug}[k] + v[k]$$

$$C_{obs,dhw} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix} \in \mathbb{R}^{2 \times 3}$$

$R_{n,dhw}$ is nu een **2×2 diagonaalmatrix**:
$$R_{n,dhw} = \begin{bmatrix} \sigma^2_{T_{top}} & 0 \\ 0 & \sigma^2_{T_{bot}} \end{bmatrix}$$

### 12.3 Nonlineaire Propagatiefunctie

De volledige nonlineaire propagatie van de augmented state is:

$$f(x_{aug}, u_{dhw}, d_{dhw}) = \begin{bmatrix}
T_{top} + \dfrac{\Delta t}{C_{top}}\!\left( -\dfrac{T_{top}-T_{bot}}{R_{strat}} - \lambda\,\dot{V}_{tap}\,T_{top} - \dfrac{T_{top}-T_{amb}}{R_{loss}} \right) \\[8pt]
T_{bot} + \dfrac{\Delta t}{C_{bot}}\!\left( \dfrac{T_{top}-T_{bot}}{R_{strat}} + P_{dhw} + \lambda\,\dot{V}_{tap}\,T_{mains} - \dfrac{T_{bot}-T_{amb}}{R_{loss}} \right) \\[8pt]
\dot{V}_{tap}
\end{bmatrix}$$

### 12.4 Jacobiaan (Linearisatie rondom schatting)

De Jacobiaan $F[k] = \left.\dfrac{\partial f}{\partial x_{aug}}\right|_{\hat{x}_{aug}[k]}$ wordt bij elke tijdstap geëvalueerd:

$$F[k] = \begin{bmatrix}
1 - a_{strat} - a_{loss} - \hat{a}_{tap}[k] & a_{strat} & -\dfrac{\Delta t}{C_{top}}\,\lambda\,\hat{T}_{top}[k] \\[6pt]
b_{strat} & 1 - b_{strat} - b_{loss} & \dfrac{\Delta t}{C_{bot}}\,\lambda\,T_{mains}[k] \\[6pt]
0 & 0 & 1
\end{bmatrix}$$

Waarbij:
$$\hat{a}_{tap}[k] = \frac{\Delta t}{C_{top}} \cdot \lambda \cdot \hat{\dot{V}}_{tap}[k]$$

> De derde kolom van $F[k]$ is de kern van de EKF: hij drukt uit hoe gevoelig $T_{top}$ en $T_{bot}$ zijn voor een verandering in $\dot{V}_{tap}$. Bij een hoge $\hat{T}_{top}[k]$ (groot temperatuurverschil met koud water) is de schatting van $\dot{V}_{tap}$ beter geconditioneerd. Bij $T_{top} \approx T_{mains}$ — geen tapegebeurtenis — is de gevoeligheid laag en domineert de procesruis; de EKF zal $\dot{V}_{tap} \approx 0$ behouden.

### 12.5 Observeerbaarheid van het augmented systeem

De linearisatie $F[k]$ is rank-3 observeerbaar voor $C_{obs,dhw}$ zolang $a_{strat} \neq 0$ **en** $\hat{T}_{top}[k] \neq T_{mains}[k]$. De observeerbaarheidsmatrix van de linearisatie is:

$$\mathcal{O}_{aug}[k] = \begin{bmatrix} C_{obs,dhw} \\ C_{obs,dhw}\,F[k] \end{bmatrix} \in \mathbb{R}^{4 \times 3}$$

$\text{rang}(\mathcal{O}_{aug}[k]) = 3$ mits $a_{strat} \neq 0$ en $\hat{T}_{top}[k] \neq T_{mains}[k]$.

> **Rand-conditie:** Als de tank volledig is afgekoeld tot $T_{top} \approx T_{mains}$ (bijv. bij een langdurig ongebruikte installatie), is $\dot{V}_{tap}$ niet observeerbaar via de temperatuursensoren. In dat geval geldt: de EKF divergeert niet (de procesruis houdt $P$ begrensd), maar de schatting van $\dot{V}_{tap}$ wordt onzeker. Dit is fysisch correct gedrag — je kunt immers geen tap detecteren als er toch al geen temperatuurverschil is.

### 12.6 EKF Algoritme

**Stap 1 — Predictie:**
$$\hat{x}^-_{aug}[k] = f\!\left(\hat{x}_{aug}[k-1],\ u_{dhw}[k-1],\ d_{dhw}[k-1]\right)$$
$$P^-_{aug}[k] = F[k-1]\,P_{aug}[k-1]\,F[k-1]^T + Q_{n,dhw,aug}$$

Met:
$$Q_{n,dhw,aug} = \begin{bmatrix} Q_{n,dhw} & 0 \\ 0 & Q_{n,\dot{V}} \end{bmatrix} \in \mathbb{R}^{3 \times 3}$$

waarbij $Q_{n,dhw}$ de 2×2 procesruiscovariantie is voor de temperatuurstates en $Q_{n,\dot{V}}$ de scalaire procesruis voor de flowschatting.

**Stap 2 — Kalman Gain:**
$$K_{aug}[k] = P^-_{aug}[k]\,C_{obs,dhw}^T \cdot \left(C_{obs,dhw}\,P^-_{aug}[k]\,C_{obs,dhw}^T + R_{n,dhw}\right)^{-1}$$

**Stap 3 — Update (Joseph-vorm voor numerieke stabiliteit):**
$$\hat{x}_{aug}[k] = \hat{x}^-_{aug}[k] + K_{aug}[k]\cdot\left(y_{dhw}[k] - C_{obs,dhw}\,\hat{x}^-_{aug}[k]\right)$$

$$P_{aug}[k] = \left(I - K_{aug}C_{obs,dhw}\right)P^-_{aug}[k]\left(I - K_{aug}C_{obs,dhw}\right)^T + R_{n,dhw}\,K_{aug}K_{aug}^T$$

**Stap 4 — Fail-Fast clamp (fysische eis):**
$$\hat{\dot{V}}_{tap}[k] \leftarrow \max\!\left(0,\ \hat{x}_{aug}[k]_3\right)$$

> Een negatief debiet is fysisch onmogelijk. De clamp is een harde post-processing stap na elke update — géén workaround maar een formele projectie op de fysisch haalbare verzameling $\dot{V}_{tap} \geq 0$.

**Stap 5 — Doorgeven aan MPC:**
$$\hat{\dot{V}}_{tap}[k] \rightarrow \text{MPC als LTV-parameter voor stap } k \text{ in de horizon}$$

---

# Deel C — Gecombineerd Systeem & MPC

---

## 13. Gecombineerd State-Vector

De twee subsystemen zijn thermisch ontkoppeld. Het gecombineerde state-vector (voor MPC) is:

$$x_{tot}[k] = \begin{bmatrix} T_r \\ T_b \\ T_{top} \\ T_{bot} \end{bmatrix}, \qquad u_{tot}[k] = \begin{bmatrix} P_{UFH} \\ P_{dhw} \end{bmatrix}$$

De gecombineerde state-space heeft blokdiagonale structuur (geen cross-termen):

$$x_{tot}[k+1] = \underbrace{\begin{bmatrix} A_{UFH} & 0 \\ 0 & A_{dhw}[k] \end{bmatrix}}_{A_{tot}[k]} x_{tot}[k] + \underbrace{\begin{bmatrix} B_{UFH} & 0 \\ 0 & B_{dhw} \end{bmatrix}}_{B_{tot}} u_{tot}[k] + \begin{bmatrix} E_{UFH} & 0 \\ 0 & E_{dhw}[k] \end{bmatrix} \begin{bmatrix} d_{UFH}[k] \\ d_{dhw}[k] \end{bmatrix}$$

> De koppeling tussen UFH en DHW loopt **uitsluitend** via de gedeelde warmtepomp-vermogensconstraint (zie §14). De EKF (§12) levert $\hat{\dot{V}}_{tap}[k]$ waarmee $A_{dhw}[k]$ en $E_{dhw}[k]$ voor de MPC-horizon worden geconstrueerd.

---

## 14. MPC Kostfunctie & Constraints

### 14.1 COP & Elektriciteitsverbruik

**Modelaanname & Pre-calculatie Tijdsvariabele COP (Carnot-benadering):**
Om de MPC als een lineair/convex QP-probleem in CVXPY te houden, mag de COP niet afhankelijk zijn van de beslissingsvariabelen (zoals actuele state-temperaturen) *tijdens* de optimalisatie. De arrays $COP_{UFH}[k]$ en $COP_{dhw}[k]$ worden daarom **voorafgaand aan de MPC-stap berekend** (pre-calculatie) op basis van de weersverwachting ($T_{out}[k]$) en de aangenomen aanvoertemperaturen.

De fysica-gebaseerde berekening hiervoor gebruikt de Carnot-efficiëntie:

$$COP[k] = \eta_{Carnot} \cdot \frac{T_{cond}[k]}{T_{cond}[k] - T_{evap}[k]}$$

Waarbij (in Kelvin):
$$T_{cond}[k] = (T_{aanvoer} + \Delta T_{cond}) + 273.15$$
$$T_{evap}[k]  = (T_{out}[k]  - \Delta T_{evap}) + 273.15$$

**Implementatie-eis:**
- Voor **UFH** wordt een weersafhankelijke stooklijn of vaste ontwerptemperatuur gebruikt voor $T_{aanvoer}$ (bijv. 30–35 °C).
- Voor **DHW** wordt de doeltemperatuur van de tank gebruikt voor $T_{aanvoer}$ (bijv. 55 °C of $T_{leg}$).
- De parameters $\eta_{Carnot}$ (typisch 0.4–0.6), $\Delta T_{cond}$ (typisch 2–5 K) en $\Delta T_{evap}$ (typisch 2–5 K) moeten in de configuratie (Pydantic) als benoemde variabelen aanwezig zijn.

**Let op de eenheden:** $P_{UFH}$ en $P_{dhw}$ zijn thermisch [kW]; $P_{UFH,elec}$ en $P_{dhw,elec}$ zijn elektrisch [kW]. De thermische vermogens zijn de beslissingsvariabelen van de MPC (ze sturen de state-dynamica); de elektrische vermogens verschijnen uitsluitend in de kostfunctie en de gedeelde vermogensconstraint.

**Fail-Fast:** Na de Carnot pre-calculatie loopt er een check: forceer $COP > 1.0$ te allen tijde. Valideer dit vóór de matrices naar CVXPY gaan. Een $COP \leq 1.0$ of $COP > COP_{max}$ gooit een exception.

### 14.2 Kostfunctie

$$J = \underbrace{\sum_{k=0}^{N-1}\!\left[ Q_c\,(T_r[k]-T_{ref}[k])^2 + p[k]\cdot \frac{P_{UFH}[k]}{COP_{UFH}[k]}\cdot\Delta t + R_c\cdot P_{UFH}[k]^2 + M \cdot \epsilon_{UFH}[k]^2 \right] + Q_N\,(T_r[N]-T_{ref}[N])^2}_{\text{UFH: comfort + energiekosten + regularisatie + soft-constraint straf}}$$
$$+\ \underbrace{\sum_{k=0}^{N-1} \left[ p[k]\cdot \frac{P_{dhw}[k]}{COP_{dhw}[k]}\cdot\Delta t + M \cdot \epsilon_{dhw}[k]^2 \right]}_{\text{DHW: energiekosten [€] + soft-constraint straf}}$$

**Toelichting per term:**

| Term | Type | Betekenis |
|---|---|---|
| $Q_c\,(T_r-T_{ref})^2$ | Kwadratisch | Comfortafwijking ruimte (strafpunten per K²) |
| $p[k]\cdot \frac{P_{UFH}[k]}{COP_{UFH}[k]}\cdot\Delta t$ | Lineair | Werkelijke UFH-elektriciteitskosten [€] — primaire optimalisatieterm |
| $R_c\cdot P_{UFH}[k]^2$ | Kwadratisch | Regularisatie: dempt pieken, geen fysische kostenterm |
| $Q_N\,(T_r[N]-T_{ref}[N])^2$ | Kwadratisch | Eindgewicht: stabiele afsluiting horizon |
| $p[k]\cdot \frac{P_{dhw}[k]}{COP_{dhw}[k]}\cdot\Delta t$ | Lineair | Werkelijke DHW-elektriciteitskosten [€] — primaire optimalisatieterm |
| $M \cdot \epsilon_{UFH}[k]^2$ | Kwadratisch | Straf op soft-constraint overtreding UFH (hoge $M$) |
| $M \cdot \epsilon_{dhw}[k]^2$ | Kwadratisch | Straf op soft-constraint overtreding DHW (hoge $M$) |

### 14.3 Constraints

**Slack-variabelen (soft constraints):**

De comfort- en tapbeperkingen zijn geïmplementeerd als **soft constraints** via slack-variabelen $\epsilon_{UFH}[k] \geq 0$ en $\epsilon_{dhw}[k] \geq 0$ om QP-onhaalbaarheid te voorkomen (thermische vertraging kan de vloer te koud laten zijn om direct aan een harde eis te voldoen). De straffactor $M$ is een configuratieparameter (bijv. $M \gg Q_c$).

**UFH actuator:**
$$0 \leq P_{UFH}[k] \leq P_{UFH,max}$$

**UFH comfort (soft):**
$$T_{min} - \epsilon_{UFH}[k] \leq T_r[k] \leq T_{max} + \epsilon_{UFH}[k], \qquad \epsilon_{UFH}[k] \geq 0$$

**UFH ramp-rate:**
$$|P_{UFH}[k] - P_{UFH}[k-1]| \leq \Delta P_{UFH,max}$$

**DHW actuator:**
$$0 \leq P_{dhw}[k] \leq P_{dhw,max}$$

**DHW comfort (soft):**
$$T_{top}[k] \geq T_{dhw,min} - \epsilon_{dhw}[k], \qquad \epsilon_{dhw}[k] \geq 0$$

**DHW ramp-rate:**
$$|P_{dhw}[k] - P_{dhw}[k-1]| \leq \Delta P_{dhw,max}$$

**DHW legionella (periodieke harde eis — beheerd door bovenliggende schil):**

$$\exists\, k^* \in \text{elke } n_{leg} \text{ stappen}:\ T_{top}[k^*] \geq T_{leg} \quad \text{gedurende } \geq \left\lceil\tfrac{t_{leg,min}}{\Delta t}\right\rceil \text{ aaneengesloten stappen}$$

> **Architectuureis legionella:** Omdat de MPC-horizon ($N$ stappen) korter is dan de legionella-cyclus ($n_{leg}$ stappen), mag de $T_{leg}$-eis **niet** blindelings als een standaard QP-constraint over de volledige horizon worden ingesteld. De MPC zou de eis dan consequent buiten zijn horizon schuiven tot de deadline bereikt is, wat kan leiden tot onhaalbaarheid.
>
> **Verplichte oplossing:** Een **bovenliggende logische schil** (State Machine) houdt bij hoeveel stappen geleden de laatste legionella-run plaatsvond. Als een run nodig is binnen de huidige horizon, forceert deze schil tijdelijk een harde ondergrens $T_{dhw,min} \leftarrow T_{leg}$ op een aaneengesloten blok van $\lceil t_{leg,min}/\Delta t \rceil$ stappen, bij voorkeur gekozen op het moment van laagste $p[k]$ binnen de horizon. Buiten dit geforceerde blok geldt de normale zachte comfortgrens.

**Gedeeld warmtepompvermogen** (indien UFH en DHW dezelfde WP gebruiken):
$$\frac{P_{UFH}[k]}{COP_{UFH}[k]} + \frac{P_{dhw}[k]}{COP_{dhw}[k]} \leq P_{hp,max,elec}$$

> Deze constraint begrenst het **elektrische** totaalvermogen van de warmtepomp. $P_{hp,max,elec}$ is de elektrische capaciteitsgrens [kW] van de installatie. Als de WP een vaste maximale thermische output heeft ($P_{hp,max,therm}$), voeg dan ook de thermische constraint $P_{UFH}[k] + P_{dhw}[k] \leq P_{hp,max,therm}$ toe — beide kunnen gelijktijdig actief zijn.

---

## 15. Parameter Woordenboek

### UFH — Huis & Systeem

| Parameter | Eenheid | Betekenis |
|---|---|---|
| $C_r$ | kWh/K | Warmtecapaciteit lucht + meubels |
| $C_b$ | kWh/K | Warmtecapaciteit vloer/beton |
| $R_{br}$ | K/kW | Thermische weerstand vloer → lucht |
| $R_{ro}$ | K/kW | Thermische weerstand huis → buiten |
| $\alpha$ | — | Fractie zonlicht direct naar lucht (0–1) |
| $\eta$ | — | Glasdoorlaatfactor (transmissie, 0–1) |
| $A_{glass}$ | m² | Raamoppervlak op de zonkant |

### UFH — MPC

| Parameter | Eenheid | Betekenis |
|---|---|---|
| $N$ | — | Vooruitkijk-horizon (bijv. 24 of 48 stappen) |
| $Q_c$ | K⁻² | Comfortgewicht |
| $R_c$ | kW⁻² | Regularisatiegewicht (demping pieken) |
| $Q_N$ | K⁻² | Eindgewicht horizon |
| $M$ | — | Straffactor soft-constraint overtreding ($M \gg Q_c$) |
| $P_{UFH,max}$ | kW | Maximaal thermisch vermogen UFH-kant warmtepomp |
| $\Delta P_{UFH,max}$ | kW/stap | Maximale ramp-rate UFH |
| $COP_{UFH}$ of $COP_{UFH}[k]$ | — | COP warmtepomp UFH-modus (schaalbaar of tijdsvariabele array) |

### UFH — Kalman Filter

| Parameter | Eenheid | Betekenis |
|---|---|---|
| $Q_{n,UFH}$ | K² | Procesruis covariantie (2×2, symmetrisch PD) |
| $R_{n,UFH}$ | K² | Meetruis variantie thermostaatsensor |
| $C_{UFH}$ | Matrix | $[1,\ 0]$ — alleen $T_r$ gemeten |

### DHW — Tank (fysisch)

| Parameter | Eenheid | Typische waarde | Betekenis |
|---|---|---|---|
| $C_{top}$ | kWh/K | 0.03–0.07 | Warmtecapaciteit bovenlaag ($\approx \tfrac{1}{2}V_{tank}\cdot\lambda$) |
| $C_{bot}$ | kWh/K | 0.03–0.07 | Warmtecapaciteit onderlaag |
| $R_{strat}$ | K/kW | 5–50 | Stratificatieweerstand (empirisch kalibreren) |
| $R_{loss}$ | K/kW | 20–100 | Isolatieweerstand boiler naar omgeving |
| $\lambda$ | kWh/(m³·K) | **1.1628** | $\rho c_p$ water (vaste fysische constante) |

> $C_{top} + C_{bot} = \lambda \cdot V_{tank}$, met $V_{tank}$ het totale tankvolume [m³].

### DHW — Verstoringen (invoer EKF en MPC)

| Parameter | Eenheid | Herkomst | Betekenis |
|---|---|---|---|
| $T_{mains}[k]$ | °C | Extern seizoensmodel of meting | Koud leidingwater (~10–12 °C NL gemiddeld) |
| $T_{amb}[k]$ | °C | Sensor of schatting | Omgevingstemperatuur rond de boiler |

### DHW — MPC

| Parameter | Eenheid | Betekenis |
|---|---|---|
| $P_{dhw,max}$ | kW | Max. thermisch vermogen naar DHW |
| $\Delta P_{dhw,max}$ | kW/stap | Max. ramp-rate DHW |
| $T_{dhw,min}$ | °C | Min. taptemperatuur comfort (bijv. 50 °C) |
| $COP_{dhw}$ of $COP_{dhw}[k]$ | — | COP warmtepomp DHW-modus (schaalbaar of tijdsvariabele array) |
| $COP_{max}$ | — | Bovengrens COP voor Fail-Fast validatie |
| $P_{hp,max,elec}$ | kW | Max. elektrisch totaalvermogen gedeelde warmtepomp |

### DHW — Legionella

| Parameter | Eenheid | Betekenis |
|---|---|---|
| $T_{leg}$ | °C | Legionella-doeltemperatuur (bijv. 60 °C) |
| $t_{leg,min}$ | h | Minimale aanhoudtijd bij $T_{leg}$ (bijv. 1 h) |
| $n_{leg}$ | stappen | Max. interval tussen twee legionella-runs (bijv. $7 \times 24 / \Delta t$) |

### DHW — Extended Kalman Filter (EKF)

| Parameter | Eenheid | Betekenis |
|---|---|---|
| $Q_{n,dhw}$ | K² | Procesruis covariantie voor temperatuurstates (2×2, symmetrisch PD) |
| $Q_{n,\dot{V}}$ | (m³/h)² | Procesruis variantie voor de $\dot{V}_{tap}$-state (scalair, tuning-parameter) |
| $Q_{n,dhw,aug}$ | gemengd | Gecombineerde 3×3 procesruiscovariantie: $\text{diag}(Q_{n,dhw},\ Q_{n,\dot{V}})$ |
| $R_{n,dhw}$ | K² | Meetruis covariantie (2×2 diagonaal): $\text{diag}(\sigma^2_{T_{top}},\ \sigma^2_{T_{bot}})$ |
| $C_{obs,dhw}$ | Matrix | $[I_{2\times2}\ \|\ 0]$ — beide temperaturen gemeten, $\dot{V}_{tap}$ niet direct gemeten |
| $\hat{\dot{V}}_{tap,init}$ | m³/h | Initiële schatting van tapwaterdebiet (typisch 0.0 bij opstart) |
| $P_{aug,init}$ | gemengd | Initiële covariantie 3×3 van augmented state (hoge onzekerheid op $\dot{V}_{tap}$ bij opstart) |

### Gedeeld

| Parameter | Eenheid | Betekenis |
|---|---|---|
| $p[k]$ | €/kWh | Dynamisch elektriciteitstarief op tijdstip $k$ |
| $\Delta t$ | h | Tijdstap (gedeeld door UFH en DHW) |

---

## 16. Software Stack & Validatie-eisen

### 16.1 Programmeertaal & Libraries

| Laag | Keuze | Motivatie |
|---|---|---|
| Taal | Python 3.10+ | Type-hints, `match`-statements, moderne dataclasses |
| Matrix-algebra | `numpy` / `scipy.sparse` | Standaard; gebruik sparse matrices voor grote horizons |
| Data-validatie (Fail-Fast) | `Pydantic v2` | Automatische type- en waardechecks bij laden van config; alle fysische onmogelijkheden (bijv. $C \leq 0$, $COP \leq 0$) worden hier onderschept vóór de solver |
| MPC-solver | **CVXPY** | Declaratieve QP/SOCP-formulering; compatibel met LTV-systemen; ondersteunt OSQP-backend voor snelheid |
| Kalman Filter | Generieke `KalmanFilter`-klasse + afgeleide `ExtendedKalmanFilter`-klasse (zie architectuureis DRY) | Geen externe library; basisklasse accepteert $A, B, E, C, Q, R$ en voert Joseph-vorm update uit; EKF-subklasse vervangt $A$ door Jacobiaan-callback $F(x, d)$ en de lineaire propagatie door $f(x, u, d)$ |

> **Solverkeuze is bindend.** De MPC-formulering moet expliciet als CVXPY-probleem worden geschreven (`cp.Problem`, `cp.Variable`, `cp.Minimize`). Het is verboden de MPC als handmatig uitgeschreven lus in numpy te implementeren.

### 16.2 Configuratieformaat

Alle parameters (zie §15) worden geladen vanuit een extern configuratiebestand (JSON of YAML). Het configuratieschema wordt gevalideerd via een `Pydantic`-model. Voorbeeld van verplichte validatieregels:

```
C_r > 0, C_b > 0                     # Warmtecapaciteiten zijn positief
R_br > 0, R_ro > 0                   # Thermische weerstanden zijn positief
0 < alpha <= 1                       # Zonlichtfractie is een kans
1 < COP_ufh <= COP_max               # COP is fysisch groter dan 1
V_tank > 0                           # Tankvolume is positief
C_top + C_bot ≈ lambda * V_tank      # Capaciteitsconservatie (tolerantie: 1e-4)
Q_n_Vtap > 0                         # Procesruis flowschatting is positief
sigma_T_top > 0, sigma_T_bot > 0     # Meetruisvarianties zijn positief
V_tap_init >= 0                      # Initieel debiet is niet-negatief
```

### 16.3 Test-eisen (pytest)

Bij de implementatie moet een `pytest`-suite worden gegenereerd die de **Fysische Consistentie-Checklist** automatisch afdwingt. Minimaal vereiste tests:

| Test | Controle | Tolerantie |
|---|---|---|
| `test_ufh_energy_balance` | $\Delta(C_r T_r + C_b T_b)/\Delta t = P_{UFH} - (T_r-T_{out})/R_{ro} + Q_{solar} + Q_{int}$ voor 10 tijdstappen | `np.testing.assert_allclose(..., rtol=1e-6)` |
| `test_dhw_energy_balance` | $\Delta(C_{top}T_{top}+C_{bot}T_{bot})/\Delta t = P_{dhw} - \dot{Q}_{tap} - \dot{Q}_{loss}$ voor 10 tijdstappen | `np.testing.assert_allclose(..., rtol=1e-6)` |
| `test_kalman_covariance_pd` | $P$ blijft symmetrisch positief-definiet na 50 Kalman-stappen (UFH) | Kleinste eigenwaarde $> 0$ |
| `test_ekf_covariance_pd` | $P_{aug}$ blijft symmetrisch positief-definiet na 50 EKF-stappen (DHW) | Kleinste eigenwaarde $> 0$ |
| `test_ekf_vtap_nonnegative` | $\hat{\dot{V}}_{tap}[k] \geq 0$ na elke EKF-updatestap, ook bij initialisatie met negatieve procesruis-realisatie | `assert all(v_tap_estimates >= 0)` |
| `test_ekf_vtap_detection` | Bij gesimuleerde tapgebeurtenis (stap in $\dot{V}_{tap}$) convergeert EKF-schatting binnen $n_{conv}$ stappen tot werkelijke waarde (binnen tolerantie $\delta_{V}$) | `assert_allclose(..., atol=delta_V)` na $n_{conv}$ stappen |
| `test_ekf_no_tap_zero` | Zonder tapgebeurtenis ($\dot{V}_{tap} = 0$, constante temperaturen): EKF-schatting convergeert naar 0 | `assert_allclose(v_tap_est, 0, atol=delta_V)` |
| `test_observability_rank` | Rang observeerbaarheidsmatrix = 2 voor UFH en 2-state DHW (MPC-model) | `np.linalg.matrix_rank(...) == 2` |
| `test_ekf_observability_rank` | Rang augmented observeerbaarheidsmatrix = 3 voor DHW-EKF bij $T_{top} \neq T_{mains}$ | `np.linalg.matrix_rank(...) == 3` |
| `test_cop_validation` | Pydantic gooit `ValidationError` bij $COP \leq 1$ of $COP > COP_{max}$ | `pytest.raises(ValidationError)` |
| `test_mpc_feasibility` | MPC-probleem is `OPTIMAL` (niet `INFEASIBLE`) voor standaardscenario | `problem.status == "optimal"` |
| `test_lambda_constant` | $\lambda$ berekend vanuit $\rho$ en $c_p$ uit config = 1.1628 binnen tolerantie | `assert_allclose(lambda_calc, 1.1628, rtol=1e-4)` |

---

## Fysische Consistentie-Checklist

Bij elke implementatie of parameterwijziging verifiëren:

| Eis | Controle |
|---|---|
| UFH energiebalans | $\Delta(C_r T_r + C_b T_b)/\Delta t = P_{UFH} - (T_r-T_{out})/R_{ro} + Q_{solar} + Q_{int}$ |
| DHW energiebalans | $\Delta(C_{top}T_{top}+C_{bot}T_{bot})/\Delta t = P_{dhw} - \dot{Q}_{tap} - \dot{Q}_{loss}$ exact per stap |
| $\lambda$ correct | $\lambda = 1.1628$ kWh/(m³·K); nooit $c_p$ in J/kgK direct gebruiken |
| Tapterm gesplitst | $-\lambda\dot{V}T_{top}$ in bovenlaag; $+\lambda\dot{V}T_{mains}$ in onderlaag |
| $P_{dhw}$ locatie | WP-wisselaar onderin → $B_{dhw}[0]=0$; elektrisch element boverin → $B_{dhw}[1]=0$ |
| $T_{dhw}$ afgeleid | Nooit als onafhankelijke state; altijd gewogen gemiddelde van $T_{top}$ en $T_{bot}$ |
| Stabiliteit Euler | $\Delta t \ll$ kleinste tijdconstante van elk subsysteem afzonderlijk |
| EKF Jacobiaan | $F[k]$ herberekenen bij elke $k$ op basis van $\hat{x}_{aug}[k]$ en $T_{mains}[k]$ |
| EKF clamp | $\hat{\dot{V}}_{tap}[k] \leftarrow \max(0,\ \hat{\dot{V}}_{tap}[k])$ na elke updatestap — harde fysische eis |
| LTV matrices (MPC) | $A_{dhw}[k]$ en $E_{dhw}[k]$ herberekenen met $\hat{\dot{V}}_{tap}[k]$ uit EKF bij elke MPC-aanroep |
| Observeerbaarheid (2-state) | Rang observeerbaarheidsmatrix = 2 voor UFH en DHW MPC-model na parametrisatie |
| Observeerbaarheid (EKF 3-state) | Rang augmented observeerbaarheidsmatrix = 3 mits $a_{strat} \neq 0$ en $T_{top} \neq T_{mains}$ |
| Kalman covariantie | $P$ en $P_{aug}$ blijven symmetrisch PD dankzij Joseph-vorm update |
| COP > 1 | Fail-Fast validatie bij laden config; kostfunctie gebruikt elektrisch vermogen = thermisch / COP |
| Soft constraints | $\epsilon_{UFH}[k] \geq 0$ en $\epsilon_{dhw}[k] \geq 0$ in QP; straffactor $M$ uit config |
| Legionella State Machine | Bovenliggende schil forceert $T_{leg}$-run binnen horizon; niet als blinde permanente QP-constraint |
| Gedeelde WP-constraint | Begrenst elektrisch vermogen via $P_{UFH}/COP_{UFH} + P_{dhw}/COP_{dhw} \leq P_{hp,max,elec}$ |
| Geen Magic Numbers | Code-review: zoeken naar hardcoded floats in wiskundige of logische vergelijkingen. Elke parameter komt uit een config-object. |
| Volledige Documentatie | Code-review: elke functie heeft een docstring mét eenheden, type-hints, en verwijzingen naar het wiskundige theorie-document. |
| CVXPY-formulering | MPC is een `cp.Problem`; geen handmatig uitgeschreven lus-optimalisatie |
| Pydantic-validatie | Alle config-parameters worden bij laden gevalideerd; fysische onmogelijkheden gooien een exception |