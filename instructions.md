# Home Optimizer: Thermisch Model (UFH + DHW + EKF + MPC) — Bindende Implementatiespecificatie

Dit document is de bindende implementatiespecificatie voor de thermische regeling van Home Optimizer. Het beschrijft een grey-box model voor ruimteverwarming via vloerverwarming (UFH), een gelaagde warmwatertank (DHW), toestandschatting met een lineair Kalman-filter en een Extended Kalman Filter (EKF), en optimalisatie met een Model Predictive Controller (MPC).

De leidende ontwerpregel is niet "zo realistisch mogelijk", maar:

> Binnen de expliciet gekozen modelaannames moet elke vergelijking intern consistent, dimensioneel correct, numeriek verantwoord en softwarematig afdwingbaar zijn.

Een grey-box model is nooit universeel exact. Daarom moeten aannames expliciet zijn, mag geen fysica impliciet worden verondersteld, en moet de code elke vereenvoudigde modelvariant weigeren zodra de bijbehorende aannames niet van toepassing zijn.

## 1. Niet-onderhandelbare eisen

### 1.1 Fysica is normatief

- De continue fysica is normatief.
- Elke discrete implementatie is een afleiding van de continue vergelijkingen, nooit andersom.
- Temperatuurverschillen mogen in graden Celsius worden uitgedrukt, omdat een verschil in `°C` numeriek gelijk is aan een verschil in `K`.
- Vermogen is altijd thermisch of elektrisch in `kW`, energie in `kWh`, temperatuur in `°C`, volume in `m³`, volumestroom in `m³/h`, tijd in `h`.
- Elke vereenvoudiging moet als modelaanname benoemd zijn.
- Als een aanname niet geldt voor de echte installatie, dan moet de code die modelvariant weigeren.

### 1.2 Geen magic numbers

- Elke fysische constante, veiligheidstemperatuur, tolerantie, tijdstap, solver-weight en validatiegrens komt uit een gevalideerd configuratie-object.
- Zelfs universele of quasi-universele constanten krijgen een naam, bijvoorbeeld:
  - `temperature_offset_c_to_k`
  - `absolute_zero_celsius`
  - `joules_per_kwh`
  - `rho_water_ref`
  - `cp_water_ref`
  - `lambda_water_ref`
- De code gebruikt geen losse numerieke constanten in wiskundige logica, behalve `0` en `1`.

### 1.3 DRY en generieke architectuur

- Schrijf generieke bouwstenen:
  - `ContinuousLinearModel`
  - `DiscreteLinearModel`
  - `LinearKalmanFilter`
  - `ExtendedKalmanFilter`
  - `Discretizer`
  - `MpcProblemBuilder`
  - `LegionellaSupervisor`
  - `HeatPumpTopologySupervisor`
- UFH en DHW delen dezelfde algebra: state-space, Joseph-update, observeerbaarheidscontrole, discretisatie en constraints. Deze logica mag niet worden gekopieerd.

### 1.4 Fail-fast

- Ontbreekt een parameter of tijdreeks, dan wordt een exception gegooid. Er zijn geen verborgen defaults.
- Elke fysisch onmogelijke waarde wordt geweigerd voordat zij model, filter of solver bereikt.
- Voorbeelden van harde blokkades:
  - warmtecapaciteit `<= 0`
  - thermische weerstand `<= 0`
  - `alpha_solar` buiten `[0, 1]`
  - `eta_window` buiten `[0, 1]`
  - `heater_split_top + heater_split_bottom` buiten tolerantie rond `1`
  - temperatuur `< absolute_zero_celsius`
  - `lambda_water_ref <= 0`
  - `cop <= cop_min_physical`
  - `cop > cop_max`
  - negatieve tapstroom aan MPC of plantmodel
  - indoor-coupled tankverliezen combineren met block-diagonale systeemopbouw
  - exclusieve warmtepomptopologie modelleren alsof simultane levering zonder modeselectie is toegestaan

### 1.5 Documentatie

- Elke klasse en functie krijgt type hints en een docstring met:
  - fysieke betekenis
  - eenheden
  - matrixdimensies
  - relevante sectie uit dit document
- Comments documenteren het waarom, niet het wat.

## 2. Globale conventies

### 2.1 Tijdsindex

Alle inputs en verstoringen zijn stukgewijs constant over het interval:

$$
[k \Delta t,\ (k+1)\Delta t)
$$

met sampletijd:

$$
\Delta t > 0
$$

### 2.2 Water-eigenschappen

De basisvariant gebruikt temperatuur-onafhankelijke referentie-eigenschappen van vloeibaar water:

$$
\lambda_{water,ref}
=
\frac{\rho_{water,ref}\, c_{p,water,ref}}{joules\_per\_kwh}
\quad
\left[\frac{kWh}{m^3 \cdot K}\right]
$$

Belangrijk:

- Dit is een modelaanname, geen natuurwet.
- In werkelijkheid hangen `rho` en `c_p` van temperatuur af.
- Als temperatuurafhankelijke watereigenschappen nodig zijn, moet `lambda_water_ref` worden vervangen door een expliciete functie `lambda_water(T)`. De rest van de architectuur blijft gelijk.
- Gebruik nooit rechtstreeks `c_p` in `J/(kg·K)` in toestandsvergelijkingen. Converteer eerst naar `lambda_water_ref` of `lambda_water(T)`.

### 2.3 Energiegrootheden: absoluut versus affine

Binnen dit document worden toestanden uitgedrukt in `°C`. Daardoor gelden uitdrukkingen van de vorm `C T` wel als affine energie-representatie voor energiebalansen en differentiaalvergelijkingen, maar niet automatisch als absolute thermodynamische energie-inhoud.

Daarom gelden de volgende regels:

- Voor afgeleiden en balansen mag men schrijven:

$$
\frac{d}{dt}(C T)
$$

- Als een absolute of interpreteerbare energie-inhoud nodig is, dan moet een referentietemperatuur worden gekozen:

$$
E^{rel} = C \left(T - T_{energy,ref}\right)
$$

- Elke KPI, SOC-achtige maat of terminal penalty die als "energie" wordt geïnterpreteerd, moet expliciet vermelden of zij affine is of relatief ten opzichte van `T_energy_ref`.

### 2.4 Procesruis: continue of discrete betekenis

Voor estimator-ruis gelden exact twee toegestane interpretaties:

1. `Q_ufh` en `Q_dhw_aug` zijn direct discrete procesruiscovarianties per sample.
2. Er is een expliciete afleiding van continue ruisspectra naar discrete covarianties via een vastgelegd discretisatiebeleid.

Bindende regel:

- De implementatie moet exact één van beide interpretaties kiezen.
- De code mag niet impliciet discrete en continue ruisspecificaties mengen.

De basisvariant van deze specificatie gebruikt:

- `Q_ufh`: discrete procesruiscovariantie
- `Q_dhw_aug`: discrete procesruiscovariantie

Voor het augmented DHW-EKF-model geldt aanvullend:

- `Q_dhw_aug` is de volledige discrete procesruiscovariantie van de augmented state `\begin{bmatrix} T_{top} & T_{bot} & \dot{V}_{tap} \end{bmatrix}^{T}`
- `Q_vtap` is de benoemde discrete procesruisvariantie van de random-walk state `\dot{V}_{tap}`
- als `Q_dhw_aug` direct wordt geconfigureerd, dan moet de implementatie expliciet vastleggen hoe `Q_vtap` correspondeert met het `(3,3)`-element of subblok van `Q_dhw_aug`
- als `Q_dhw_aug` programmatisch wordt opgebouwd, dan moet `Q_vtap` expliciet worden gebruikt als de procesruis voor de derde augmented state
- de code mag `Q_vtap` niet behandelen als een extra, losstaande covariantie naast `Q_dhw_aug`; het is een benoemde parameter binnen dezelfde augmented-ruissemantiek

### 2.5 Discretisatiebeleid

De productie-implementatie ondersteunt precies twee schema's:

1. `exact_zoh`
2. `forward_euler`

Regels:

- `exact_zoh` is de standaard voor lineaire modellen en lineair bevroren LTV-stappen.
- `forward_euler` is nooit de impliciete default; het is uitsluitend toegestaan als expliciet geconfigureerde en runtime-gevalideerde fallback.
- `forward_euler` mag alleen gebruikt worden als een runtime-validator aantoont dat de discrete stap voor de actuele parameters numeriek stabiel is en geen onfysische tekenomkeringen introduceert.
- De code mag nooit impliciet van schema wisselen.

Voor een continu lineair model

$$
\dot{x}(t) = A_c x(t) + B_c u(t) + E_c d(t)
$$

met stukgewijs constante `u` en `d` over één sample, is de exacte ZOH-discretisatie:

$$
x[k+1] = A_d x[k] + B_d u[k] + E_d d[k]
$$

met

$$
\exp\!\left(
\begin{bmatrix}
A_c & B_c & E_c \\
0   & 0   & 0 \\
0   & 0   & 0
\end{bmatrix}
\Delta t
\right)
=
\begin{bmatrix}
A_d & B_d & E_d \\
0   & I   & 0 \\
0   & 0   & I
\end{bmatrix}
$$

Deze augmented-matrixvorm moet gebruikt worden; vermijd formules die `A_c^{-1}` vereisen.

Voor `forward_euler` geldt:

$$
A_d^{Euler} = I + \Delta t\,A_c,\qquad
B_d^{Euler} = \Delta t\,B_c,\qquad
E_d^{Euler} = \Delta t\,E_c
$$

### 2.6 Bindende Euler-admissibiliteit

De check

$$
\rho\!\left(A_d^{Euler}\right) < 1
$$

is noodzakelijk, maar niet voldoende.

Bij gebruik van `forward_euler` moet de runtime-validator minimaal controleren:

1. spectrale stabiliteit:

$$
\rho\!\left(A_d^{Euler}\right) < 1
$$

2. behoud van fysisch monotone zelfdemping voor thermische states:
   - relevante diagonale zelfcoëfficiënten mogen niet negatief worden door de tijdstap
   - voor bekende referentievormen moeten de expliciete voorwaarden worden gecontroleerd

3. geen onfysische tekenomkering in warmte-uitwisselingscoëfficiënten

Voor de expliciete referentievormen betekent dit minimaal:

UFH:

$$
1 - a_{br} - a_{ro} \ge 0
$$

$$
1 - b_{br} \ge 0
$$

DHW:

$$
1 - a_{strat} - a_{loss} - a_{tap}[k] \ge 0
$$

$$
1 - b_{strat} - b_{loss} - b_{tap}[k] \ge 0
$$

Voor LTV-systemen gebeurt deze check op elke stap van de horizon of op een bewezen conservatieve worst-case.

Als één van deze checks faalt, dan moet `forward_euler` worden geweigerd voor de actuele configuratie of stap.

## 3. UFH: ruimteverwarming

### 3.1 States, input en verstoringen

States:

- `T_r` [°C]: ruimtetemperatuur
- `T_b` [°C]: temperatuur van de thermische massa of vloerbuffer

Actuator:

- `P_ufh` [kW_th]: thermisch vermogen naar de UFH-lus

Verstoringen:

- `T_out` [°C]: buitentemperatuur
- `Q_solar` [kW_th]: zonnewinst via ramen
- `Q_int_base` [kW_th]: interne warmtelast zonder boilerverliezen

De totale interne warmtelast voor het UFH-model is:

$$
Q_{int,eff} = Q_{int,base} + Q_{tank \rightarrow room}
$$

waarbij `Q_{tank \rightarrow room}` nul is in de ontkoppelde variant en expliciet wordt gedefinieerd in sectie 5.2 wanneer boilerverliezen in de woning terechtkomen.

### 3.2 Zonmodel

Het lineaire venstermodel luidt:

$$
Q_{solar}[k]
=
\frac{A_{glass,eff}\, GTI_{window}[k]\, \eta_{window}}{power\_unit\_scale}
$$

waar:

- `GTI_window` de instraling op het werkelijke raamvlak is, niet op een horizontaal vlak
- `A_glass_eff` het effectieve transparante oppervlak is
- `eta_window` een effectieve factor is die transmissie, kozijnfractie, schaduw en eventueel zonwering mag samenvatten
- `power_unit_scale` de configuratieconstante is die van `W` naar `kW` converteert

### 3.3 Continue fysica

De UFH-plant is een twee-knoops RC-model:

$$
C_b \frac{dT_b}{dt}
=
P_{ufh}
- \frac{T_b - T_r}{R_{br}}
+ (1-\alpha_{solar}) Q_{solar}
$$

$$
C_r \frac{dT_r}{dt}
=
\frac{T_b - T_r}{R_{br}}
- \frac{T_r - T_{out}}{R_{ro}}
+ \alpha_{solar} Q_{solar}
+ Q_{int,eff}
$$

met:

- `C_r`, `C_b` [kWh/K]
- `R_br`, `R_ro` [K/kW]
- `alpha_solar` [-] in `[0, 1]`

### 3.4 Continue state-space

Kies:

$$
x_{ufh} = \begin{bmatrix} T_r \\ T_b \end{bmatrix},\qquad
u_{ufh} = \begin{bmatrix} P_{ufh} \end{bmatrix},\qquad
d_{ufh} = \begin{bmatrix} T_{out} \\ Q_{solar} \\ Q_{int,eff} \end{bmatrix}
$$

Dan geldt:

$$
\dot{x}_{ufh} = A_{c,ufh}\,x_{ufh} + B_{c,ufh}\,u_{ufh} + E_{c,ufh}\,d_{ufh}
$$

met:

$$
A_{c,ufh}
=
\begin{bmatrix}
-\left(\frac{1}{C_r R_{br}} + \frac{1}{C_r R_{ro}}\right) & \frac{1}{C_r R_{br}} \\
\frac{1}{C_b R_{br}} & -\frac{1}{C_b R_{br}}
\end{bmatrix}
$$

$$
B_{c,ufh}
=
\begin{bmatrix}
0 \\
\frac{1}{C_b}
\end{bmatrix}
$$

$$
E_{c,ufh}
=
\begin{bmatrix}
\frac{1}{C_r R_{ro}} & \frac{\alpha_{solar}}{C_r} & \frac{1}{C_r} \\
0 & \frac{1-\alpha_{solar}}{C_b} & 0
\end{bmatrix}
$$

De energiebalans van het volledige UFH-subsysteem is:

$$
\frac{d}{dt}\left(C_r T_r + C_b T_b\right)
=
P_{ufh}
- \frac{T_r - T_{out}}{R_{ro}}
+ Q_{solar}
+ Q_{int,eff}
$$

Deze vergelijking is normatief binnen de gekozen modelaannames. Als een absolute of SOC-achtige energiegrootheid nodig is, dan moet ook hier een expliciete referentietemperatuur worden gedefinieerd conform sectie 2.3.

### 3.5 Discrete vorm

De implementatie moet `A_d`, `B_d` en `E_d` afleiden uit `A_c`, `B_c` en `E_c` met het gekozen schema.

Alleen als `forward_euler` expliciet is ingeschakeld, mag onderstaande referentievorm gebruikt worden:

$$
a_{br} = \frac{\Delta t}{C_r R_{br}},\qquad
a_{ro} = \frac{\Delta t}{C_r R_{ro}},\qquad
b_{br} = \frac{\Delta t}{C_b R_{br}}
$$

$$
A_{d,ufh}^{Euler}
=
\begin{bmatrix}
1 - a_{br} - a_{ro} & a_{br} \\
b_{br} & 1 - b_{br}
\end{bmatrix}
$$

$$
B_{d,ufh}^{Euler}
=
\begin{bmatrix}
0 \\
\frac{\Delta t}{C_b}
\end{bmatrix}
$$

$$
E_{d,ufh}^{Euler}
=
\begin{bmatrix}
a_{ro} & \alpha_{solar}\frac{\Delta t}{C_r} & \frac{\Delta t}{C_r} \\
0 & (1-\alpha_{solar})\frac{\Delta t}{C_b} & 0
\end{bmatrix}
$$

### 3.6 Observeerbaarheid

De UFH-meetmatrix is:

$$
C_{obs,ufh} = \begin{bmatrix} 1 & 0 \end{bmatrix}
$$

Na parametrisatie moet de code numeriek controleren:

$$
\mathcal{O}_{ufh}
=
\begin{bmatrix}
C_{obs,ufh} \\
C_{obs,ufh} A_{d,ufh}
\end{bmatrix}
$$

Bindende regels:

- de code moet de rang numeriek controleren
- de code moet daarnaast een conditioneringsmaat controleren, bijvoorbeeld de kleinste singuliere waarde of het condition number
- een formeel volle rang maar numeriek bijna-singuliere observability-matrix geldt niet als robuust observeerbaar
- voor de UFH-plant is onvoldoende observeerbaarheid een harde configuratiefout

Bij de Euler-vorm reduceert de determinant tot een term evenredig met `b_br`; het systeem is dus structureel observeerbaar zolang `C_b`, `R_br` en `delta_t` fysisch geldig en eindig zijn. De runtime-implementatie mag dit structurele argument echter niet gebruiken als vervanging voor de numerieke check.

### 3.7 Lineair Kalman-filter

Meetmodel:

$$
y_{ufh}[k] = C_{obs,ufh}\,x_{ufh}[k] + v[k]
$$

met:

- `Q_ufh`: discrete procesruiscovariantie, symmetrisch positief semidefiniet
- `R_ufh`: discrete meetruiscovariantie, symmetrisch positief definiet

Predictie:

$$
\hat{x}_{ufh}^{-}[k]
=
A_{d,ufh}\,\hat{x}_{ufh}[k-1]
+ B_{d,ufh}\,u_{ufh}[k-1]
+ E_{d,ufh}\,d_{ufh}[k-1]
$$

$$
P_{ufh}^{-}[k]
=
A_{d,ufh}\,P_{ufh}[k-1]\,A_{d,ufh}^{T}
+ Q_{ufh}
$$

Kalman gain:

$$
S_{ufh}[k]
=
C_{obs,ufh}\,P_{ufh}^{-}[k]\,C_{obs,ufh}^{T} + R_{ufh}
$$

$$
K_{ufh}[k]
=
P_{ufh}^{-}[k]\,C_{obs,ufh}^{T}\,S_{ufh}[k]^{-1}
$$

Update in Joseph-vorm:

$$
\hat{x}_{ufh}[k]
=
\hat{x}_{ufh}^{-}[k]
+ K_{ufh}[k]\left(y_{ufh}[k] - C_{obs,ufh}\hat{x}_{ufh}^{-}[k]\right)
$$

$$
P_{ufh}[k]
=
\left(I - K_{ufh}[k] C_{obs,ufh}\right) P_{ufh}^{-}[k] \left(I - K_{ufh}[k] C_{obs,ufh}\right)^T
+ K_{ufh}[k] R_{ufh} K_{ufh}[k]^T
$$

Bindende numerieke regel:

- `P_ufh` moet na elke stap symmetrisch blijven
- `P_ufh` moet positief semidefiniet blijven binnen `covariance_psd_tolerance`
- als de implementatie strikt positief definiete covariantie verlangt, dan moet de gebruikte regularisatie expliciet worden gedocumenteerd

## 4. DHW: tweelaags warmwatertank

### 4.1 Verplichte modelaannames

De basisvariant van het DHW-model gebruikt de volgende expliciete aannames:

- `A1`: elke laag is perfect gemengd
- `A2`: inter-node uitwisseling wordt gemodelleerd als een effectieve conductantie of mengterm via `R_strat`
- `A3`: tijdens tappen verlaat warm water de bovenlaag en komt leidingwater onderaan binnen
- `A4`: leidingwater heeft temperatuur `T_mains`
- `A5`: de tankwandverliezen worden gemodelleerd als lineair met de temperatuurgradiënt naar een omgeving `T_amb_tank`
- `A6`: er is geen directe flowmeting; `Vdot_tap` is een augmented state
- `A7`: de MPC gebruikt de EKF-schatting `Vdot_tap_hat` als bekende LTV-parameter
- `A8`: de verwarmingsbron levert thermisch vermogen volgens een configurabele verdeling `heater_split_top`, `heater_split_bottom`

Aanname `A2` verdient expliciete nuance:

- `R_strat` is geen first-principles materiaalweerstand.
- `R_strat` is een effectieve grey-box parameter die turbulente menging, interne recirculatie en warmtewisselaarinvloed samenvat.
- `R_strat` moet aan meetdata worden gekalibreerd en mag niet uit materiaaldata van water worden afgeleid.

### 4.2 Semantiek van `Vdot_tap`

De state `Vdot_tap` representeert in de basisvariant het effectieve volumetrische tapdebiet dat energetisch overeenkomt met:

- warm water dat uit de bovenlaag vertrekt
- vervanging door mains-water aan de onderzijde

Bindende scopebeperking:

- downstream mengventielen
- ringleiding of recirculatie
- tappuntdynamiek buiten de tank
- complexe interne warmtewisselaars

vallen buiten scope, tenzij expliciet als extra modelvariant gemodelleerd.

### 4.3 States, input, verstoringen en afgeleide grootheden

Augmented state voor de EKF:

$$
x_{dhw,aug}
=
\begin{bmatrix}
T_{top} \\
T_{bot} \\
\dot{V}_{tap}
\end{bmatrix}
$$

State voor de MPC:

$$
x_{dhw}
=
\begin{bmatrix}
T_{top} \\
T_{bot}
\end{bmatrix}
$$

Actuator:

$$
u_{dhw} = \begin{bmatrix} P_{dhw} \end{bmatrix}
$$

Verstoringen voor de MPC in de ontkoppelde basisvariant:

$$
d_{dhw}
=
\begin{bmatrix}
T_{amb,tank} \\
T_{mains}
\end{bmatrix}
$$

Heater-split:

$$
heater\_split_{top} + heater\_split_{bottom} = 1
$$

met:

$$
0 \le heater\_split_{top} \le 1,\qquad
0 \le heater\_split_{bottom} \le 1
$$

Afgeleide signalen:

- affiene energie-state:

$$
E_{dhw,state}^{affine} = C_{top} T_{top} + C_{bot} T_{bot}
$$

- relatieve energie ten opzichte van referentie:

$$
E_{dhw,state}^{rel}
=
C_{top}\left(T_{top} - T_{energy,ref}\right)
+ C_{bot}\left(T_{bot} - T_{energy,ref}\right)
$$

- energie-gewogen gemiddelde toestandstemperatuur:

$$
T_{dhw,energy}
=
\frac{C_{top}T_{top} + C_{bot}T_{bot}}{C_{top} + C_{bot}}
$$

Belangrijk:

- `T_dhw_energy` is alleen een energie-gewogen state-gemiddelde.
- Het is niet automatisch gelijk aan de volume-gemiddelde watertemperatuur.
- Gebruik `T_top` voor tapcomfort.
- Gebruik een expliciete legionella-surrogaatconstraint voor hygiëne.
- Gebruik voor terminal penalties of SOC-achtige logica alleen een expliciet affine of relatieve energiedefinitie; laat die semantiek nooit impliciet.

### 4.4 Continue fysica

Standby-verliezen:

$$
\dot{Q}_{loss,top}
=
\frac{T_{top} - T_{amb,tank}}{R_{loss,top}}
$$

$$
\dot{Q}_{loss,bot}
=
\frac{T_{bot} - T_{amb,tank}}{R_{loss,bot}}
$$

Effectieve inter-node uitwisseling:

$$
\dot{Q}_{strat}
=
\frac{T_{top} - T_{bot}}{R_{strat}}
$$

Tapgerelateerde energiestromen:

$$
\dot{Q}_{tap,total}
=
\lambda_{water,ref}\,\dot{V}_{tap}\,(T_{top} - T_{mains})
$$

Gesplitst per laag:

$$
\dot{Q}_{tap,top}
=
\lambda_{water,ref}\,\dot{V}_{tap}\,(T_{bot} - T_{top})
$$

$$
\dot{Q}_{tap,bot}
=
\lambda_{water,ref}\,\dot{V}_{tap}\,(T_{mains} - T_{bot})
$$

Continue ODE's:

$$
C_{top}\frac{dT_{top}}{dt}
=
-\dot{Q}_{strat}
+ \dot{Q}_{tap,top}
+ heater\_split_{top} P_{dhw}
- \dot{Q}_{loss,top}
$$

$$
C_{bot}\frac{dT_{bot}}{dt}
=
\dot{Q}_{strat}
+ \dot{Q}_{tap,bot}
+ heater\_split_{bottom} P_{dhw}
- \dot{Q}_{loss,bot}
$$

Volledige energiebalans:

$$
\frac{d}{dt}\left(C_{top}T_{top} + C_{bot}T_{bot}\right)
=
P_{dhw}
- \lambda_{water,ref}\,\dot{V}_{tap}\,(T_{top} - T_{mains})
- \dot{Q}_{loss,top}
- \dot{Q}_{loss,bot}
$$

Deze vergelijking is exact binnen de gekozen aannames.

### 4.5 Capaciteitsconservatie

De code mag niet blind afdwingen:

$$
C_{top} + C_{bot} = \lambda_{water,ref} V_{tank}
$$

want dat is alleen exact als:

- de state-capaciteiten uitsluitend watercapaciteit representeren
- tankwand, warmtewisselaar, dompelhuls en overige parasitaire massa worden verwaarloosd

De fysisch correcte validatieregel is:

$$
C_{top} + C_{bot}
=
\lambda_{water,ref} V_{tank,active}
+ C_{tank,parasitic}
$$

binnen een benoemde tolerantie `capacity_balance_tolerance`.

Als de implementatie de pure-waterbenadering wil gebruiken, dan moet expliciet worden ingesteld:

$$
C_{tank,parasitic} = 0
$$

### 4.6 Continue LTV-vorm voor de MPC

Tijdens de MPC wordt de door de EKF geleverde `\hat{\dot{V}}_{tap}[k]` als bekende parameter per stap behandeld.

Definieer:

$$
A_{c,dhw}[k]
=
\begin{bmatrix}
-\left(
\frac{1}{C_{top}R_{strat}}
+ \frac{1}{C_{top}R_{loss,top}}
+ \frac{\lambda_{water,ref}\hat{\dot{V}}_{tap}[k]}{C_{top}}
\right)
&
\frac{1}{C_{top}R_{strat}}
+ \frac{\lambda_{water,ref}\hat{\dot{V}}_{tap}[k]}{C_{top}}
\\
\frac{1}{C_{bot}R_{strat}}
&
-\left(
\frac{1}{C_{bot}R_{strat}}
+ \frac{1}{C_{bot}R_{loss,bot}}
+ \frac{\lambda_{water,ref}\hat{\dot{V}}_{tap}[k]}{C_{bot}}
\right)
\end{bmatrix}
$$

$$
B_{c,dhw}
=
\begin{bmatrix}
\frac{heater\_split_{top}}{C_{top}} \\
\frac{heater\_split_{bottom}}{C_{bot}}
\end{bmatrix}
$$

$$
E_{c,dhw}[k]
=
\begin{bmatrix}
\frac{1}{C_{top}R_{loss,top}} & 0 \\
\frac{1}{C_{bot}R_{loss,bot}} & \frac{\lambda_{water,ref}\hat{\dot{V}}_{tap}[k]}{C_{bot}}
\end{bmatrix}
$$

De discrete matrices `A_dhw[k]`, `B_dhw[k]` en `E_dhw[k]` moeten daaruit met het gekozen schema worden afgeleid.

### 4.7 Forward-Euler referentievorm

Alleen voor `forward_euler`:

$$
a_{strat} = \frac{\Delta t}{C_{top}R_{strat}},\qquad
b_{strat} = \frac{\Delta t}{C_{bot}R_{strat}}
$$

$$
a_{loss} = \frac{\Delta t}{C_{top}R_{loss,top}},\qquad
b_{loss} = \frac{\Delta t}{C_{bot}R_{loss,bot}}
$$

$$
a_{tap}[k] = \frac{\Delta t\,\lambda_{water,ref}\hat{\dot{V}}_{tap}[k]}{C_{top}},\qquad
b_{tap}[k] = \frac{\Delta t\,\lambda_{water,ref}\hat{\dot{V}}_{tap}[k]}{C_{bot}}
$$

$$
A_{d,dhw}^{Euler}[k]
=
\begin{bmatrix}
1 - a_{strat} - a_{loss} - a_{tap}[k] & a_{strat} + a_{tap}[k] \\
b_{strat} & 1 - b_{strat} - b_{loss} - b_{tap}[k]
\end{bmatrix}
$$

$$
B_{d,dhw}^{Euler}
=
\begin{bmatrix}
\frac{\Delta t\,heater\_split_{top}}{C_{top}} \\
\frac{\Delta t\,heater\_split_{bottom}}{C_{bot}}
\end{bmatrix}
$$

$$
E_{d,dhw}^{Euler}[k]
=
\begin{bmatrix}
a_{loss} & 0 \\
b_{loss} & b_{tap}[k]
\end{bmatrix}
$$

### 4.8 Observeerbaarheid van het 2-state MPC-model

Als alleen `T_top` gemeten zou worden, is de Euler-vorm observeerbaar zolang:

$$
a_{strat} + a_{tap}[k] \ne 0
$$

In de feitelijke implementatie met ZOH of met beide sensoren beschikbaar moet de code de observeerbaarheid numeriek controleren na parametrisatie, niet via een los handmatig argument.

Bindende regels:

- voor het 2-state MPC-model moet de observeerbaarheidscontrole worden uitgevoerd na parametrisatie
- naast rang moet ook een conditioneringsmaat worden gecontroleerd
- voor het 2-state MPC-model is onvoldoende observeerbaarheid een harde configuratiefout

### 4.9 Extended Kalman Filter

#### 4.9.1 Meetmodel

De DHW-sensoren meten beide lagen:

$$
y_{dhw}[k]
=
\begin{bmatrix}
T_{top}^{meas}[k] \\
T_{bot}^{meas}[k]
\end{bmatrix}
=
C_{obs,dhw}\,x_{dhw,aug}[k] + v[k]
$$

$$
C_{obs,dhw}
=
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0
\end{bmatrix}
$$

`R_dhw` is een `2 \times 2` symmetrisch positief definiete matrix.

#### 4.9.2 Random-walk model voor het debiet

$$
\dot{V}_{tap}[k+1]
=
\dot{V}_{tap}[k] + w_{\dot{V}}[k]
$$

met `Q_vtap` als discrete procesruisvariantie.

#### 4.9.3 Toegestane discrete procesmap voor de EKF

De EKF-basisklasse accepteert een discrete transitiecallback:

$$
x[k+1] = f_d(x[k], u[k], d[k]) + w[k]
$$

Bindende regel:

- de Jacobiaan moet horen bij exact dezelfde discrete procesmap als de predictiestap
- het is fysisch en numeriek onjuist om een Euler-Jacobiaan te combineren met een niet-Euler predictiestap

De basisvariant van deze specificatie staat precies twee EKF-beleidskeuzes toe:

1. basisbeleid: EKF gebruikt `forward_euler` als discrete procesmap
2. uitgebreid beleid: een hogere-orde discrete procesmap is toegestaan, mits zowel predictiestap als Jacobiaan uit diezelfde map worden afgeleid

Operationele default:

- de productie-default is een Euler-EKF
- een hogere-orde EKF is alleen toegestaan bij expliciete activatie in configuratie

#### 4.9.4 Discrete niet-lineaire procesfunctie voor Euler

Voor `forward_euler` is de expliciete procesmap:

$$
f_d(x_{dhw,aug}, u_{dhw}, d_{dhw})
=
\begin{bmatrix}
T_{top}
+ \frac{\Delta t}{C_{top}}
\left(
-\frac{T_{top}-T_{bot}}{R_{strat}}
+ \lambda_{water,ref}\dot{V}_{tap}(T_{bot}-T_{top})
+ heater\_split_{top} P_{dhw}
- \frac{T_{top}-T_{amb,tank}}{R_{loss,top}}
\right)
\\
T_{bot}
+ \frac{\Delta t}{C_{bot}}
\left(
\frac{T_{top}-T_{bot}}{R_{strat}}
+ \lambda_{water,ref}\dot{V}_{tap}(T_{mains}-T_{bot})
+ heater\_split_{bottom} P_{dhw}
- \frac{T_{bot}-T_{amb,tank}}{R_{loss,bot}}
\right)
\\
\dot{V}_{tap}
\end{bmatrix}
$$

#### 4.9.5 Jacobiaan van de Euler-procesmap

De Jacobiaan moet geëvalueerd worden op het punt:

$$
\left(\hat{x}_{dhw,aug}[k-1],\,u_{dhw}[k-1],\,d_{dhw}[k-1]\right)
$$

dus strikt vóór de predictiestap.

Voor de Euler-discrete map is:

$$
F_d[k-1]
=
\begin{bmatrix}
1 - a_{strat} - a_{loss} - \hat{a}_{tap}[k-1]
&
a_{strat} + \hat{a}_{tap}[k-1]
&
\frac{\Delta t\,\lambda_{water,ref}}{C_{top}}
\left(\hat{T}_{bot}[k-1] - \hat{T}_{top}[k-1]\right)
\\
b_{strat}
&
1 - b_{strat} - b_{loss} - \hat{b}_{tap}[k-1]
&
\frac{\Delta t\,\lambda_{water,ref}}{C_{bot}}
\left(T_{mains}[k-1] - \hat{T}_{bot}[k-1]\right)
\\
0 & 0 & 1
\end{bmatrix}
$$

met:

$$
\hat{a}_{tap}[k-1]
=
\frac{\Delta t\,\lambda_{water,ref}\hat{\dot{V}}_{tap}[k-1]}{C_{top}}
$$

$$
\hat{b}_{tap}[k-1]
=
\frac{\Delta t\,\lambda_{water,ref}\hat{\dot{V}}_{tap}[k-1]}{C_{bot}}
$$

#### 4.9.6 Observeerbaarheid van het augmented systeem

Omdat beide temperaturen direct gemeten worden, leveren de eerste twee kolommen van de observeerbaarheidsmatrix al rang `2`. De derde state `\dot{V}_{tap}` is lokaal observeerbaar als ten minste één van de gevoeligheden naar de temperatuurmetingen niet nul is:

$$
\frac{\partial f_{top}}{\partial \dot{V}_{tap}}
=
\frac{\Delta t\,\lambda_{water,ref}}{C_{top}}(T_{bot} - T_{top})
$$

$$
\frac{\partial f_{bot}}{\partial \dot{V}_{tap}}
=
\frac{\Delta t\,\lambda_{water,ref}}{C_{bot}}(T_{mains} - T_{bot})
$$

Voor de lokale linearisatie is een voldoende voorwaarde voor observeerbaarheid van de derde state:

$$
\left(T_{top} \ne T_{bot}\right)
\ \lor\
\left(T_{bot} \ne T_{mains}\right)
$$

De vuistregel "observeerbaar zolang `T_top != T_mains`" is onvoldoende precies en mag niet gebruikt worden.

Deze diagnose is lokaal en linearisatie-gebaseerd rond het actuele lineariseerpunct; zij is geen globale observeerbaarheidsbewijsstelling voor de volledige niet-lineaire dynamica.

Bindende interpretatie:

- tijdelijke lokale onobserveerbaarheid van `\dot{V}_{tap}` is een operationele toestand, geen configuratiefout
- de implementatie mag hier dus niet hard op falen
- de implementatie moet wel:
  - een diagnostische status registreren
  - de schatting begrensd houden
  - eventueel covariantie-inflatie of een vergelijkbare robuustheidsmaatregel ondersteunen

#### 4.9.7 EKF-algoritme

Predictie:

$$
\hat{x}_{dhw,aug}^{-}[k]
=
f_d\!\left(\hat{x}_{dhw,aug}[k-1], u_{dhw}[k-1], d_{dhw}[k-1]\right)
$$

$$
P_{dhw,aug}^{-}[k]
=
F_d[k-1]\,P_{dhw,aug}[k-1]\,F_d[k-1]^T + Q_{dhw,aug}
$$

Kalman gain:

$$
S_{dhw}[k]
=
C_{obs,dhw}\,P_{dhw,aug}^{-}[k]\,C_{obs,dhw}^T + R_{dhw}
$$

$$
K_{dhw}[k]
=
P_{dhw,aug}^{-}[k]\,C_{obs,dhw}^T\,S_{dhw}[k]^{-1}
$$

Update:

$$
\hat{x}_{dhw,aug}[k]
=
\hat{x}_{dhw,aug}^{-}[k]
+ K_{dhw}[k]
\left(
y_{dhw}[k] - C_{obs,dhw}\hat{x}_{dhw,aug}^{-}[k]
\right)
$$

Joseph-vorm:

$$
P_{dhw,aug}[k]
=
\left(I - K_{dhw}[k] C_{obs,dhw}\right)
P_{dhw,aug}^{-}[k]
\left(I - K_{dhw}[k] C_{obs,dhw}\right)^T
+ K_{dhw}[k] R_{dhw} K_{dhw}[k]^T
$$

#### 4.9.8 Harde fysische projectie op tapdebiet

Na elke update geldt verplicht:

$$
\hat{\dot{V}}_{tap}[k]
\leftarrow
\max\!\left(0,\ \hat{x}_{dhw,aug}[k]_3\right)
$$

Verplicht:

- de clamp gebeurt na iedere update
- de geclamte waarde wordt doorgegeven aan de MPC
- de MPC mag nooit een negatieve tapstroom ontvangen

Belangrijke numerieke nuance:

- deze projectie is een bewuste projected-EKF heuristic
- de geprojecteerde state is fysisch correcter dan de ongeprojecteerde Gaussian-update
- de implementatie moet documenteren dat deze stap buiten de standaard Gaussian EKF-formulering valt
- als frequente of grote clamping optreedt, moet de filtertuning worden herzien

Bindende covariantieregel:

- `P_dhw_aug` moet na elke stap symmetrisch blijven
- `P_dhw_aug` moet positief semidefiniet blijven binnen `covariance_psd_tolerance`

## 5. Gekoppeld systeem

### 5.1 Variant A: thermisch ontkoppeld

Deze variant is alleen toegestaan als ten minste één van de volgende uitspraken waar is:

- de tank staat buiten de geconditioneerde zone
- standby-verliezen zijn verwaarloosbaar voor de ruimtebalans
- standby-verliezen zijn al verwerkt in `Q_int_base`

Alleen dan mag de gecombineerde MPC-state block-diagonaal zijn:

$$
x_{tot}
=
\begin{bmatrix}
T_r \\
T_b \\
T_{top} \\
T_{bot}
\end{bmatrix},
\qquad
u_{tot}
=
\begin{bmatrix}
P_{ufh} \\
P_{dhw}
\end{bmatrix}
$$

$$
x_{tot}[k+1]
=
\begin{bmatrix}
A_{ufh} & 0 \\
0 & A_{dhw}[k]
\end{bmatrix}
x_{tot}[k]
+
\begin{bmatrix}
B_{ufh} & 0 \\
0 & B_{dhw}
\end{bmatrix}
u_{tot}[k]
+
\begin{bmatrix}
E_{ufh} & 0 \\
0 & E_{dhw}[k]
\end{bmatrix}
\begin{bmatrix}
d_{ufh}[k] \\
d_{dhw}[k]
\end{bmatrix}
$$

### 5.2 Variant B: standby-verliezen koppelen terug naar de woning

Als de tank in de geconditioneerde zone staat en de verliezen thermisch bijdragen aan de woning, dan is block-diagonale ontkoppeling fysisch onjuist.

Gebruik dan:

$$
Q_{tank \rightarrow room}
=
\beta_{tank \rightarrow room}
\left(
\dot{Q}_{loss,top} + \dot{Q}_{loss,bot}
\right)
$$

met:

$$
0 \le \beta_{tank \rightarrow room} \le 1
$$

Speciaal geval:

- als `T_amb_tank = T_r`, dan ontstaan extra terugkoppelingen van `T_top` en `T_bot` naar de ruimtebalans en is de totale `A`-matrix niet block-diagonaal

Bindende regel:

- de implementatie moet exact één van beide varianten kiezen
- variant A en variant B mogen niet impliciet door elkaar gebruikt worden

Extra bindende verduidelijking:

- in de indoor-coupled variant met `T_amb_tank = T_r` is `T_amb_tank` geen exogene verstoring meer
- die term moet dan als state-koppeling in de systeemdynamica worden opgenomen
- het is verboden dezelfde grootheid tegelijk als exogene verstoring en als state-afhankelijke koppeling te modelleren

## 6. Warmtepomp, COP en MPC

### 6.1 COP-precalculatie

Om het optimalisatieprobleem convex te houden, worden `COP_ufh[k]` en `COP_dhw[k]` vóór de solverstap voorgecalculeerd uit exogene voorspellingen en configuratieparameters.

Definieer:

$$
T_{cond,ufh,K}[k]
=
T_{supply,ufh,C}[k]
+ \Delta T_{cond,pinch,ufh}
+ temperature\_offset\_c\_to\_k
$$

$$
T_{cond,dhw,K}[k]
=
T_{supply,dhw,C}[k]
+ \Delta T_{cond,pinch,dhw}
+ temperature\_offset\_c\_to\_k
$$

$$
T_{evap,K}[k]
=
T_{out}[k]
- \Delta T_{evap,pinch}
+ temperature\_offset\_c\_to\_k
$$

$$
COP_{mode}[k]
=
\eta_{carnot,mode}
\frac{T_{cond,mode,K}[k]}{T_{cond,mode,K}[k] - T_{evap,K}[k]}
$$

Fail-fast validaties:

- `T_cond,mode,K > T_evap,K`
- `T_evap,K > 0`
- `COP_mode[k] > cop_min_physical`
- `COP_mode[k] <= cop_max`

Belangrijke modelbeperking:

- deze COP hangt niet af van de actuele optimalisatiestates
- daardoor onderschat de MPC de werkelijke kosten wanneer de vereiste sinktemperatuur stijgt, vooral bij DHW op hoge tanktemperaturen
- dit is alleen toegestaan als expliciete modelkeuze voor convexiteit

### 6.2 Hydraulische topologie van de warmtepomp

De volgende twee installatietypen mogen niet verward worden:

1. gelijktijdige levering mogelijk
2. UFH en DHW zijn wederzijds exclusieve modi via een driewegklep of vergelijkbare topologie

De elektrische somconstraint alleen:

$$
\frac{P_{ufh}[k]}{COP_{ufh}[k]}
+
\frac{P_{dhw}[k]}{COP_{dhw}[k]}
\le
P_{hp,max,elec}
$$

is fysisch alleen correct voor type `1`.

Voor type `2` geldt:

- gelijktijdig `P_ufh > 0` en `P_dhw > 0` is fysiek niet toegestaan
- omdat de solverkeuze CVXPY + OSQP een convex QP vereist, moet een bovenliggende supervisor per sample of per horizon de modus vastleggen
- die supervisor zet ofwel `P_ufh,max_available = 0` ofwel `P_dhw,max_available = 0`

De MPC mag exclusieve hardware niet modelleren alsof alleen een vermogenssom geldt.

### 6.3 Mode-switch en ramp-rate consistentie

Bij exclusieve topologie kunnen ramp-rate constraints conflicteren met een plotselinge modeselectie. Daarom geldt bindend:

- de supervisor moet bij een moduswissel expliciet definiëren hoe de ramp-constraints worden behandeld
- toegestane beleidskeuzes zijn:
  1. aangepaste `P_prev` op de switch-stap
  2. expliciete switch-slack
  3. supervisorlogica die alleen fysisch haalbare mode-transities toelaat

Het is niet toegestaan om:

- enerzijds `P_ufh,max_available = 0` of `P_dhw,max_available = 0` af te dwingen
- en anderzijds een ongewijzigde ramp-constraint te laten staan die daardoor infeasible wordt

### 6.4 Kostfunctie met consistente euro-eenheden

Alle stage-kosten moeten euro-dimensie hebben. Daarom wordt `\Delta t` expliciet meegenomen in alle termen die een continue grootheid per tijdseenheid representeren.

De totale kost:

$$
J
=
\sum_{k=0}^{N-1}
\Delta t
\Bigg[
w_{room,comfort}
\left(T_r[k] - T_{ref}[k]\right)^2
+
p[k]\frac{P_{ufh}[k]}{COP_{ufh}[k]}
+
w_{ufh,power} P_{ufh}[k]^2
+
w_{ufh,slack}\,\epsilon_{ufh}[k]^2
+
p[k]\frac{P_{dhw}[k]}{COP_{dhw}[k]}
+
w_{dhw,slack}\,\epsilon_{dhw}[k]^2
\Bigg]
+
w_{room,terminal}
\left(T_r[N] - T_{ref}[N]\right)^2
$$

Eenheden:

- `w_room,comfort`: `€/(K²·h)`
- `w_ufh,power`: `€/(kW²·h)`
- `w_ufh,slack`: `€/(K²·h)`
- `w_dhw,slack`: `€/(K²·h)`
- `w_room,terminal`: `€/K²`

De energietermen zijn de enige directe eurokosten van elektriciteitsgebruik. De overige termen zijn expliciete ontwerpgewichten in euro-equivalent, niet verborgen dimensieloze straftermen.

### 6.5 DHW terminalgedrag

Om end-of-horizon depletie van de tank te voorkomen, moet de implementatie minstens één expliciete DHW-terminalstrategie ondersteunen.

Toegestane opties:

1. terminal lower bound op `T_top[N]`
2. terminal lower bound op `T_dhw_energy[N]`
3. terminal penalty op relatieve energie-inhoud
4. supervisorlogica die forecasted draws buiten de horizon meeneemt

Bindende regel:

- de productie-implementatie mag de tank niet impliciet "leeg optimaliseren" aan het horizon-einde zonder dat dit een expliciete ontwerpkeuze is
- als geen DHW-terminalstrategie actief is, moet dit als bewuste modelbeperking gedocumenteerd zijn

### 6.6 Constraints

UFH-actuator:

$$
0 \le P_{ufh}[k] \le P_{ufh,max,available}[k]
$$

UFH-soft-comfort:

$$
T_{room,min}[k] - \epsilon_{ufh}[k]
\le
T_r[k]
\le
T_{room,max}[k] + \epsilon_{ufh}[k]
$$

$$
\epsilon_{ufh}[k] \ge 0
$$

UFH-ramp-rate:

$$
\left|P_{ufh}[k] - P_{ufh}[k-1]\right|
\le
\Delta P_{ufh,max}
$$

DHW-actuator:

$$
0 \le P_{dhw}[k] \le P_{dhw,max,available}[k]
$$

DHW-soft-comfort voor tapbeschikbaarheid:

$$
T_{top}[k]
\ge
T_{dhw,draw,min}[k] - \epsilon_{dhw}[k]
$$

$$
\epsilon_{dhw}[k] \ge 0
$$

DHW-ramp-rate:

$$
\left|P_{dhw}[k] - P_{dhw}[k-1]\right|
\le
\Delta P_{dhw,max}
$$

Harde veiligheidstemperaturen:

$$
T_r[k] \ge T_{state,min,physical},\quad
T_b[k] \ge T_{state,min,physical},\quad
T_{top}[k] \ge T_{state,min,physical},\quad
T_{bot}[k] \ge T_{state,min,physical}
$$

$$
T_r[k] \le T_{room,max,safe},\quad
T_b[k] \le T_{floor,max,safe},\quad
T_{top}[k] \le T_{dhw,max,safe},\quad
T_{bot}[k] \le T_{dhw,max,safe}
$$

Eventuele elektrische en thermische vermogensgrenzen:

$$
\frac{P_{ufh}[k]}{COP_{ufh}[k]}
+
\frac{P_{dhw}[k]}{COP_{dhw}[k]}
\le
P_{hp,max,elec}
$$

$$
P_{ufh}[k] + P_{dhw}[k]
\le
P_{hp,max,therm}
$$

### 6.7 Vorige stap als expliciete MPC-parameter

Omdat ramp-rate constraints `P[k-1]` gebruiken, moet de MPC-probleemdefinitie expliciet de parameters bevatten:

- `P_ufh_prev`
- `P_dhw_prev`

Voor `k=0` in de horizon wordt dus niet impliciet een default aangenomen.

### 6.8 Legionella: fysisch correcte 2-node surrogate

Alleen `T_top >= T_leg` afdwingen is niet voldoende om een volledige stratificatietank te representeren. In een tweelaags model is een fysisch strengere surrogate:

$$
T_{top}[k] \ge T_{leg,target}
\quad \land \quad
T_{bot}[k] \ge T_{leg,target}
$$

gedurende minimaal:

$$
n_{leg,hold}
=
\left\lceil
\frac{t_{leg,hold,min}}{\Delta t}
\right\rceil
$$

aaneengesloten stappen.

Bindende regel:

- de legionella-logica wordt beheerd door een bovenliggende supervisor
- die supervisor plant het blok binnen de horizon of forceert een reeks opeenvolgende horizons
- de standaardcomfortconstraint `T_dhw,draw,min` is geen vervanging voor de legionella-eis

## 7. Configuratie en validatie

### 7.1 Pydantic-model

Alle parameters komen uit een gevalideerd configuratiemodel. De implementatie heeft minimaal de volgende domeinen:

- `PhysicalConstantsConfig`
- `DiscretizationConfig`
- `UfhConfig`
- `DhwConfig`
- `HeatPumpConfig`
- `EstimatorConfig`
- `MpcConfig`
- `SupervisorConfig`

Naamgevingsregel voor configuratievelden:

- configuratievelden gebruiken consequent `snake_case`
- eenzelfde fysische of numerieke grootheid mag niet met meerdere naamstijlen door elkaar worden benoemd
- documentatie, configmodellen, serialisatie en tests gebruiken voor dezelfde parameter exact dezelfde veldnaam

Voor numerieke observeerbaarheidsdiagnostiek bevat `EstimatorConfig` minimaal benoemde drempels zoals:

- `observability_rank_tolerance`
- exact één van de volgende conditioneringsbeleidskeuzes:
  - `observability_condition_min_sv`
  - `observability_condition_max`

Bindende regel:

- een deployment of configuratie kiest exact één conditioneringsbeleid
- `observability_condition_min_sv` en `observability_condition_max` mogen niet tegelijk actief zijn

### 7.2 Verplichte validatieregels

#### Statische validatie

- `delta_t > 0`
- `C_r > 0`, `C_b > 0`, `C_top > 0`, `C_bot > 0`
- `R_br > 0`, `R_ro > 0`, `R_strat > 0`, `R_loss_top > 0`, `R_loss_bot > 0`
- `0 <= alpha_solar <= 1`
- `0 <= eta_window <= 1`
- `0 <= heater_split_top <= 1`
- `0 <= heater_split_bottom <= 1`
- `abs(heater_split_top + heater_split_bottom - 1) <= split_sum_tolerance`
- `rho_water_ref > 0`
- `cp_water_ref > 0`
- `joules_per_kwh > 0`
- `abs(lambda_water_ref - rho_water_ref * cp_water_ref / joules_per_kwh) <= lambda_consistency_tolerance`
- `V_tank_active > 0`
- `C_tank_parasitic >= 0`
- `capacity_balance_residual <= capacity_balance_tolerance`
- `Q_ufh` en `Q_dhw_aug` symmetrisch positief semidefiniet
- `R_ufh` en `R_dhw` symmetrisch positief definiet
- initiële temperaturen `> absolute_zero_celsius`
- `Vdot_tap_init >= 0`
- `T_room_max_safe > T_state_min_physical`
- `T_floor_max_safe > T_state_min_physical`
- `T_dhw_max_safe > T_state_min_physical`
- `T_dhw_draw_min <= T_dhw_max_safe`
- `T_leg_target <= T_dhw_max_safe`
- `cop_min_physical > 1`
- `cop_max > cop_min_physical`
- `eta_carnot_mode > 0`

#### Runtime-validatie

Voor `forward_euler`:

- `rho(A_d^{Euler}) < 1`
- relevante diagonale zelfcoëfficiënten niet negatief volgens sectie 2.6
- voor DHW: stabiliteitscheck op actuele of conservatieve worst-case `Vdot_tap`

Observeerbaarheid:

- UFH: numerieke rang én conditioneringscheck na parametrisatie
- 2-state DHW-model: numerieke rang én conditioneringscheck na parametrisatie
- augmented EKF: lokale observeerbaarheidsdiagnostiek; geen hard fail op tijdelijke operationele degeneratie

Voor numerieke rangbepaling geldt bindend:

- de numerieke rang wordt bepaald via SVD
- de gebruikte tolerantiedrempel is `observability_rank_tolerance`
- de implementatie gebruikt dezelfde rangdefinitie consistent in validatie, diagnostiek en tests

Voor conditioneringschecks geldt bindend:

- de implementatie moet een configureerbare numerieke drempel gebruiken
- die drempel moet normatief uit config komen en mag niet impliciet in code verstopt zijn
- de implementatie gebruikt ofwel een minimale singuliere waarde met drempel `observability_condition_min_sv`, ofwel een maximale conditiegetaldrempel `observability_condition_max`
- exact één van beide beleidskeuzes is actief per deployment of configuratie
- de gekozen maat en drempel moeten consistent in validatie, diagnostiek en tests worden gebruikt

Voor PSD-validatie van covarianties geldt bindend:

- PSD-validatie gebeurt numeriek
- kleine negatieve eigenwaarden door floating-point afronding mogen worden getolereerd
- de validatieregel luidt dat de kleinste eigenwaarde ten minste `-covariance_psd_tolerance` moet zijn
- dezelfde PSD-definitie wordt consistent gebruikt in runtime-validatie en tests

Koppeling en topologie:

- als `tank_loss_mode = indoor_coupled`, dan mag block-diagonale systeemopbouw niet gebruikt worden
- als `heat_pump_topology = exclusive`, dan moeten `P_ufh,max_available` en `P_dhw,max_available` door de supervisor worden gezet
- de MPC mag exclusieve hardware niet modelleren alsof simultane productie via alleen een vermogenssom is toegestaan
- als een exclusieve mode-switch plaatsvindt, dan moet de ramp-policy expliciet zijn

## 8. Testeisen

De implementatie moet een `pytest`-suite genereren die minimaal de volgende fysische en numerieke controles bevat.

### 8.1 Energie en eenheden

- `test_ufh_energy_balance`
  - controleert:

$$
\frac{d}{dt}(C_r T_r + C_b T_b)
=
P_{ufh} - \frac{T_r - T_{out}}{R_{ro}} + Q_{solar} + Q_{int,eff}
$$

- `test_dhw_energy_balance`
  - controleert:

$$
\frac{d}{dt}(C_{top} T_{top} + C_{bot} T_{bot})
=
P_{dhw}
- \lambda_{water,ref} \dot{V}_{tap}(T_{top} - T_{mains})
- \dot{Q}_{loss,top}
- \dot{Q}_{loss,bot}
$$

- `test_lambda_consistency`
  - controleert:

$$
\lambda_{water,ref}
=
\rho_{water,ref}\, c_{p,water,ref} / joules\_per\_kwh
$$

- `test_capacity_balance`
  - controleert:

$$
C_{top} + C_{bot}
=
\lambda_{water,ref} V_{tank,active} + C_{tank,parasitic}
$$

  binnen `capacity_balance_tolerance`

### 8.2 Filters

- `test_kalman_covariance_psd`
  - `P_ufh` blijft symmetrisch positief semidefiniet binnen `covariance_psd_tolerance`
  - de kleinste eigenwaarde voldoet numeriek aan `lambda_min(P_ufh) >= -covariance_psd_tolerance`
- `test_ekf_covariance_psd`
  - `P_dhw_aug` blijft symmetrisch positief semidefiniet binnen `covariance_psd_tolerance`
  - de kleinste eigenwaarde voldoet numeriek aan `lambda_min(P_dhw_aug) >= -covariance_psd_tolerance`
- `test_ekf_vtap_nonnegative`
  - controleert de verplichte post-update clamp
- `test_ekf_vtap_detection`
  - gesimuleerde tapgebeurtenis moet binnen `vtap_convergence_steps` convergeren tot binnen `vtap_convergence_tolerance`
- `test_ekf_no_tap_zero_when_observable`
  - zonder tap, maar met een observeerbare temperatuurgradiënt, moet `Vdot_tap_hat` naar nul relaxeren binnen `vtap_zero_tolerance`
- `test_ekf_unobservable_zero_gradient_case_is_bounded`
  - als `T_top = T_bot = T_mains`, dan mag de filter geen onterechte observeerbaarheid aannemen; de schatting moet niet-negatief en numeriek begrensd blijven
- `test_ekf_jacobian_eval_point`
  - verifieert dat de Jacobiaan geëvalueerd is op het pre-predictiepunt via finite differences
- `test_projected_ekf_clamp_is_forwarded_to_mpc`
  - controleert dat de geclamte `Vdot_tap_hat` en niet de ongeclampte estimate naar de MPC gaat

### 8.3 Observeerbaarheid en modelkeuze

- `test_ufh_observability_rank`
- `test_dhw_observability_rank`
- `test_ufh_observability_conditioning`
- `test_dhw_observability_conditioning`
- `test_observability_condition_threshold_is_config_driven`
- `test_ekf_augmented_local_observability_when_gradient_present`
  - gebruikt een toestand met `T_top != T_bot` of `T_bot != T_mains`
- `test_block_diagonal_forbidden_when_tank_losses_couple_to_room`
- `test_exclusive_heat_pump_topology_requires_supervisor_mode`
- `test_exclusive_mode_switch_has_ramp_policy`

### 8.4 Euler-validatie

- `test_forward_euler_spectral_radius_validation`
- `test_forward_euler_self_damping_nonnegative_ufh`
- `test_forward_euler_self_damping_nonnegative_dhw`
- `test_dhw_forward_euler_worst_case_vtap_validation`

### 8.5 MPC

- `test_cop_validation`
  - gooit `ValidationError` bij onfysische COP
- `test_mpc_feasibility_nominal`
  - standaardscenario moet oplosbaar zijn
- `test_legionella_surrogate_uses_both_nodes`
  - de supervisor mag niet alleen `T_top` gebruiken
- `test_temperature_safety_bounds_enforced`
  - harde temperatuurgrenzen moeten in het probleem aanwezig zijn
- `test_mpc_requires_explicit_previous_input_parameters`
  - `P_ufh_prev` en `P_dhw_prev` moeten expliciete parameters zijn
- `test_dhw_terminal_strategy_present_or_explicitly_disabled`
  - DHW-terminalgedrag moet aanwezig zijn of als bewuste beperking geconfigureerd staan

## 9. Aanbevolen softwarestructuur

De code moet stateful, generiek en herbruikbaar zijn.

### 9.1 Model-laag

- `UfhContinuousModel`
- `DhwContinuousModel`
- `CoupledThermalModel`

Elk model levert minimaal:

- continue matrices of continue ODE-callbacks
- afgeleide discrete modellen via een gedeelde `Discretizer`
- energiebalansfuncties voor tests
- observeerbaarheidscontrole
- Euler-admissibiliteitscontrole waar van toepassing

### 9.2 Estimator-laag

- `LinearKalmanFilter`
- `ExtendedKalmanFilter`

De EKF erft de Joseph-update en innovatie-algebra van de lineaire filterklasse. Alleen de predictiestap en Jacobiaanprovider verschillen.

De estimator-laag ondersteunt expliciet:

- discrete `Q`-semantiek
- projected state updates
- observability diagnostics
- covariance symmetrization of numerieke regularisatie indien toegepast

### 9.3 Control-laag

- `CopPrecalculator`
- `HeatPumpTopologySupervisor`
- `LegionellaSupervisor`
- `MpcProblemBuilder`

De supervisoren bepalen de fysisch toelaatbare constraint-set voordat het CVXPY-probleem wordt opgebouwd.

`MpcProblemBuilder` krijgt expliciet als inputs:

- huidige state estimates
- forecastreeksen
- `P_ufh_prev`
- `P_dhw_prev`
- topology mode of availability masks
- legionella supervisory constraints
- DHW terminal policy

## 10. Bekende modelbeperkingen

Deze beperkingen zijn toegestaan, maar moeten expliciet in code en documentatie terugkomen:

- UFH is een lumped two-state model en vangt geen ruimtelijke temperatuurgradiënten in de vloer.
- DHW gebruikt een two-node stratificatiemodel en vangt geen fijnere thermocline-structuur.
- `R_strat` is empirisch en niet afleidbaar uit first principles.
- Een vaste `lambda_water_ref` veronderstelt temperatuur-onafhankelijke watereigenschappen.
- COP-precalculatie houdt de optimalisatie convex, maar veroorzaakt dispatch-fout bij hoge sinktemperaturen.
- Een convex QP met continue vermogensvariabelen veronderstelt dat de installatie effectief moduleerbaar is binnen het toegestane bereik.
- De projected-EKF clamp op `Vdot_tap` is fysisch wenselijk maar statistisch een heuristische projectiestap.
- Tijdelijke lokale onobserveerbaarheid van `Vdot_tap` kan optreden en is niet per definitie een modeldefect.
- Zonder expliciete terminalstrategie kan DHW-MPC horizon-einde artefacten vertonen.

## 11. Samenvatting van bindende correcties in deze versie

Deze versie legt de volgende fysische, numerieke en softwarematige correcties vast:

- continue fysica is normatief; Euler is slechts een expliciete discretisatievariant
- energiebalansen mogen affine energiestates gebruiken; absolute energie vereist een referentietemperatuur
- kostengewichten hebben euro-consistente eenheden
- `Q_ufh` en `Q_dhw_aug` zijn positief semidefiniet; alleen meetruis is verplicht positief definiet
- `R_loss_top` en `R_loss_bot` zijn gescheiden
- `C_top + C_bot = lambda * V_tank` is niet universeel waar; parasitaire thermische massa is expliciet toegevoegd
- de augmented observeerbaarheidsvoorwaarde voor `Vdot_tap` is gecorrigeerd naar `T_top != T_bot` of `T_bot != T_mains`
- tijdelijke lokale onobserveerbaarheid van `Vdot_tap` is een operationele toestand, geen harde configuratiefout
- observeerbaarheid wordt niet alleen via rang maar ook via numerieke conditionering beoordeeld
- block-diagonale UFH/DHW-koppeling is alleen toegestaan onder expliciete tankverliesaanname
- in indoor-coupled modus met `T_amb_tank = T_r` is `T_amb_tank` geen exogene verstoring meer
- een warmtepomp met exclusieve UFH/DHW-modi mag niet met alleen een vermogenssomconstraint gemodelleerd worden
- exclusieve mode-switches moeten expliciet ramp-rate consistent worden behandeld
- de legionella-surrogaatconstraint gebruikt beide tanklagen, niet alleen `T_top`
- DHW-MPC moet een terminalstrategie ondersteunen of dit expliciet als beperking documenteren
- harde veiligheidstemperaturen zijn verplicht
- de EKF-clamp op `Vdot_tap` is verplicht en de geclamte waarde moet naar de MPC gaan
- `P_ufh_prev` en `P_dhw_prev` zijn expliciete MPC-parameters
- `forward_euler` vereist naast spectrale stabiliteit ook expliciete checks op fysisch monotone zelfdemping

## 12. Leidende regel bij conflict

Dit document is leidend.

Als code, configuratie, filters, supervisorlogica, optimizer-opbouw of tests hiermee in strijd zijn, dan moet de implementatie worden aangepast en niet het document.
