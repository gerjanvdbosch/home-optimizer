# Home Optimizer: Thermisch Model (UFH + DHW + EKF + MPC)

Dit document is de bindende implementatiespecificatie voor de thermische regeling van Home Optimizer. Het beschrijft een grey-box model voor ruimteverwarming via vloerverwarming (UFH), een gelaagde warmwatertank (DHW), toestandschatting met een lineair Kalman-filter en een Extended Kalman Filter (EKF), en optimalisatie met een Model Predictive Controller (MPC).

De kernregel is niet "zo realistisch mogelijk", maar: binnen de expliciet gekozen modelaannames moet elke vergelijking intern consistent, dimensioneel correct en numeriek verantwoord zijn. Een grey-box model is nooit universeel exact; het moet daarom zijn aannames expliciet maken en mag niets impliciet verzwijgen.

## 1. Niet-onderhandelbare eisen

### 1.1 Fysica eerst

- De continue fysica is normatief. De discrete implementatie is altijd een afleiding van de continue vergelijkingen, nooit andersom.
- Temperatuurverschillen mogen in graden Celsius worden uitgedrukt omdat een verschil in `°C` numeriek gelijk is aan een verschil in `K`.
- Vermogen is altijd thermisch of elektrisch in `kW`, energie in `kWh`, temperatuur in `°C`, volume in `m³`, volumestroom in `m³/h`, tijd in `h`.
- Elke vereenvoudiging moet als modelaanname benoemd zijn. Als een aanname niet geldt voor de echte installatie, moet de code die vereenvoudigde modelvariant weigeren.

### 1.2 Geen magic numbers

- Elke fysische constante, veiligheidstemperatuur, tolerantie, tijdstap, solver-weight en validatiegrens komt uit een gevalideerd configuratie-object.
- Zelfs universele of quasi-universele constanten krijgen een naam, bijvoorbeeld:
  - `temperature_offset_c_to_k`
  - `absolute_zero_celsius`
  - `joules_per_kwh`
  - `rho_water_ref`
  - `cp_water_ref`
  - `lambda_water_ref`
- De code gebruikt geen losse floats in wiskundige logica behalve `0` en `1`.

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
- UFH en DHW delen dezelfde algebra: state-space, Joseph-update, observability-check, discretisatie, constraints. Kopieer die logica niet.

### 1.4 Fail-fast

- Ontbreekt een parameter of tijdreeks, dan wordt een exception gegooid. Er zijn geen verborgen defaults.
- Elke fysisch onmogelijke waarde wordt geweigerd voordat zij een model, filter of solver bereikt.
- Voorbeelden van harde blokkades:
  - warmtecapaciteit `<= 0`
  - thermische weerstand `<= 0`
  - `alpha_solar` buiten `[0, 1]`
  - `eta_window` buiten `[0, 1]`
  - `heater_split_top + heater_split_bottom != 1`
  - temperatuur `< absolute_zero_celsius`
  - `lambda_water_ref <= 0`
  - `cop <= cop_min_physical`
  - `cop > cop_max`
  - negatieve geclampt doorgegeven tapstroom

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

$$[k \Delta t,\ (k+1)\Delta t)$$

met sampletijd:

$$\Delta t > 0$$

### 2.2 Water-eigenschappen

De basisvariant gebruikt temperatuur-onafhankelijke referentie-eigenschappen van vloeibaar water:

$$\lambda_{water,ref} = \frac{\rho_{water,ref}\, c_{p,water,ref}}{joules\_per\_kwh} \quad \left[\frac{kWh}{m^3 \cdot K}\right]$$

Belangrijk:

- Dit is een modelaanname, geen natuurwet. In werkelijkheid hangen `rho` en `c_p` van temperatuur af.
- Als temperatuurafhankelijke watereigenschappen nodig zijn, moet `lambda_water_ref` worden vervangen door een expliciete functie `lambda_water(T)`. De rest van de architectuur blijft gelijk.
- Gebruik nooit rechtstreeks `c_p` in `J/(kg·K)` in de toestandsvergelijkingen. Converteer eerst naar `lambda_water_ref` of `lambda_water(T)`.

### 2.3 Discretisatiebeleid

De productie-implementatie ondersteunt precies twee schema's:

1. `exact_zoh`
2. `forward_euler`

Regels:

- `exact_zoh` is de standaard voor lineaire modellen en lineair bevroren LTV-stappen.
- `forward_euler` mag alleen gebruikt worden als een runtime-validator aantoont dat de discrete stap voor de actuele parameters numeriek stabiel is en geen onfysische tekenomkeringen introduceert.
- De code mag nooit impliciet van schema wisselen.

Voor een continu lineair model

$$\dot{x}(t) = A_c x(t) + B_c u(t) + E_c d(t)$$

met stukgewijs constante `u` en `d` over één sample, is de exacte ZOH-discretisatie:

$$x[k+1] = A_d x[k] + B_d u[k] + E_d d[k]$$

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

$$A_d^{Euler} = I + \Delta t\,A_c,\qquad B_d^{Euler} = \Delta t\,B_c,\qquad E_d^{Euler} = \Delta t\,E_c$$

en de runtime-validator moet minimaal controleren:

$$\rho\!\left(A_d^{Euler}\right) < 1$$

waar `rho(.)` de spectrale straal is. Voor LTV-systemen gebeurt deze check op elke stap van de horizon of op een bewezen conservatieve worst-case.

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

$$Q_{int,eff} = Q_{int,base} + Q_{tank \rightarrow room}$$

waarbij `Q_{tank -> room}` nul is in de ontkoppelde variant en expliciet wordt gedefinieerd in sectie 5.2 wanneer boilerverliezen in de woning terechtkomen.

### 3.2 Zonmodel

Het lineaire venstermodel luidt:

$$Q_{solar}[k] = \frac{A_{glass,eff}\, GTI_{window}[k]\, \eta_{window}}{power\_unit\_scale}$$

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

$$x_{ufh} = \begin{bmatrix} T_r \\ T_b \end{bmatrix},\qquad
u_{ufh} = \begin{bmatrix} P_{ufh} \end{bmatrix},\qquad
d_{ufh} = \begin{bmatrix} T_{out} \\ Q_{solar} \\ Q_{int,eff} \end{bmatrix}$$

Dan geldt:

$$\dot{x}_{ufh} = A_{c,ufh}\,x_{ufh} + B_{c,ufh}\,u_{ufh} + E_{c,ufh}\,d_{ufh}$$

met:

$$
A_{c,ufh} =
\begin{bmatrix}
-\left(\frac{1}{C_r R_{br}} + \frac{1}{C_r R_{ro}}\right) & \frac{1}{C_r R_{br}} \\
\frac{1}{C_b R_{br}} & -\frac{1}{C_b R_{br}}
\end{bmatrix}
$$

$$
B_{c,ufh} =
\begin{bmatrix}
0 \\
\frac{1}{C_b}
\end{bmatrix}
$$

$$
E_{c,ufh} =
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

### 3.5 Discrete vorm

De implementatie moet `A_d`, `B_d` en `E_d` afleiden uit `A_c`, `B_c` en `E_c` met het gekozen schema.

Alleen als `forward_euler` expliciet is ingeschakeld, mag onderstaande referentievorm gebruikt worden:

$$a_{br} = \frac{\Delta t}{C_r R_{br}},\qquad
a_{ro} = \frac{\Delta t}{C_r R_{ro}},\qquad
b_{br} = \frac{\Delta t}{C_b R_{br}}$$

$$
A_{d,ufh}^{Euler} =
\begin{bmatrix}
1 - a_{br} - a_{ro} & a_{br} \\
b_{br} & 1 - b_{br}
\end{bmatrix}
$$

$$
B_{d,ufh}^{Euler} =
\begin{bmatrix}
0 \\
\frac{\Delta t}{C_b}
\end{bmatrix}
$$

$$
E_{d,ufh}^{Euler} =
\begin{bmatrix}
a_{ro} & \alpha_{solar}\frac{\Delta t}{C_r} & \frac{\Delta t}{C_r} \\
0 & (1-\alpha_{solar})\frac{\Delta t}{C_b} & 0
\end{bmatrix}
$$

### 3.6 Observeerbaarheid

De UFH-meetmatrix is:

$$C_{obs,ufh} = \begin{bmatrix} 1 & 0 \end{bmatrix}$$

Na parametrisatie moet de code de rang numeriek controleren van:

$$
\mathcal{O}_{ufh}
=
\begin{bmatrix}
C_{obs,ufh} \\
C_{obs,ufh} A_{d,ufh}
\end{bmatrix}
$$

Bij de Euler-vorm reduceert de determinant tot een term evenredig met `b_br`; het systeem is dus observeerbaar zolang `C_b`, `R_br` en `delta_t` fysisch geldig en eindig zijn.

### 3.7 Lineair Kalman-filter

Meetmodel:

$$y_{ufh}[k] = C_{obs,ufh}\, x_{ufh}[k] + v[k]$$

met:

- `Q_ufh` discrete procesruiscovariantie, symmetrisch positief semidefiniet
- `R_ufh` discrete meetruiscovariantie, symmetrisch positief definiet

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

### 4.2 States, input, verstoringen en afgeleide grootheden

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

$$u_{dhw} = \begin{bmatrix} P_{dhw} \end{bmatrix}$$

Verstoringen voor de MPC:

$$
d_{dhw}
=
\begin{bmatrix}
T_{amb,tank} \\
T_{mains}
\end{bmatrix}
$$

Heater-split:

$$heater\_split_{top} + heater\_split_{bottom} = 1$$

met:

$$0 \le heater\_split_{top} \le 1,\qquad 0 \le heater\_split_{bottom} \le 1$$

Afgeleide signalen:

- opgeslagen thermische energie in modeltoestand:

$$E_{dhw,state} = C_{top} T_{top} + C_{bot} T_{bot}$$

- energie-gewogen gemiddelde toestandstemperatuur:

$$
T_{dhw,energy}
=
\frac{C_{top}T_{top} + C_{bot}T_{bot}}{C_{top} + C_{bot}}
$$

Belangrijk:

- `T_dhw_energy` is alleen een energie-gewogen state-gemiddelde.
- Het is niet automatisch gelijk aan de volume-gemiddelde watertemperatuur.
- Gebruik `T_top` voor tapcomfort en een expliciete legionella-surrogaatconstraint voor hygiëne.

### 4.3 Continue fysica

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

### 4.4 Capaciteitsconservatie

De code mag niet blind afdwingen:

$$C_{top} + C_{bot} = \lambda_{water,ref} V_{tank}$$

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

$$C_{tank,parasitic} = 0$$

### 4.5 Continue LTV-vorm voor de MPC

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

De discrete matrices `A_dhw[k]`, `B_dhw[k]`, `E_dhw[k]` moeten daaruit met het gekozen schema worden afgeleid.

### 4.6 Forward-Euler referentievorm

Alleen voor `forward_euler`:

$$a_{strat} = \frac{\Delta t}{C_{top}R_{strat}},\qquad
b_{strat} = \frac{\Delta t}{C_{bot}R_{strat}}$$

$$a_{loss} = \frac{\Delta t}{C_{top}R_{loss,top}},\qquad
b_{loss} = \frac{\Delta t}{C_{bot}R_{loss,bot}}$$

$$a_{tap}[k] = \frac{\Delta t\,\lambda_{water,ref}\hat{\dot{V}}_{tap}[k]}{C_{top}},\qquad
b_{tap}[k] = \frac{\Delta t\,\lambda_{water,ref}\hat{\dot{V}}_{tap}[k]}{C_{bot}}$$

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

### 4.7 Observeerbaarheid van het 2-state MPC-model

Als alleen `T_top` gemeten zou worden, is de Euler-vorm observeerbaar zolang:

$$a_{strat} + a_{tap}[k] \ne 0$$

In de feitelijke implementatie met ZOH of met beide sensoren beschikbaar moet de code de rang numeriek controleren na parametrisatie, niet via een los handmatig argument.

### 4.8 Extended Kalman Filter

#### 4.8.1 Meetmodel

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

`R_dhw` is een `2 x 2` symmetrisch positief definiete matrix.

#### 4.8.2 Random-walk model voor het debiet

$$
\dot{V}_{tap}[k+1]
=
\dot{V}_{tap}[k] + w_{\dot{V}}[k]
$$

met `Q_vtap` als discrete procesruisvariantie.

#### 4.8.3 Discrete niet-lineaire procesfunctie

De EKF-basisklasse accepteert een discrete transitiecallback:

$$x[k+1] = f_d(x[k], u[k], d[k]) + w[k]$$

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

Als de implementatie een andere discrete map gebruikt, bijvoorbeeld een hogere-orde integrator, dan moet ook de Jacobiaan bij diezelfde discrete map horen. Het is fysisch en numeriek onjuist om een Euler-Jacobiaan te combineren met een niet-Euler predictiestap.

#### 4.8.4 Jacobiaan van de Euler-procesmap

De Jacobiaan moet geëvalueerd worden op het punt:

$$
\left(\hat{x}_{dhw,aug}[k-1],\,u_{dhw}[k-1],\,d_{dhw}[k-1]\right)
$$

dus strikt voor de predictiestap.

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

#### 4.8.5 Observeerbaarheid van het augmented systeem

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

Dus lokaal geldt:

$$
rank(\mathcal{O}_{dhw,aug}) = 3
\quad \Longleftarrow \quad
\left(T_{top} \ne T_{bot}\right)
\ \lor\
\left(T_{bot} \ne T_{mains}\right)
$$

De oude vuistregel "observeerbaar zolang `T_top != T_mains`" is onvoldoende precies en mag niet gebruikt worden.

#### 4.8.6 EKF-algoritme

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

Harde fysische projectie na elke update:

$$
\hat{\dot{V}}_{tap}[k]
\leftarrow
\max\!\left(0,\ \hat{x}_{dhw,aug}[k]_3\right)
$$

Verplicht:

- de clamp gebeurt na iedere update
- de geclamte waarde wordt doorgegeven aan de MPC
- de MPC mag nooit een negatieve tapstroom ontvangen

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

$$0 \le \beta_{tank \rightarrow room} \le 1$$

Speciaal geval:

- als `T_amb_tank = T_r`, dan ontstaan extra terugkoppelingen van `T_top` en `T_bot` naar de ruimtebalans en is de totale `A`-matrix niet block-diagonaal

Bindende regel:

- de implementatie moet exact één van beide varianten kiezen
- variant A en variant B mogen niet impliciet door elkaar gebruikt worden

## 6. Warmtepomp, COP en MPC

### 6.1 COP-precalculatie

Om het optimalisatieprobleem convex te houden, worden `COP_ufh[k]` en `COP_dhw[k]` voor de solverstap voorgecalculeerd uit exogene voorspellingen en configuratieparameters.

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
2. UFH en DHW zijn wederzijds exclusieve modi via een driewegklep

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

### 6.3 Kostfunctie met consistente euro-eenheden

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

### 6.4 Constraints

UFH-actuator:

$$0 \le P_{ufh}[k] \le P_{ufh,max,available}[k]$$

UFH-soft-comfort:

$$
T_{room,min}[k] - \epsilon_{ufh}[k]
\le
T_r[k]
\le
T_{room,max}[k] + \epsilon_{ufh}[k]
$$

$$\epsilon_{ufh}[k] \ge 0$$

UFH-ramp-rate:

$$
\left|P_{ufh}[k] - P_{ufh}[k-1]\right|
\le
\Delta P_{ufh,max}
$$

DHW-actuator:

$$0 \le P_{dhw}[k] \le P_{dhw,max,available}[k]$$

DHW-soft-comfort voor tapbeschikbaarheid:

$$
T_{top}[k]
\ge
T_{dhw,draw,min}[k] - \epsilon_{dhw}[k]
$$

$$\epsilon_{dhw}[k] \ge 0$$

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

### 6.5 Legionella: fysisch correcte 2-node surrogate

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

### 7.2 Verplichte validatieregels

Statische validatie:

- `delta_t > 0`
- `C_r > 0`, `C_b > 0`, `C_top > 0`, `C_bot > 0`
- `R_br > 0`, `R_ro > 0`, `R_strat > 0`, `R_loss_top > 0`, `R_loss_bot > 0`
- `0 <= alpha_solar <= 1`
- `0 <= eta_window <= 1`
- `0 <= heater_split_top <= 1`
- `0 <= heater_split_bottom <= 1`
- `heater_split_top + heater_split_bottom == 1` binnen `split_sum_tolerance`
- `rho_water_ref > 0`
- `cp_water_ref > 0`
- `joules_per_kwh > 0`
- `lambda_water_ref == rho_water_ref * cp_water_ref / joules_per_kwh` binnen `lambda_consistency_tolerance`
- `V_tank_active > 0`
- `C_tank_parasitic >= 0`
- `capacity_balance_residual <= capacity_balance_tolerance`
- `Q_ufh` en `Q_dhw_aug` symmetrisch positief semidefiniet
- `R_ufh` en `R_dhw` symmetrisch positief definiet
- initiële temperaturen `> absolute_zero_celsius`
- `Vdot_tap_init >= 0`
- `T_room,max,safe > T_state,min,physical`
- `T_floor,max,safe > T_state,min,physical`
- `T_dhw,max,safe > T_state,min,physical`
- `T_dhw,draw,min <= T_dhw,max,safe`
- `T_leg,target <= T_dhw,max,safe`
- `cop_min_physical > 1`
- `cop_max > cop_min_physical`
- `eta_carnot_mode > 0`

Runtime-validatie:

- voor `forward_euler`: `rho(A_d^{Euler}) < 1`
- voor `forward_euler` in DHW: stabiliteitscheck op de actuele of worst-case `Vdot_tap`
- observability-rank van UFH en DHW na parametrisatie
- als `tank_loss_mode = indoor_coupled`, dan mag block-diagonale systeemopbouw niet gebruikt worden
- als `heat_pump_topology = exclusive`, dan moeten `P_ufh,max_available` en `P_dhw,max_available` door de supervisor worden gezet; het model mag niet zelf simultane productie toestaan

## 8. Testeisen

De implementatie moet een `pytest`-suite genereren die minimaal de volgende fysische en numerieke controles bevat.

### 8.1 Energie en eenheden

- `test_ufh_energy_balance`
  - controleert:
    - `d/dt(C_r T_r + C_b T_b) = P_ufh - (T_r - T_out)/R_ro + Q_solar + Q_int_eff`
- `test_dhw_energy_balance`
  - controleert:
    - `d/dt(C_top T_top + C_bot T_bot) = P_dhw - lambda_water_ref * Vdot_tap * (T_top - T_mains) - Q_loss_top - Q_loss_bot`
- `test_lambda_consistency`
  - controleert:
    - `lambda_water_ref = rho_water_ref * cp_water_ref / joules_per_kwh`
- `test_capacity_balance`
  - controleert:
    - `C_top + C_bot = lambda_water_ref * V_tank_active + C_tank_parasitic` binnen `capacity_balance_tolerance`

### 8.2 Filters

- `test_kalman_covariance_spd`
  - `P_ufh` blijft symmetrisch positief definiet binnen `covariance_pd_tolerance`
- `test_ekf_covariance_spd`
  - `P_dhw_aug` blijft symmetrisch positief definiet binnen `covariance_pd_tolerance`
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

### 8.3 Observeerbaarheid en modelkeuze

- `test_ufh_observability_rank`
- `test_dhw_observability_rank`
- `test_ekf_augmented_observability_rank`
  - gebruikt een toestand met:
    - `T_top != T_bot` of `T_bot != T_mains`
- `test_block_diagonal_forbidden_when_tank_losses_couple_to_room`
- `test_exclusive_heat_pump_topology_requires_supervisor_mode`

### 8.4 MPC

- `test_cop_validation`
  - gooit `ValidationError` bij onfysische COP
- `test_mpc_feasibility_nominal`
  - standaardscenario moet oplosbaar zijn
- `test_legionella_surrogate_uses_both_nodes`
  - de supervisor mag niet alleen `T_top` gebruiken
- `test_temperature_safety_bounds_enforced`
  - harde temperatuurgrenzen moeten in het probleem aanwezig zijn

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

### 9.2 Estimator-laag

- `LinearKalmanFilter`
- `ExtendedKalmanFilter`

De EKF erft de Joseph-update en innovatie-algebra van de lineaire filterklasse. Alleen de predictiestap en Jacobiaanprovider verschillen.

### 9.3 Control-laag

- `CopPrecalculator`
- `HeatPumpTopologySupervisor`
- `LegionellaSupervisor`
- `MpcProblemBuilder`

De supervisoren bepalen de fysisch toelaatbare constraint-set voordat het CVXPY-probleem wordt opgebouwd.

## 10. Bekende modelbeperkingen

Deze beperkingen zijn toegestaan, maar moeten expliciet in code en documentatie terugkomen:

- UFH is een lumped two-state model en vangt geen ruimtelijke temperatuurgradiënten in de vloer.
- DHW gebruikt een two-node stratificatiemodel en vangt geen fijnere thermocline-structuur.
- `R_strat` is empirisch en niet afleidbaar uit first principles.
- Een vaste `lambda_water_ref` veronderstelt temperatuur-onafhankelijke watereigenschappen.
- COP-precalculatie houdt de optimalisatie convex, maar veroorzaakt dispatch-fout bij hoge sinktemperaturen.
- Een convex QP met continue vermogensvariabelen veronderstelt dat de installatie effectief moduleerbaar is binnen het toegestane bereik.

## 11. Samenvatting van expliciete correcties

Deze versie legt de volgende fysische correcties vast:

- continue fysica is normatief; Euler is slechts een expliciete gekozen discretisatievariant
- kostengewichten hebben euro-consistente eenheden
- `Q_ufh`, `Q_dhw_aug` zijn positief semidefiniet; alleen meetruis is verplicht positief definiet
- `R_loss_top` en `R_loss_bot` zijn gescheiden
- `C_top + C_bot = lambda * V_tank` is niet universeel waar; parasitaire thermische massa is expliciet toegevoegd
- de augmented observability-voorwaarde voor `Vdot_tap` is gecorrigeerd naar:
  - `T_top != T_bot` of `T_bot != T_mains`
- block-diagonale UFH/DHW-koppeling is alleen toegestaan onder een expliciete tankverliesaanname
- een warmtepomp met exclusieve UFH/DHW-modi mag niet met alleen een vermogenssomconstraint gemodelleerd worden
- de legionella-surrogaatconstraint gebruikt beide tanklagen, niet alleen `T_top`
- harde veiligheidstemperaturen zijn verplicht, zodat negatieve stroomprijzen of modelmismatch niet tot onveilige oververhitting leiden

Dit document is leidend. Als code, config of tests hiermee in strijd zijn, dan moet de implementatie aangepast worden en niet het document.
