"""FastAPI web interface for the Home Optimizer.

Endpoints
---------
GET  /               HTML single-page application
GET  /api/defaults   Default RunRequest as JSON
POST /api/optimize   Run MPC, return Plotly chart JSON

Run with:
  uvicorn home_optimizer.api:app --reload
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import plotly.graph_objects as go
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field

from .mpc import UFHMPCController
from .thermal_model import ThermalModel
from .types import ForecastHorizon, MPCParameters, ThermalParameters

# ---------------------------------------------------------------------------
# Preset forecast data
# ---------------------------------------------------------------------------

# Typical Dutch day-ahead price pattern [€/kWh], hour 0–23
_PRICES_24H = np.array(
    [
        0.21,
        0.20,
        0.19,
        0.18,
        0.19,
        0.22,  # 00–05 (cheap night)
        0.28,
        0.35,
        0.38,
        0.36,
        0.32,
        0.28,  # 06–11 (morning peak)
        0.25,
        0.24,
        0.24,
        0.25,
        0.28,
        0.35,  # 12–17 (afternoon)
        0.42,
        0.45,
        0.40,
        0.35,
        0.28,
        0.23,  # 18–23 (evening peak)
    ],
    dtype=float,
)


def _solar_gti(hour: int) -> float:
    """Simple bell-shaped GTI [W/m²] centred at solar noon (13 h)."""
    if 7 <= hour <= 19:
        return 800.0 * np.sin(np.pi * (hour - 7) / 12.0)
    return 0.0


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    """All adjustable parameters for one MPC solve."""

    # House thermal parameters
    C_r: float = Field(3.0, ge=0.5, le=50.0, description="Room thermal capacity [kWh/K]")
    C_b: float = Field(18.0, ge=1.0, le=200.0, description="Floor thermal capacity [kWh/K]")
    R_br: float = Field(2.5, ge=0.1, le=20.0, description="Floor→room resistance [K/kW]")
    R_ro: float = Field(4.0, ge=0.1, le=30.0, description="Room→outside resistance [K/kW]")
    alpha: float = Field(0.35, ge=0.0, le=1.0, description="Solar fraction to room air")

    # Initial state
    T_r_init: float = Field(20.8, ge=5.0, le=35.0, description="Initial room temperature [°C]")
    T_b_init: float = Field(24.0, ge=5.0, le=45.0, description="Initial floor temperature [°C]")
    previous_power_kw: float = Field(0.5, ge=0.0, le=20.0, description="Previous UFH power [kW]")

    # MPC settings
    horizon_hours: int = Field(24, ge=4, le=48, description="Horizon N [h, Δt = 1 h]")
    Q_c: float = Field(8.0, ge=0.0, description="Comfort weight Q_c")
    R_c: float = Field(0.05, ge=0.0, description="Regularisation weight R_c")
    P_max: float = Field(4.0, ge=0.5, le=20.0, description="Max UFH power [kW]")
    delta_P_max: float = Field(1.0, ge=0.1, le=10.0, description="Max ramp-rate [kW/step]")
    T_min: float = Field(19.0, ge=10.0, le=25.0, description="Min comfort temperature [°C]")
    T_max: float = Field(22.5, ge=16.0, le=30.0, description="Max comfort temperature [°C]")
    T_ref: float = Field(21.0, ge=15.0, le=26.0, description="Setpoint temperature [°C]")

    # Forecast
    outdoor_temperature_c: float = Field(
        10.0, ge=-20.0, le=35.0, description="Outdoor temperature [°C]"
    )
    dynamic_price: bool = Field(True, description="Use typical Dutch price pattern")
    flat_price: float = Field(0.25, ge=0.0, le=2.0, description="Flat price [€/kWh]")
    solar_gain: bool = Field(True, description="Include solar irradiance profile")
    internal_gains_kw: float = Field(0.35, ge=0.0, le=3.0, description="Internal gains [kW]")


class OptimizeResponse(BaseModel):
    status: str
    objective: float
    total_energy_kwh: float
    total_cost_eur: float
    first_power_kw: float
    max_comfort_violation_c: float
    temperature_fig: str  # Plotly figure JSON
    power_fig: str  # Plotly figure JSON


# ---------------------------------------------------------------------------
# Business logic helpers
# ---------------------------------------------------------------------------


def _build_forecast(req: RunRequest, start_hour: int) -> ForecastHorizon:
    N = req.horizon_hours
    hours = [(start_hour + k) % 24 for k in range(N)]

    prices = (
        np.array([_PRICES_24H[h] for h in hours])
        if req.dynamic_price
        else np.full(N, req.flat_price)
    )
    gti = np.array([_solar_gti(h) for h in hours], dtype=float) if req.solar_gain else np.zeros(N)
    return ForecastHorizon(
        outdoor_temperature_c=np.full(N, req.outdoor_temperature_c),
        gti_w_per_m2=gti,
        internal_gains_kw=np.full(N, req.internal_gains_kw),
        price_eur_per_kwh=prices,
        room_temperature_ref_c=np.full(N + 1, req.T_ref),
    )


def _time_labels(start_hour: int, n_points: int) -> list[str]:
    base = datetime.now(tz=timezone.utc).replace(hour=start_hour, minute=0, second=0, microsecond=0)
    return [(base + timedelta(hours=k)).strftime("%H:%M") for k in range(n_points)]


def _temperature_figure(
    labels: list[str],
    T_r: np.ndarray,
    T_ref: float,
    T_min: float,
    T_max: float,
    T_out: float,
) -> str:
    fig = go.Figure()

    # Comfort band (fill between T_min and T_max)
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=[T_max] * len(labels),
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=[T_min] * len(labels),
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(100,149,237,0.15)",
            line=dict(width=0),
            name="Comfortband",
            hoverinfo="skip",
        )
    )

    # Outdoor temperature
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=[T_out] * len(labels),
            name="T<sub>buiten</sub>",
            mode="lines",
            line=dict(color="#999", width=1.5, dash="dot"),
        )
    )

    # Setpoint
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=[T_ref] * len(labels),
            name="T<sub>ref</sub> (setpoint)",
            mode="lines",
            line=dict(color="#2ca02c", width=1.5, dash="dash"),
        )
    )

    # Predicted room temperature
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=T_r,
            name="T<sub>r</sub> (kamer)",
            mode="lines+markers",
            line=dict(color="#1e6bbf", width=2.5),
            marker=dict(size=5),
        )
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="Temperatuur [°C]", gridcolor="#f5f5f5", zeroline=False),
        xaxis=dict(gridcolor="#f5f5f5"),
        legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
    )
    return fig.to_json()


def _power_figure(
    labels: list[str],
    P_UFH: np.ndarray,
    prices: np.ndarray,
    P_max: float,
) -> str:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # UFH power bars
    fig.add_trace(
        go.Bar(
            x=labels,
            y=P_UFH,
            name="P<sub>UFH</sub> [kW]",
            marker_color=[f"rgba(30,107,191,{0.5 + 0.5 * v / max(P_max, 0.01)})" for v in P_UFH],
        ),
        secondary_y=False,
    )

    # Electricity price line
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=prices,
            name="Prijs [€/kWh]",
            mode="lines+markers",
            line=dict(color="#e74c3c", width=2),
            marker=dict(size=5),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.22, font=dict(size=11)),
        hovermode="x unified",
        barmode="group",
    )
    fig.update_yaxes(
        title_text="Vermogen [kW]",
        secondary_y=False,
        range=[0, P_max * 1.1],
        gridcolor="#f5f5f5",
        zeroline=False,
    )
    fig.update_yaxes(
        title_text="Prijs [€/kWh]",
        secondary_y=True,
        range=[0, max(prices) * 1.5],
        gridcolor=None,
        zeroline=False,
        showgrid=False,
    )
    return fig.to_json()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Home Optimizer",
    description="UFH MPC dashboard",
    version="0.1.0",
)

_HTML = """<!DOCTYPE html>
<html lang="nl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Home Optimizer</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#f0f2f5;color:#333;font-size:14px}
header{background:#1e3a5f;color:#fff;padding:.9rem 1.4rem;display:flex;align-items:center;gap:1rem}
header h1{font-size:1.1rem;font-weight:700}
header p{font-size:.8rem;opacity:.75;margin-top:.15rem}
.layout{display:grid;grid-template-columns:270px 1fr;min-height:calc(100vh - 54px)}
.sidebar{background:#fff;border-right:1px solid #e0e0e0;padding:.8rem;overflow-y:auto;display:flex;flex-direction:column;gap:.1rem}
.section-title{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:#888;
  border-top:1px solid #f0f0f0;padding:.7rem 0 .3rem;margin-top:.3rem}
.section-title:first-child{border-top:none;padding-top:0}
.field{margin:.3rem 0}
.field label{display:flex;justify-content:space-between;color:#555;margin-bottom:.2rem}
.field label span{font-weight:700;color:#1e3a5f;min-width:2.5rem;text-align:right}
.field input[type=number]{width:100%;padding:.3rem .45rem;border:1px solid #ddd;border-radius:4px;font-size:.82rem}
.field input[type=number]:focus{outline:none;border-color:#1e6bbf}
.toggle{display:flex;align-items:center;gap:.5rem;margin:.4rem 0}
.toggle label{color:#555}
.toggle input{width:16px;height:16px;accent-color:#1e3a5f;cursor:pointer}
.btn{width:100%;padding:.65rem;background:#1e3a5f;color:#fff;border:none;border-radius:6px;
  font-size:.88rem;font-weight:700;cursor:pointer;margin-top:.8rem;letter-spacing:.02em;transition:background .15s}
.btn:hover{background:#2a5491}
.btn:disabled{background:#aaa;cursor:not-allowed}
.spin{display:inline-block;width:14px;height:14px;border:2px solid #fff;border-top-color:transparent;
  border-radius:50%;animation:spin .6s linear infinite;vertical-align:middle;margin-right:.4rem}
@keyframes spin{to{transform:rotate(360deg)}}
.main{padding:.9rem;display:flex;flex-direction:column;gap:.8rem}
.stats{display:flex;flex-wrap:wrap;gap:.6rem}
.stat-card{background:#fff;border-radius:7px;padding:.6rem .9rem;flex:1;min-width:120px;
  box-shadow:0 1px 3px rgba(0,0,0,.08)}
.stat-label{font-size:.68rem;text-transform:uppercase;letter-spacing:.05em;color:#888}
.stat-value{font-size:1.1rem;font-weight:700;color:#1e3a5f;margin-top:.1rem}
.stat-unit{font-size:.75rem;color:#888;margin-left:.2rem}
.chart-card{background:#fff;border-radius:7px;padding:.9rem;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.chart-card h3{font-size:.8rem;font-weight:700;text-transform:uppercase;letter-spacing:.04em;
  color:#888;margin-bottom:.6rem}
.error-banner{background:#fde;color:#a00;border-radius:6px;padding:.6rem .9rem;font-size:.82rem;display:none}
.warn-banner{background:#fff3cd;color:#856404;border-radius:6px;padding:.6rem .9rem;font-size:.82rem;display:none}
</style>
</head>
<body>
<header>
  <div>
    <h1>🏠 Home Optimizer &mdash; Vloerverwarming MPC</h1>
    <p>2-state thermisch model &middot; Kalman filter &middot; Model Predictive Control</p>
  </div>
</header>
<div class="layout">
  <div class="sidebar">
    <div class="section-title">Huis (thermisch)</div>
    <div class="field"><label>C_r – ruimtecapaciteit [kWh/K]<span id="vCr">3.0</span></label>
      <input type="number" id="C_r" value="3.0" min="0.5" max="50" step="0.5" oninput="upd('vCr',this.value)"></div>
    <div class="field"><label>C_b – vloercapaciteit [kWh/K]<span id="vCb">18.0</span></label>
      <input type="number" id="C_b" value="18.0" min="1" max="200" step="1" oninput="upd('vCb',this.value)"></div>
    <div class="field"><label>R_br – weerstand vloer→kamer [K/kW]<span id="vRbr">2.5</span></label>
      <input type="number" id="R_br" value="2.5" min="0.1" max="20" step="0.5" oninput="upd('vRbr',this.value)"></div>
    <div class="field"><label>R_ro – weerstand kamer→buiten [K/kW]<span id="vRro">4.0</span></label>
      <input type="number" id="R_ro" value="4.0" min="0.1" max="30" step="0.5" oninput="upd('vRro',this.value)"></div>

    <div class="section-title">Beginstaat</div>
    <div class="field"><label>T_r begin [°C]<span id="vTri">20.8</span></label>
      <input type="number" id="T_r_init" value="20.8" min="5" max="35" step="0.1" oninput="upd('vTri',this.value)"></div>
    <div class="field"><label>T_b begin [°C]<span id="vTbi">24.0</span></label>
      <input type="number" id="T_b_init" value="24.0" min="5" max="45" step="0.5" oninput="upd('vTbi',this.value)"></div>
    <div class="field"><label>Vorig vermogen [kW]<span id="vPprev">0.5</span></label>
      <input type="number" id="previous_power_kw" value="0.5" min="0" max="20" step="0.1" oninput="upd('vPprev',this.value)"></div>

    <div class="section-title">MPC instellingen</div>
    <div class="field"><label>Horizon N [uur]<span id="vN">24</span></label>
      <input type="number" id="horizon_hours" value="24" min="4" max="48" step="1" oninput="upd('vN',this.value)"></div>
    <div class="field"><label>T_ref – setpoint [°C]<span id="vTref">21.0</span></label>
      <input type="number" id="T_ref" value="21.0" min="15" max="26" step="0.5" oninput="upd('vTref',this.value)"></div>
    <div class="field"><label>T_min [°C]<span id="vTmin">19.0</span></label>
      <input type="number" id="T_min" value="19.0" min="10" max="25" step="0.5" oninput="upd('vTmin',this.value)"></div>
    <div class="field"><label>T_max [°C]<span id="vTmax">22.5</span></label>
      <input type="number" id="T_max" value="22.5" min="16" max="30" step="0.5" oninput="upd('vTmax',this.value)"></div>
    <div class="field"><label>P_max [kW]<span id="vPmax">4.0</span></label>
      <input type="number" id="P_max" value="4.0" min="0.5" max="20" step="0.5" oninput="upd('vPmax',this.value)"></div>
    <div class="field"><label>ΔP_max [kW/stap]<span id="vDPmax">1.0</span></label>
      <input type="number" id="delta_P_max" value="1.0" min="0.1" max="10" step="0.1" oninput="upd('vDPmax',this.value)"></div>
    <div class="field"><label>Q_c – comfortgewicht<span id="vQc">8.0</span></label>
      <input type="number" id="Q_c" value="8.0" min="0" max="100" step="1" oninput="upd('vQc',this.value)"></div>
    <div class="field"><label>R_c – regularisatie<span id="vRc">0.05</span></label>
      <input type="number" id="R_c" value="0.05" min="0" max="5" step="0.01" oninput="upd('vRc',this.value)"></div>

    <div class="section-title">Weersvoorspelling</div>
    <div class="field"><label>T_buiten [°C]<span id="vTout">10.0</span></label>
      <input type="number" id="outdoor_temperature_c" value="10.0" min="-20" max="35" step="1" oninput="upd('vTout',this.value)"></div>
    <div class="field"><label>Interne winst [kW]<span id="vQint">0.35</span></label>
      <input type="number" id="internal_gains_kw" value="0.35" min="0" max="3" step="0.05" oninput="upd('vQint',this.value)"></div>
    <div class="toggle"><input type="checkbox" id="solar_gain" checked>
      <label for="solar_gain">Zoninstraling meenemen</label></div>
    <div class="toggle"><input type="checkbox" id="dynamic_price" checked>
      <label for="dynamic_price">Dynamische stroomprijs</label></div>
    <div class="field" id="flat_price_field" style="display:none">
      <label>Vaste prijs [€/kWh]<span id="vFlatP">0.25</span></label>
      <input type="number" id="flat_price" value="0.25" min="0" max="2" step="0.01" oninput="upd('vFlatP',this.value)">
    </div>

    <button class="btn" id="run-btn" onclick="runOptimize()">&#9654; Optimaliseer</button>
  </div>

  <div class="main">
    <div class="error-banner" id="err"></div>
    <div class="warn-banner" id="warn" style="display:none"></div>
    <div class="stats">
      <div class="stat-card"><div class="stat-label">Solver status</div>
        <div class="stat-value" id="s-status">–</div></div>
      <div class="stat-card"><div class="stat-label">Eerste P_UFH</div>
        <div class="stat-value" id="s-power">–<span class="stat-unit">kW</span></div></div>
      <div class="stat-card"><div class="stat-label">Totale energie</div>
        <div class="stat-value" id="s-energy">–<span class="stat-unit">kWh</span></div></div>
      <div class="stat-card"><div class="stat-label">Energiekosten</div>
        <div class="stat-value" id="s-cost">–<span class="stat-unit">€</span></div></div>
      <div class="stat-card"><div class="stat-label">Comfortoverschrijding</div>
        <div class="stat-value" id="s-viol">–<span class="stat-unit">K</span></div></div>
    </div>
    <div class="chart-card">
      <h3>Kamertemperatuur T_r &mdash; MPC horizon</h3>
      <div id="temp-chart" style="height:310px"></div>
    </div>
    <div class="chart-card">
      <h3>UFH vermogen &amp; stroomprijs</h3>
      <div id="power-chart" style="height:260px"></div>
    </div>
  </div>
</div>

<script>
function upd(id, val){document.getElementById(id).textContent=parseFloat(val).toFixed(+val<1&&val!='0'?2:1)}

document.getElementById('dynamic_price').addEventListener('change', function(){
  document.getElementById('flat_price_field').style.display = this.checked ? 'none' : '';
});

function getReq(){
  const n=id=>parseFloat(document.getElementById(id).value);
  const b=id=>document.getElementById(id).checked;
  return {
    C_r:n('C_r'),C_b:n('C_b'),R_br:n('R_br'),R_ro:n('R_ro'),alpha:0.35,
    T_r_init:n('T_r_init'),T_b_init:n('T_b_init'),previous_power_kw:n('previous_power_kw'),
    horizon_hours:parseInt(document.getElementById('horizon_hours').value),
    Q_c:n('Q_c'),R_c:n('R_c'),P_max:n('P_max'),delta_P_max:n('delta_P_max'),
    T_min:n('T_min'),T_max:n('T_max'),T_ref:n('T_ref'),
    outdoor_temperature_c:n('outdoor_temperature_c'),
    dynamic_price:b('dynamic_price'),flat_price:n('flat_price'),
    solar_gain:b('solar_gain'),internal_gains_kw:n('internal_gains_kw'),
  };
}

async function runOptimize(){
  const btn=document.getElementById('run-btn');
  btn.disabled=true;
  btn.innerHTML='<span class="spin"></span>Bezig…';
  document.getElementById('err').style.display='none';

  try{
    const resp=await fetch('/api/optimize',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(getReq()),
    });
    if(!resp.ok){
      const msg=(await resp.json()).detail||resp.statusText;
      throw new Error(msg);
    }
    const d=await resp.json();

    document.getElementById('s-status').textContent=d.status;
    document.getElementById('s-power').innerHTML=d.first_power_kw.toFixed(2)+'<span class="stat-unit">kW</span>';
    document.getElementById('s-energy').innerHTML=d.total_energy_kwh.toFixed(2)+'<span class="stat-unit">kWh</span>';
    document.getElementById('s-cost').innerHTML=d.total_cost_eur.toFixed(3)+'<span class="stat-unit">€</span>';
    const viol=d.max_comfort_violation_c;
    const violEl=document.getElementById('s-viol');
    violEl.innerHTML=(viol>0.01?'⚠ '+viol.toFixed(2):'✓ 0.00')+'<span class="stat-unit">K</span>';
    violEl.style.color=viol>0.01?'#c0392b':'#27ae60';
    // Show warning banner if physics forced a comfort violation
    const warn=document.getElementById('warn');
    if(viol>0.01){
      warn.textContent='⚠ Comfortgrens overschreden met '+viol.toFixed(2)+' K – de warmtepomp draait al op maximum gegeven de ramp-rate. Vergroot P_max, ΔP_max of verlaag T_min.';
      warn.style.display='block';
    } else {
      warn.style.display='none';
    }

    const tf=JSON.parse(d.temperature_fig);
    const pf=JSON.parse(d.power_fig);
    Plotly.react('temp-chart', tf.data, tf.layout, {responsive:true,displayModeBar:false});
    Plotly.react('power-chart', pf.data, pf.layout, {responsive:true,displayModeBar:false});
  }catch(e){
    const el=document.getElementById('err');
    el.textContent='⚠ '+e.message;
    el.style.display='block';
  }finally{
    btn.disabled=false;
    btn.innerHTML='&#9654; Optimaliseer';
  }
}

window.addEventListener('load', runOptimize);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    return HTMLResponse(_HTML)


@app.get("/api/defaults")
async def defaults() -> RunRequest:
    return RunRequest()


@app.post("/api/optimize", response_model=OptimizeResponse)
async def optimize(req: RunRequest) -> OptimizeResponse:  # noqa: C901
    start_hour = datetime.now().hour

    # Build physical model
    try:
        thermal_params = ThermalParameters(
            dt_hours=1.0,
            C_r=req.C_r,
            C_b=req.C_b,
            R_br=req.R_br,
            R_ro=req.R_ro,
            alpha=req.alpha,
            eta=0.62,
            A_glass=12.0,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    mpc_params = MPCParameters(
        horizon_steps=req.horizon_hours,
        Q_c=req.Q_c,
        R_c=req.R_c,
        Q_N=req.Q_c * 1.5,
        P_max=req.P_max,
        delta_P_max=req.delta_P_max,
        T_min=req.T_min,
        T_max=req.T_max,
    )

    forecast = _build_forecast(req, start_hour)
    model = ThermalModel(thermal_params)
    controller = UFHMPCController(model=model, params=mpc_params)

    try:
        sol = controller.solve(
            initial_state_c=np.array([req.T_r_init, req.T_b_init]),
            forecast=forecast,
            previous_power_kw=req.previous_power_kw,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    N = req.horizon_hours
    dt = thermal_params.dt_hours
    labels_states = _time_labels(start_hour, N + 1)
    labels_controls = _time_labels(start_hour, N)
    T_r = sol.predicted_states_c[:, 0]
    P_UFH = sol.control_sequence_kw
    prices = forecast.price_eur_per_kwh

    total_energy = float(np.sum(P_UFH) * dt)
    total_cost = float(np.sum(P_UFH * prices * dt))

    temp_fig = _temperature_figure(
        labels_states, T_r, req.T_ref, req.T_min, req.T_max, req.outdoor_temperature_c
    )
    power_fig = _power_figure(labels_controls, P_UFH, prices, req.P_max)

    return OptimizeResponse(
        status=sol.solver_status,
        objective=sol.objective_value,
        total_energy_kwh=total_energy,
        total_cost_eur=total_cost,
        first_power_kw=sol.first_control_kw,
        max_comfort_violation_c=sol.max_comfort_violation_c,
        temperature_fig=temp_fig,
        power_fig=power_fig,
    )
