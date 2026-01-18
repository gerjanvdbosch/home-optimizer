import logging
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np

from plotly.subplots import make_subplots
from sklearn.inspection import partial_dependence, permutation_importance
from operator import itemgetter
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from datetime import timedelta, datetime, timezone
from pathlib import Path


logger = logging.getLogger(__name__)

api = FastAPI(title="Home Optimizer API")

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@api.get("/", response_class=HTMLResponse)
def index(request: Request, explain: str = None):
    """
    Dashboard Home. Leest status direct uit Context.
    """
    coordinator = request.app.state.coordinator
    context = coordinator.context

    # 1. Grafieken genereren
    plot_html = _get_solar_forecast_plot(request)
    importance_html = ""
    behavior_html = ""

    # Formatteer starttijd
    start_str = "-"
    planned_start = getattr(context, "planned_start", None)
    if planned_start and isinstance(planned_start, datetime):
        local_tz = datetime.now().astimezone().tzinfo
        local_start = planned_start.astimezone(local_tz)
        start_str = local_start.strftime("%H:%M")

    # Formatteer status (kan een Enum zijn of string)
    action_val = getattr(context, "action", "Onbekend")
    if hasattr(action_val, "value"): # Als het een Enum is
        action_val = action_val.value

    details = {
        "Status": action_val,
        "Reden": getattr(context, "reason", "-"),
        "Geplande Start": start_str,
        "PV Huidig": f"{context.stable_pv:.2f} kW" if context.stable_pv is not None else "-",
        "Load Huidig": f"{context.stable_load:.2f} kW" if context.stable_load is not None else "-",
        "Boiler Solar": f"{getattr(context, 'boiler_solar_kwh', 0.0):.2f} kWh",
        "Verwachte Load": f"{getattr(context, 'predicted_load_now', 0.0):.2f} kW",
    }

    # Bias info toevoegen indien beschikbaar
    if hasattr(context, "solar_bias"):
        details["Solar Bias"] = f"{context.solar_bias:.2f}"
    if hasattr(context, "load_bias"):
        details["Load Bias"] = f"{context.load_bias:.2f}"

    # 3. Explain (SHAP) data genereren indien aangevraagd (?explain=1)
    explanation = {}
    if explain == "1":
        explanation = _get_explanation_data(coordinator)
        importance_html = _get_importance_plot_plotly(request)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "forecast_plot": plot_html,
            "importance_plot": importance_html,
            "details": details,
            "explanation": explanation,
        },
    )


@api.post("/solar/train", response_class=JSONResponse)
def trigger_training(request: Request):
    """
    Forceer een hertraining van het model.
    """
    coordinator = request.app.state.coordinator

    try:
        coordinator.solar.train()
        new_mae = coordinator.solar.model.mae

        return {
            "status": "success",
            "message": f"Model getraind. Nieuwe MAE: {new_mae:.3f} kW",
        }

    except Exception as e:
        logger.error(f"[API] Training mislukt: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


def _get_solar_forecast_plot(request: Request) -> str:
    """
    Genereert de interactieve Plotly grafiek.
    Past smoothing toe op historische data voor mooie weergave.
    """
    coordinator = request.app.state.coordinator
    context = coordinator.context
    database = coordinator.collector.database

    # Check of er data is
    if not hasattr(context, "forecast_df") or context.forecast_df is None:
        return "<div class='p-4 text-muted'>Geen forecast data beschikbaar.</div>"

    local_tz = datetime.now().astimezone().tzinfo
    local_now = context.now.astimezone(local_tz).replace(tzinfo=None)

    # --- 1. FORECAST DATA VOORBEREIDING ---
    df = context.forecast_df.copy()
    df["timestamp_local"] = df["timestamp"].dt.tz_convert(local_tz).dt.tz_localize(None)

    is_night = df["timestamp_local"].dt.hour.isin([23, 0, 1, 2, 3, 4, 5])

    for col in ["pv_estimate", "power_ml", "power_ml_raw", "power_corrected"]:
        if col in df.columns:
            df[col] = df[col].round(2)
            df.loc[is_night, col] = 0.0

    if df.empty:
        return ""

    # --- 2. HISTORIE OPHALEN & SMOOTHEN ---
    cutoff_date = (
        local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        .replace(tzinfo=local_tz)
        .astimezone(timezone.utc)
    )

    df_hist = database.get_history(cutoff_date)
    df_hist_plot = pd.DataFrame()

    if not df_hist.empty:
        df_hist["timestamp_local"] = (
            df_hist["timestamp"].dt.tz_convert(local_tz).dt.tz_localize(None)
        )

        # === SMOOTHING VOOR GRAFIEK ===
        # De database bevat "blokken" (Average kW per 15m).
        df_hist["pv_smooth"] = (
            df_hist["pv_actual"]
            .fillna(0)
            .rolling(window=4, center=True, min_periods=1)
            .mean()
        )

        df_hist_plot = df_hist.copy()

    # --- 3. BEREIK BEPALEN ---
    active_col = "power_corrected" if "power_corrected" in df.columns else "pv_estimate"
    zon_uren = df[df[active_col] > 0]
    if not zon_uren.empty and not df_hist_plot.empty:
        x_start = zon_uren["timestamp_local"].min() - timedelta(hours=2)
        x_end = max(
            zon_uren["timestamp_local"].max(), df_hist_plot["timestamp_local"].max()
        ) + timedelta(hours=2)
    else:
        x_start = df["timestamp_local"].min()
        x_end = df["timestamp_local"].max()

    fig = go.Figure()

    # A. Raw Solcast (Grijs, dashed)
    fig.add_trace(
        go.Scatter(
            x=df["timestamp_local"],
            y=df["pv_estimate"],
            mode="lines",
            name="Solcast",
            line=dict(color="#888888", dash="dash", width=1),
            opacity=0.7,
        )
    )

    # B. Model Correction (Blauw, dot)
    if "power_ml" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp_local"],
                y=df["power_ml"],
                mode="lines",
                name="Blended",
                line=dict(color="#4fa8ff", dash="dot", width=1),
                opacity=0.8,
                visible="legendonly",
            )
        )

    if "power_ml_raw" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp_local"],
                y=df["power_ml_raw"],
                mode="lines",
                name="Model",
                line=dict(color="#9467bd", dash="dot", width=1.5),  # Paars stippel
                opacity=0.6,
                visible="legendonly",
            )
        )

    if not df_hist_plot.empty:
        fig.add_trace(go.Scatter(
            x=df_hist_plot["timestamp_local"], y=df_hist_plot["pv_actual"],
            mode="lines", line=dict(color="#ffa500", width=1, shape="hv"),
            opacity=0.3, showlegend=False, hoverinfo="skip"
        ))

        # Doe hetzelfde voor Load als je die historisch toont
        # df_hist["load_smooth"] = df_hist["load_actual"].rolling(window=2, center=True).mean()
        fig.add_trace(
            go.Scatter(
                x=df_hist_plot["timestamp_local"],
                y=df_hist_plot["pv_smooth"],
                mode="lines",
                name="Historie",
                legendgroup="history",
                line=dict(color="#ffa500", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(255, 165, 0, 0.1)",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[df_hist_plot["timestamp_local"].iloc[-1], local_now],
                y=[df_hist_plot["pv_actual"].iloc[-1], context.stable_pv],
                mode="lines",
                line=dict(color="#ffa500", dash="dash", width=1.5),  # Wit en gestippeld
                fill="tozeroy",
                fillcolor="rgba(255, 165, 0, 0.1)",
                showlegend=False,  # We hoeven deze niet apart in de legenda
                hoverinfo="skip",  # Geen popup als je over het lijntje muist
                legendgroup="history",
            )
        )

    # C. Actuele PV Meting (Stip)
    # We tekenen alleen de stip, de horizontale stippellijn doen we via shapes of een losse trace als je wilt
    fig.add_trace(
        go.Scatter(
            x=[local_now],
            y=[context.stable_pv],
            mode="markers",
            name="Now",
            showlegend=False,
            marker=dict(color="#ffa500", size=12, line=dict(color="white", width=2)),
            zorder=10,
        )
    )

    if "power_corrected" in df.columns:
        # D. Corrected Solar
        df_future = df[df["timestamp_local"] >= local_now]

        # Om de lijn visueel aan het 'Huidig PV' bolletje te knopen,
        # plakken we het huidige punt vooraan de lijst.
        x_future = [local_now] + df_future["timestamp_local"].tolist()
        y_future = [context.stable_pv] + df_future["power_corrected"].tolist()

        fig.add_trace(
            go.Scatter(
                x=x_future,
                y=y_future,
                mode="lines",
                name="Forecast",
                line=dict(color="#ffffff", dash="dash", width=2),
                fill="tozeroy",  # Vul tot aan de X-as (0)
                fillcolor="rgba(255, 255, 255, 0.05)",
                opacity=0.8,
            )
        )

    # E. Load Forecast (Rood)
    if "load_corrected" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp_local"], y=df["load_corrected"], mode="lines",
            name="Load", line=dict(color="#ff5555", width=1.5),
            opacity=0.8
        ))

    # F. Boiler Start Lijn (Direct uit Context)
    planned_start = getattr(context, "planned_start", None)
    if planned_start and isinstance(planned_start, datetime):
        local_start = planned_start.astimezone(local_tz).replace(tzinfo=None)

        # Check of starttijd in de grafiek past
        if local_start >= x_start and local_start <= x_end + timedelta(hours=12):
            fig.add_vline(
                x=local_start,
                line_width=2, line_dash="dot", line_color="#2ca02c",
                annotation_text="Start", annotation_position="top right"
            )

            # 3. TRACE B: Het gearceerde vlak
            # We tekenen een onzichtbare lijn rondom het gebied en vullen die in ("toself")
            fig.add_trace(
                go.Scatter(
                    x=[local_start, duration_end, duration_end, local_start],
                    y=[0, 0, y_top, y_top],
                    fill="toself",
                    fillcolor="rgba(44, 160, 44, 0.15)",
                    mode="lines",
                    line=dict(width=0),  # Geen rand
                    name="Start",  # Zelfde naam
                    legendgroup="start",  # Zelfde groep!
                    showlegend=False,  # Niet dubbel in de lijst tonen
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgb(28, 28, 28)",
        plot_bgcolor="rgb(28, 28, 28)",
        xaxis=dict(
            title="Tijd",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            range=[x_start, x_end],
        ),
        yaxis=dict(
            title="Vermogen (kW)",
            # range=[0, y_max],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            itemdoubleclick=False,
        ),
        margin=dict(l=40, r=20, t=80, b=40),
        height=500,
        hovermode="x unified",  # Laat alle waardes zien op 1 verticale lijn
    )

    return pio.to_html(
        fig, full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False}
    )


def _get_explanation_data(coordinator) -> dict:
    """ Bereidt SHAP data voor. """
    context = coordinator.context
    forecaster = coordinator.solar

    if not hasattr(context, "forecast_df") or context.forecast_df is None:
        return None

    if not forecaster.model.is_fitted:
        return None

    try:
        local_tz = datetime.now().astimezone().tzinfo
        current_time = datetime.now(local_tz)
        df = context.forecast_df.copy()

        target_col = "power_ml_raw"

        # 3. Zoek 'Nu'
        target_time = current_time.astimezone(df["timestamp"].dt.tz)
        idx_now = df["timestamp"].searchsorted(target_time)
        idx_now = min(idx_now, len(df) - 1)

        # 4. Slimme Logica: Nu of Piek?
        prediction_now = df.iloc[idx_now][target_col]

        row = None
        time_label = ""

        # Als het donker is (< 0.05 kW), zoek de piek van de dag
        if prediction_now < 0.05:
            idx_max = df[target_col].idxmax()
            peak_val = df.iloc[idx_max][target_col]

            if peak_val > 0.1:
                row = df.iloc[[idx_max]].copy()
                ts = row["timestamp"].dt.tz_convert(local_tz).iloc[0]
                time_label = f"Piek om {ts.strftime('%H:%M')}"
            else:
                row = df.iloc[[idx_now]].copy()
                time_label = "Nu (Nacht)"
        else:
            row = df.iloc[[idx_now]].copy()
            time_label = "Nu"

        # 5. Vraag SHAP waardes op
        shap_data = forecaster.model.explain(row)

        # 6. Formatteren en Opschonen
        # Haal base en prediction eruit (kleine letters want SolarModel geeft lowercase terug)
        base_val = shap_data.pop("base", "0.00")
        pred_val = shap_data.pop("prediction", "0.00")

        # Verwijder rommel
        shap_data.pop("Info", None)
        shap_data.pop("_meta_label", None)

        label_map = {
            "pv_estimate": "Solcast Basis",
            "pv_estimate10": "Solcast Min (10%)",
            "pv_estimate90": "Solcast Max (90%)",
            "radiation": "Straling",
            "hour_cos": "Tijdstip (Uur)",
            "hour_sin": "Tijdstip (Cyclisch)",
            "temp": "Temperatuur",
            "cloud": "Bewolking",
            "diffuse": "Diffuus Licht",
            "tilted": "Dakhelling Effect",
            "uncertainty": "Onzekerheid",
            "wind": "Wind",
            "doy_cos": "Seizoen",
            "doy_sin": "Seizoen (Cyclisch)",
        }

        factors = []
        for key, val_str in shap_data.items():
            try:
                val = float(val_str)
                # Filter verwaarloosbare waardes (< 10 Watt)
                if abs(val) < 0.01:
                    continue

                label = label_map.get(key, key)  # Vertaal of gebruik origineel

                factors.append(
                    {
                        "label": label,
                        "value": val_str,
                        "abs_value": abs(val),
                        "css_class": "val-pos" if val >= 0 else "val-neg",
                    }
                )
            except Exception:
                continue

        # Sorteer op absolute impact (grootste bovenaan)
        factors.sort(key=itemgetter("abs_value"), reverse=True)

        return {
            "time_label": time_label,
            "base": base_val,
            "prediction": pred_val,
            "factors": factors,
        }

    except Exception as e:
        logger.error(f"Error preparing explanation: {e}")
        return None


def _get_importance_plot_plotly(request: Request) -> str:
    """
    Genereert een Plotly Bar Chart met de feature importance.
    Let op: Dit is rekenintensief, dus we doen dit op een beperkte set.
    """
    coordinator = request.app.state.coordinator
    forecaster = coordinator.solar
    database = coordinator.collector.database

    # 1. Check: Is het model getraind?
    if not forecaster.model.is_fitted:
        return "<div class='p-4 text-muted'>Model nog niet getraind.</div>"

    # 2. Data ophalen (Beperk tot laatste 14 dagen voor snelheid)
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=14)
    df_hist = database.get_history(cutoff_date)

    if df_hist.empty:
        return "<div class='p-4 text-muted'>Onvoldoende data voor analyse.</div>"

    # 3. Filter op daglicht (cruciaal, anders klopt de importance niet)
    # We gebruiken dezelfde logica als bij het trainen/analyseren
    is_daytime = (df_hist["pv_estimate"] > 0.01) | (df_hist["pv_actual"] > 0.01)
    df_day = df_hist[is_daytime].copy()

    if len(df_day) < 10:
        return "<div class='p-4 text-muted'>Wachten op meer daglicht-data...</div>"

    # 4. Features voorbereiden
    try:
        X = forecaster.model._prepare_features(df_day)
        y = df_day["pv_actual"]

        # 5. Bereken Importance (n_repeats=2 houdt het snel genoeg voor een dashboard)
        # Voor wetenschappelijke precisie wil je 10, voor een dashboard is 2-3 prima.
        result = permutation_importance(
            forecaster.model.model, X, y, n_repeats=2, random_state=42, n_jobs=-1
        )
    except Exception as e:
        logger.error(f"Fout bij berekenen importance: {e}")
        return "<div class='p-4 text-danger'>Kon importance niet berekenen.</div>"

    # 6. Sorteren en Labelen
    sorted_idx = result.importances_mean.argsort()

    # Vertaal de labels naar leesbaar Nederlands
    labels_raw = X.columns[sorted_idx]
    labels_clean = []

    label_map = {
        "pv_estimate": "Solcast Basis",
        "pv_estimate10": "Solcast Min (10%)",
        "pv_estimate90": "Solcast Max (90%)",
        "radiation": "Straling",
        "hour_cos": "Tijdstip (Uur)",
        "hour_sin": "Tijdstip (Cyclisch)",
        "temp": "Temperatuur",
        "cloud": "Bewolking",
        "diffuse": "Diffuus Licht",
        "tilted": "Dakhelling Effect",
        "uncertainty": "Onzekerheid",
        "wind": "Wind",
        "doy_cos": "Seizoen",
        "doy_sin": "Seizoen (Cyclisch)",
    }

    for label in labels_raw:
        clean = label_map.get(label, label)  # Pak vertaling of origineel
        labels_clean.append(clean)

    values = result.importances_mean[sorted_idx]

    # 7. Plotly Grafiek Maken
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels_clean,
            orientation="h",
            marker=dict(
                color=values,
                colorscale="Viridis",  # Of 'Blues', 'Magma'
                showscale=False,
            ),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgb(28, 28, 28)",
        plot_bgcolor="rgb(28, 28, 28)",
        margin=dict(l=20, r=20, t=30, b=30),
        height=400,
        xaxis=dict(
            title="Impact op voorspelling",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
        ),
        yaxis=dict(showgrid=False),
    )

    return pio.to_html(
        fig, full_html=False, include_plotlyjs=False, config={"displayModeBar": False}
    )
