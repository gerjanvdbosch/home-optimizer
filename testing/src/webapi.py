import logging
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd

from sklearn.inspection import permutation_importance
from operator import itemgetter
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from datetime import timedelta, datetime, timezone, date
from pathlib import Path

logger = logging.getLogger(__name__)

api = FastAPI(title="Home Optimizer API")

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@api.get("/", response_class=HTMLResponse)
def index(request: Request, explain: str = None, train: str = None, view: str = "hour", date_str: str = None):
    """
    Dashboard Home. Leest status direct uit Context.
    """
    coordinator = request.app.state.coordinator
    context = coordinator.context

    local_tz = datetime.now().astimezone().tzinfo
    today = context.now.astimezone(local_tz).date()

    if date_str:
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            target_date = today
    else:
        target_date = today

    # Navigatie datums berekenen
    prev_date = target_date - timedelta(days=1)
    next_date = target_date + timedelta(days=1)

    # Is het vandaag? (Voor UI highlight en auto-refresh logica)
    is_today = (target_date == today)

    # 1. Grafieken genereren
    plot_html = _get_solar_forecast_plot(request, target_date)
    view_mode = "15min" if view == "15min" else "hour"
    measurements_data = _get_energy_table(request, view_mode, target_date)
    importance_html = ""

    result = context.result if hasattr(context, "result") else None

    details = {
        "Mode": result["mode"] if result is not None else "-",
        "PV Huidig": (
            f"{context.stable_pv:.2f} kW" if context.stable_pv is not None else "-"
        ),
        "Load Huidig": (
            f"{context.stable_load:.2f} kW" if context.stable_load is not None else "-"
        ),
        "Boiler Solar": f"{getattr(context, 'boiler_solar_kwh', 0.0):.2f} kWh",
        "Verwachte Load": f"{getattr(context, 'predicted_load_now', 0.0):.2f} kW",
    }

    # Bias info toevoegen indien beschikbaar
    if hasattr(context, "solar_bias"):
        details["Solar Bias"] = f"{context.solar_bias * 100:.1f} %"
    if hasattr(context, "load_bias"):
        details["Load Bias"] = f"{context.load_bias:.2f} kW"

    # 3. Explain (SHAP) data genereren indien aangevraagd (?explain=1)
    explanation = {}
    if explain == "1":
        explanation = _get_explanation_data(coordinator)
        importance_html = _get_importance_plot_plotly(request)

    if train == "1":
        try:
            coordinator.train()
        except Exception as e:
            logger.error(f"Fout bij trainen van modellen: {e}")

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "forecast_plot": plot_html,
            "importance_plot": importance_html,
            "details": details,
            "explanation": explanation,
            "measurements": measurements_data,
            "optimization_plan": getattr(context, "optimization_plan", None),
            "current_view": view_mode,
            "target_date": target_date,
            "prev_date": prev_date,
            "next_date": next_date,
            "is_today": is_today,
            "today_date": today
        },
    )


def _get_solar_forecast_plot(request: Request, target_date: date) -> str:
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

    start_of_day = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=local_tz)
    end_of_day = datetime.combine(target_date, datetime.max.time()).replace(tzinfo=local_tz)

    # --- 1. FORECAST DATA VOORBEREIDING ---
    df = context.forecast_df.copy()
    df["timestamp_local"] = df["timestamp"].dt.tz_convert(local_tz).dt.tz_localize(None)

    is_night = df["timestamp_local"].dt.hour.isin([23, 0, 1, 2, 3, 4, 5])

    for col in [
        "pv_estimate",
        "pv_actual",
        "power_ml",
        "power_ml_raw",
        "power_corrected",
        "load_corrected",
    ]:
        if col in df.columns:
            df[col] = df[col].round(2)
            if col.startswith("power"):
                df.loc[is_night, col] = 0.0

    if df.empty:
        return ""

    # --- 2. HISTORIE OPHALEN & SMOOTHEN ---
    cutoff_date = start_of_day.astimezone(timezone.utc)

    df_hist = database.get_history(cutoff_date)
    df_hist_plot = pd.DataFrame()

    if not df_hist.empty:
        df_hist["timestamp_local"] = (
            df_hist["timestamp"].dt.tz_convert(local_tz).dt.tz_localize(None)
        )

        # --- A. HISTORIE: VLOEIENDE LIJN (FIX VOOR HET STUITEREN) ---
        # 1. Zet index
        df_hist_smooth = df_hist.set_index("timestamp_local").sort_index()

        df_hist_smooth["base_load"] = (
            df_hist_smooth["grid_import"]
            - df_hist_smooth["grid_export"]
            + df_hist_smooth["pv_actual"]
            - df_hist_smooth.get("wp_actual")
        ).clip(lower=0)

        # 4. Laatste check op negatieve waarden
        df_hist_smooth["pv_actual"] = df_hist_smooth["pv_actual"].clip(lower=0).round(2)

        df_hist_plot = df_hist_smooth.reset_index()

    # --- 3. BEREIK BEPALEN ---
    # active_col = "power_corrected" if "power_corrected" in df.columns else "pv_estimate"
    # zon_uren = df[df[active_col] > 0]
    # if not zon_uren.empty and not df_hist_plot.empty:
    #     x_start = zon_uren["timestamp_local"].min() - timedelta(hours=2)
    #     x_end = max(
    #         zon_uren["timestamp_local"].max(), df_hist_plot["timestamp_local"].max()
    #     ) + timedelta(hours=2)
    # else:
    #     x_start = df["timestamp_local"].min()
    #     x_end = df["timestamp_local"].max()

    x_start = start_of_day
    x_end = end_of_day

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
            hovertemplate="%{y:.2f} kW<extra></extra>",
            # hoverinfo="skip",
        )
    )

    if "power_ml_raw" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp_local"],
                y=df["power_ml_raw"],
                mode="lines",
                name="Model",
                line=dict(color="#FF9100", dash="dot", width=1),
                opacity=0.6,
                visible="legendonly",
                legendgroup="model",
                hovertemplate="%{y:.2f} kW<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df["timestamp_local"],
                y=df["load_ml"],
                mode="lines",
                line=dict(color="#F50057", dash="dot", width=1, shape="hv"),
                showlegend=False,
                visible="legendonly",
                legendgroup="model",
                hovertemplate="%{y:.2f} kW<extra></extra>",
                opacity=0.6,
            )
        )

    if not df_hist_plot.empty:
        fig.add_trace(
            go.Scatter(
                x=df_hist_plot["timestamp_local"],
                y=df_hist_plot["base_load"],
                name="Base load",
                mode="lines",
                legendgroup="load",
                line=dict(color="#F50057", width=1, shape="hv"),
                hovertemplate="%{y:.2f} kW<extra></extra>",
                opacity=0.8,
            )
        )

        # Verbind historie met 'nu' punt
        fig.add_trace(
            go.Scatter(
                x=[df_hist_plot["timestamp_local"].iloc[-1], local_now],
                y=[df_hist_plot["base_load"].iloc[-1], context.stable_load],
                mode="lines",
                line=dict(color="#F50057", width=1, shape="hv"),
                showlegend=False,
                legendgroup="load",
                hoverinfo="skip",
                opacity=0.8,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_hist_plot["timestamp_local"],
                y=df_hist_plot["pv_actual"],
                name="Solar",
                mode="lines",
                legendgroup="solar",
                line=dict(color="#FF9100", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(255, 145, 0, 0.07)",
                hovertemplate="%{y:.2f} kW<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[df_hist_plot["timestamp_local"].iloc[-1], local_now],
                y=[df_hist_plot["pv_actual"].iloc[-1], context.stable_pv],
                mode="lines",
                line=dict(color="#FF9100", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(255, 145, 0, 0.07)",
                showlegend=False,  # We hoeven deze niet apart in de legenda
                hoverinfo="skip",  # Geen popup als je over het lijntje muist
                legendgroup="solar",
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
            marker=dict(color="#FF9100", size=12, line=dict(color="white", width=1)),
            zorder=10,
            hoverinfo="skip",
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
                showlegend=False,
                legendgroup="solar",
                line=dict(color="#ffffff", dash="dash", width=1.5),
                fill="tozeroy",  # Vul tot aan de X-as (0)
                fillcolor="rgba(255, 255, 255, 0.05)",
                opacity=0.8,
                hovertemplate="%{y:.2f} kW<extra></extra>",
                # hoverinfo="skip",
            )
        )

    # E. Load Forecast (Rood)
    if "load_corrected" in df.columns:
        df_future_load = df[df["timestamp_local"] >= local_now]

        # Verbind 'nu' met de toekomst
        x_load_future = [local_now] + df_future_load["timestamp_local"].tolist()
        y_load_future = [context.stable_load] + df_future_load[
            "load_corrected"
        ].tolist()

        fig.add_trace(
            go.Scatter(
                x=x_load_future,
                y=y_load_future,
                mode="lines",
                name="Base load",
                showlegend=False,
                legendgroup="load",
                # Dash voor toekomst, solid voor historie
                line=dict(color="#F50057", width=1, shape="hv"),
                hovertemplate="%{y:.2f} kW<extra></extra>",
                opacity=0.8,
            )
        )

    fig.add_vline(
        x=local_now, line_width=1, line_dash="solid", line_color="white", opacity=0.6
    )

    # F. Boiler Start Lijn (Direct uit Context)
    planned_start = getattr(context, "planned_start", None)
    if planned_start and isinstance(planned_start, datetime):
        local_start = planned_start.astimezone(local_tz).replace(tzinfo=None)

        # Check of starttijd in de grafiek past
        if local_start >= x_start and local_start <= x_end + timedelta(hours=12):
            fig.add_vline(
                x=local_start,
                line_width=2,
                line_dash="dot",
                line_color="#2ca02c",
                annotation_text="Start",
                annotation_position="top right",
                hovertemplate="%{y:.2f} kW<extra></extra>",
                # hoverinfo="skip",
            )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgb(28, 28, 28)",
        plot_bgcolor="rgb(28, 28, 28)",
        xaxis=dict(
            title=None,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            range=[x_start, x_end],
            fixedrange=False,  # Sta zoomen toe
        ),
        yaxis=dict(
            title="Vermogen (kW)",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            fixedrange=True,  # Y-as vast
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            itemdoubleclick=False,
        ),
        margin=dict(l=40, r=20, t=60, b=40),
        height=450,
        hovermode="x unified",
    )

    return pio.to_html(
        fig, full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False}
    )


def _get_explanation_data(coordinator) -> dict:
    """Bereidt SHAP data voor."""
    context = coordinator.context
    forecaster = coordinator.solar

    if not hasattr(context, "forecast_df") or context.forecast_df is None:
        return "Geen forecast data beschikbaar."

    if not forecaster.model.is_fitted:
        return "Model nog niet getraind."

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
    df_hist = df_hist.copy()
    is_daytime = (df_hist["pv_estimate"] > 0.01) | (df_hist["pv_actual"] > 0.01)

    df_train = df_hist[is_daytime].copy()
    df_train = df_train.dropna(subset=["pv_actual", "pv_estimate"])

    if len(df_train) < 10:
        return "<div class='p-4 text-muted'>Wachten op meer daglicht-data...</div>"

    # 4. Features voorbereiden
    try:
        X = forecaster.model._prepare_features(df_train)
        y = df_train["pv_actual"].clip(0, coordinator.config.pv_max_kw)

        # 5. Bereken Importance (n_repeats=2 houdt het snel genoeg voor een dashboard)
        # Voor wetenschappelijke precisie wil je 10, voor een dashboard is 2-3 prima.
        result = permutation_importance(
            forecaster.model.model, X, y, n_repeats=3, random_state=42, n_jobs=-1
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


def _get_energy_table(request: Request, view_mode: str, target_date: date):
    """
    Haalt data op en verwerkt deze op basis van de modus:
    - "15min": Toont vermogen (kW) met smoothing.
    - "hour":  Toont energie (kWh) per uur (sommatie).
    """
    coordinator = request.app.state.coordinator
    database = coordinator.collector.database

    # 1. Algemene Data Fetch (Gedeeld)
    local_tz = datetime.now().astimezone().tzinfo
    start_dt = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=local_tz)
    start_utc = start_dt.astimezone(timezone.utc)

    # Haal data op (get_history haalt alles op NA de datum, dus we moeten straks filteren op eindtijd)
    df = database.get_history(start_utc)
    df["timestamp"] = df["timestamp"].dt.tz_convert(local_tz)

    if df.empty:
        return []

    # Zet tijdzone goed en indexeer
    end_dt = start_dt + timedelta(days=1)
    df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)]

    df = df.set_index("timestamp").sort_index()

    # Kolommen die we gaan bewerken
    process_cols = ["grid_import", "grid_export", "pv_actual", "wp_actual"]
    # Zorg dat kolommen bestaan (voorkom key errors)
    process_cols = [c for c in df.columns if c in process_cols]

    # --- SPLITSING IN LOGICA ---
    if view_mode == "hour":
        # A. UUR-MODUS (kWh)
        df[process_cols] = df[process_cols].apply(pd.to_numeric, errors="coerce")
        df[process_cols] = df[process_cols].fillna(0.0)

        # 1. kW naar kWh (delen door 4)
        df[process_cols] = df[process_cols] * 0.25

        # 2. Sommeer per uur
        df = df.resample("1h", label="left").sum(numeric_only=True)

        # 3. Filter toekomst weg (anders heb je lege rijen voor vanavond)
        if target_date == datetime.now(local_tz).date():
            current_hour = datetime.now(local_tz).replace(minute=0, second=0, microsecond=0)
            df = df[df.index <= current_hour]

    # --- EINDE SPLITSING ---

    # 2. Bereken Totalen (Gedeelde logica)
    # De wiskunde is nu voor beide gelijk (of het nu kW of kWh is)
    df["total_calc"] = (df["grid_import"] - df["grid_export"] + df["pv_actual"]).clip(
        lower=0.0
    )

    df["base_calc"] = (df["total_calc"] - df["wp_actual"]).clip(lower=0.0)

    # 3. Formatteren voor output
    df = df.reset_index().sort_values("timestamp", ascending=False)

    table_data = []
    for _, row in df.iterrows():
        table_data.append(
            {
                "time": row["timestamp"].strftime("%H:%M"),
                # We gebruiken dezelfde keys, de template past de eenheid aan
                "pv": f"{row['pv_actual']:.2f}",
                "wp": f"{row['wp_actual']:.2f}",
                "import": f"{row['grid_import']:.2f}",
                "export": f"{row['grid_export']:.2f}",
                "total": f"{row['total_calc']:.2f}",
                "base": f"{row['base_calc']:.2f}",
            }
        )

    return table_data
