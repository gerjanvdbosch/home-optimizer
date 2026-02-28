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
from context import HvacMode

logger = logging.getLogger(__name__)

api = FastAPI(title="Home Optimizer API")

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@api.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    explain: str = None,
    train: str = None,
    view: str = "hour",
    date_str: str = None,
):
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
    is_today = target_date == today

    result = context.result if hasattr(context, "result") else None
    plan = result.get("plan") if result else None

    plan_display = []
    if plan:
        for p in plan:
            p_copy = p.copy()
            # Zet het object om naar tekst voor de tabel
            p_copy["time"] = p["time"].strftime("%H:%M")
            plan_display.append(p_copy)

    # 1. Grafieken genereren
    plot_html = _get_solar_forecast_plot(request, target_date, plan)
    view_mode = "15min" if view == "15min" else "hour"
    measurements_data = _get_energy_table(request, view_mode, target_date)
    importance_html = ""

    # Huidige temperaturen veilig ophalen
    room_t = getattr(context, "room_temp", 0.0)
    dhw_t = (
        getattr(context, "dhw_top", 0.0) + getattr(context, "dhw_bottom", 0.0)
    ) / 2.0

    # Actuele COP bepalen uit het plan (rij 0 is 'nu')
    current_cop = "-"
    if plan and len(plan) > 0:
        current_mode = result.get("mode", "OFF")
        if current_mode == "UFH":
            current_cop = plan[0].get("cop_ufh", "-")
        elif current_mode == "DHW":
            current_cop = plan[0].get("cop_dhw", "-")

    # Gestructureerde lijst maken
    details = []
    if result:
        details.extend(
            [
                {"label": "Modus", "value": result.get("mode", "-")},
                {
                    "label": "Zon Actueel",
                    "value": (
                        f"{context.stable_pv:.2f}"
                        if getattr(context, "stable_pv", None) is not None
                        else "-"
                    ),
                    "unit": "kW",
                    # "color": "#FF9100"
                },
                {
                    "label": "Huis Actueel",
                    "value": (
                        f"{context.stable_load:.2f}"
                        if getattr(context, "stable_load", None) is not None
                        else "-"
                    ),
                    "unit": "kW",
                    # "color": "#F50057"
                },
                {"label": "Kamer Temp", "value": f"{room_t:.1f}", "unit": "°C"},
                {"label": "Boiler Temp", "value": f"{dhw_t:.1f}", "unit": "°C"},
                {
                    "label": "Doel Aanvoer",
                    "value": f"{result.get('target_supply_temp', 0):.1f}",
                    "unit": "°C",
                },
                {
                    "label": "Doel Vermogen",
                    "value": f"{result.get('target_pel_kw', 0):.2f}",
                    "unit": "kW",
                },
            ]
        )

        # Alleen de COP tonen als hij daadwerkelijk draait
        if current_cop != "-":
            details.append({"label": "Verwachte COP", "value": current_cop, "unit": ""})

    # Optionele debug/ML informatie toevoegen
    if hasattr(context, "solar_bias"):
        details.append(
            {
                "label": "Zon Bias",
                "value": f"{context.solar_bias * 100:.1f}",
                "unit": "%",
            }
        )
    if hasattr(context, "load_bias"):
        details.append(
            {"label": "Load Bias", "value": f"{context.load_bias:.2f}", "unit": "kW"}
        )

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
            "plan": plan_display,
            "current_view": view_mode,
            "target_date": target_date,
            "prev_date": prev_date,
            "next_date": next_date,
            "is_today": is_today,
            "today_date": today,
        },
    )


def _get_solar_forecast_plot(
    request: Request, target_date: date, plan: list = None
) -> str:
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

    start_of_day = datetime.combine(target_date, datetime.min.time()).replace(
        tzinfo=local_tz
    )
    end_of_day = datetime.combine(target_date, datetime.max.time()).replace(
        tzinfo=local_tz
    )

    # --- 1. FORECAST DATA VOORBEREIDING ---
    df = context.forecast_df.copy()
    df["timestamp_local"] = df["timestamp"].dt.tz_convert(local_tz).dt.tz_localize(None)

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

    x_start = start_of_day
    x_end = end_of_day

    fig = go.Figure()

    # --- 1. SOLCAST TOTAAL (HISTORIE + TOEKOMST) ---
    sol_x = []
    sol_y = []

    # Voeg historie toe
    if not df_hist_plot.empty:
        df_hist_sol = df_hist_plot[df_hist_plot["timestamp_local"] <= local_now]
        for i, row in df_hist_sol.iterrows():
            sol_x.append(row["timestamp_local"])
            sol_y.append(row["pv_estimate"])

    # Voeg toekomst toe (vanaf het laatste punt in de historie of nu)
    last_ts = sol_x[-1] if sol_x else local_now
    df_future_sol = df[df["timestamp_local"] > last_ts]

    for i, row in df_future_sol.iterrows():
        sol_x.append(row["timestamp_local"])
        sol_y.append(row["pv_estimate"])

    # --- 2. OPSCHONEN (VOOR DE NACHT-BREUK) ---
    # Dezelfde logica als je al had: haal de 0-lijnen 's nachts weg
    clean_sol_x = []
    clean_sol_y = []

    for i in range(len(sol_x)):
        val = sol_y[i]
        is_active = val > 0.01
        next_active = i < len(sol_x) - 1 and sol_y[i + 1] > 0.01
        prev_active = i > 0 and sol_y[i - 1] > 0.01

        if is_active or next_active or prev_active:
            clean_sol_x.append(sol_x[i])
            clean_sol_y.append(val)
        else:
            if clean_sol_x and clean_sol_y[-1] is not None:
                clean_sol_x.append(sol_x[i])
                clean_sol_y.append(None)

    # --- 3. TEKEN DE GECOMBINEERDE LIJN ---
    if clean_sol_x:
        fig.add_trace(
            go.Scatter(
                x=clean_sol_x,
                y=clean_sol_y,
                mode="lines",
                name="Solcast",
                line=dict(color="#888888", dash="dash", width=1),
                opacity=0.5,
                connectgaps=False,
                hovertemplate="%{y:.2f} kW<extra></extra>",
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

        # --- 1. HISTORISCHE SOLAR (pv_actual) ZONDER 0-LIJN ---
        pv_x = []
        pv_y = []
        for i in range(len(df_hist_plot)):
            val = df_hist_plot.iloc[i]["pv_actual"]
            ts = df_hist_plot.iloc[i]["timestamp_local"]

            # Check of dit punt, het volgende of het vorige punt positief is
            is_active = val > 0.01
            next_is_active = (
                i < len(df_hist_plot) - 1
                and df_hist_plot.iloc[i + 1]["pv_actual"] > 0.01
            )
            prev_is_active = i > 0 and df_hist_plot.iloc[i - 1]["pv_actual"] > 0.01

            # We nemen het punt mee als het zelf actief is,
            # OF als het de 0 is die direct naast een actief blok ligt (voor de ramp)
            if is_active or next_is_active or prev_is_active:
                pv_x.append(ts)
                pv_y.append(val)
            else:
                # Alleen None toevoegen als we net uit een actieve zone komen
                # Dit breekt de lijn zodat hij 's nachts niet over de bodem loopt
                if pv_x and pv_y[-1] is not None:
                    pv_x.append(ts)
                    pv_y.append(None)

        if pv_x:
            fig.add_trace(
                go.Scatter(
                    x=pv_x,
                    y=pv_y,
                    name="Solar",
                    mode="lines",
                    legendgroup="solar",
                    line=dict(color="#FF9100", width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(255, 145, 0, 0.07)",
                    connectgaps=False,  # Zorg dat de Nones de lijn echt breken
                    hovertemplate="%{y:.2f} kW<extra></extra>",
                )
            )

        # --- 2. VERBINDING NAAR 'NU' (Alleen als er productie is) ---
        if not df_hist_plot.empty and (
            df_hist_plot["pv_actual"].iloc[-1] > 0.01 or context.stable_pv > 0.01
        ):
            last_ts = df_hist_plot["timestamp_local"].iloc[-1]
            last_val = df_hist_plot["pv_actual"].iloc[-1]

            # Alleen tekenen als we niet al een 'None' hebben aan het einde
            fig.add_trace(
                go.Scatter(
                    x=[last_ts, local_now],
                    y=[last_val, context.stable_pv],
                    mode="lines",
                    line=dict(color="#FF9100", width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(255, 145, 0, 0.07)",
                    showlegend=False,
                    hoverinfo="skip",
                    legendgroup="solar",
                )
            )

        # --- HISTORISCH VERBRUIK WARMTEPOMP (DHW & UFH) ---
        # 1. Zorg dat we met getallen werken en vang ontbrekende kolommen/data op
        # Dit voorkomt de 'round' of 'float' errors in de lambda
        temp_df = df_hist_plot.copy()
        temp_df["hvac_mode"] = pd.to_numeric(
            temp_df.get("hvac_mode", 0), errors="coerce"
        ).fillna(0)
        temp_df["wp_actual"] = pd.to_numeric(
            temp_df.get("wp_actual", 0), errors="coerce"
        ).fillna(0)

        for mode_id, mode_key, color, fill, label in [
            (
                HvacMode.HEATING.value,
                "ufh",
                "#d05ce3",
                "rgba(208, 92, 227, 0.15)",
                "UFH",
            ),
            (HvacMode.DHW.value, "dhw", "#02cfe7", "rgba(0, 229, 255, 0.15)", "DHW"),
        ]:
            hist_x = []
            hist_y = []

            # 2. Gebruik vectorized selection (veel sneller en veiliger dan .apply)
            # We pakken wp_actual waar de mode matcht, anders 0
            mask = temp_df["hvac_mode"].round() == mode_id
            wp_mode_series = temp_df["wp_actual"].where(mask, 0)

            # 3. Bouw de grafiekpunten op (exact hetzelfde als voorheen)
            for i in range(len(temp_df)):
                val = wp_mode_series.iloc[i]
                ts = temp_df.iloc[i]["timestamp_local"]

                is_active = val > 0.01
                was_active = i > 0 and wp_mode_series.iloc[i - 1] > 0.01
                is_last = i == len(temp_df) - 1 or wp_mode_series.iloc[i + 1] <= 0.01

                if is_active:
                    if not was_active:
                        hist_x.append(ts - timedelta(seconds=1))
                        hist_y.append(0.0)

                    hist_x.append(ts)
                    hist_y.append(val)

                    if is_last:
                        # Gebruik hier de echte volgende timestamp voor een strakke aansluiting
                        end_ts = ts + timedelta(minutes=15)
                        hist_x.append(end_ts)
                        hist_y.append(0.0)
                        hist_x.append(end_ts)
                        hist_y.append(None)

            if hist_x:
                fig.add_trace(
                    go.Scatter(
                        x=hist_x,
                        y=hist_y,
                        name=f"{label}",
                        legendgroup=label,
                        mode="lines",
                        line=dict(color=color, width=1.5, shape="hv"),
                        fill="tozeroy",
                        fillcolor=fill,
                        connectgaps=False,
                        hovertemplate="%{y:.2f} kW<extra></extra>",
                    )
                )

    if "power_corrected" in df.columns:
        df_future = df[df["timestamp_local"] >= local_now].copy()

        # We beginnen bij het huidige bolletje (Now)
        corr_x = [local_now]
        corr_y = [context.stable_pv]

        for i in range(len(df_future)):
            val = df_future.iloc[i]["power_corrected"]
            ts = df_future.iloc[i]["timestamp_local"]

            is_active = val > 0.01
            # Check buren voor natuurlijke helling (ramp)
            next_active = (
                i < len(df_future) - 1
                and df_future.iloc[i + 1]["power_corrected"] > 0.01
            )
            prev_active = (
                i > 0 and df_future.iloc[i - 1]["power_corrected"] > 0.01
            ) or (context.stable_pv > 0.01 and i == 0)

            if is_active or next_active or prev_active:
                corr_x.append(ts)
                corr_y.append(val)
            else:
                # Breekt de lijn als het donker is
                if corr_x and corr_y[-1] is not None:
                    corr_x.append(ts)
                    corr_y.append(None)

        if len(corr_x) > 1:
            fig.add_trace(
                go.Scatter(
                    x=corr_x,
                    y=corr_y,
                    mode="lines",
                    showlegend=False,
                    legendgroup="solar",
                    line=dict(color="#ffffff", dash="dash", width=1.5),
                    fill="tozeroy",
                    opacity=0.8,
                    fillcolor="rgba(255, 255, 255, 0.05)",
                    connectgaps=False,  # Zorg dat de Nones de lijn echt verbreken
                    hovertemplate="%{y:.2f} kW<extra></extra>",
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

        # --- PLANNING: TOON VOLLEDIGE DAGPLANNING ---
        if plan:
            for mode_key, color, fill, label in [
                ("UFH", "#d05ce3", "rgba(208, 92, 227, 0.15)", "UFH"),
                ("DHW", "#02cfe7", "rgba(0, 229, 255, 0.15)", "DHW"),
            ]:
                x_vals = []
                y_vals = []

                for i in range(len(plan)):
                    slot_time = plan[i]["time"].replace(tzinfo=None)
                    val = float(plan[i][f"p_el_{mode_key.lower()}"])
                    is_active = plan[i]["mode"] == mode_key

                    # Bepaal of dit het begin of einde is van een blok
                    was_active = i > 0 and plan[i - 1]["mode"] == mode_key
                    is_last_active = is_active and (
                        i == len(plan) - 1 or plan[i + 1]["mode"] != mode_key
                    )

                    if is_active:
                        if not was_active:
                            # Start van blok: verticaal omhoog vanaf 0
                            x_vals.append(slot_time - timedelta(seconds=1))
                            y_vals.append(0.0)

                        # Voeg het datapunt toe (de hoogte uit de tabel)
                        x_vals.append(slot_time)
                        y_vals.append(val)

                        if is_last_active:
                            # Einde van blok: verticaal omlaag naar 0
                            end_of_slot = slot_time + timedelta(minutes=15)
                            x_vals.append(end_of_slot)
                            y_vals.append(0.0)
                            x_vals.append(end_of_slot)
                            y_vals.append(
                                None
                            )  # Onderbreek de lijn voor het volgende blok

                if x_vals:
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            name=label,
                            mode="lines",
                            line=dict(
                                color=color, width=1.5, shape="hv"
                            ),  # 'hv' zorgt voor strakke trappetjes
                            fill="tozeroy",
                            fillcolor=fill,
                            connectgaps=False,
                            showlegend=False,
                            hovertemplate="%{y:.2f} kW<extra></extra>",
                            legendgroup=label,
                        )
                    )

    fig.add_vline(
        x=local_now, line_width=1, line_dash="solid", line_color="white", opacity=0.6
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
        idx_now = max(0, min(idx_now, len(df) - 1))

        # 4. Slimme Logica: Nu of Piek?
        prediction_now = df.iloc[idx_now][target_col]

        # Als het donker is (< 0.05 kW), zoek de piek van de dag
        if prediction_now < 0.05:
            idx_max = df[target_col].argmax()
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
    start_dt = datetime.combine(target_date, datetime.min.time()).replace(
        tzinfo=local_tz
    )
    start_utc = start_dt.astimezone(timezone.utc)

    # Haal data op (get_history haalt alles op NA de datum, dus we moeten straks filteren op eindtijd)
    df = database.get_history(start_utc)

    if df.empty:
        return []

    df["timestamp"] = df["timestamp"].dt.tz_convert(local_tz)

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

        # Energie-waarden tellen we op (sum)
        # De modus bepalen we door de meest voorkomende waarde te kiezen (lambda x.mode)
        agg_map = {
            "grid_import": "sum",
            "grid_export": "sum",
            "pv_actual": "sum",
            "wp_actual": "sum",
            "hvac_mode": lambda x: (
                x[x != 0].mode().iloc[0] if not x[x != 0].mode().empty else 0
            ),
        }

        # Voer de bewerking uit
        df = df.resample("1h", label="left").agg(agg_map)

        # 3. Filter toekomst weg (anders heb je lege rijen voor vanavond)
        if target_date == datetime.now(local_tz).date():
            current_hour = datetime.now(local_tz).replace(
                minute=0, second=0, microsecond=0
            )
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
        raw_mode = row.get("hvac_mode", 0)
        mode = int(round(raw_mode)) if pd.notna(raw_mode) else 0

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
                "mode": mode,
            }
        )

    return table_data
