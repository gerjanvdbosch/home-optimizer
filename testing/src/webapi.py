import logging
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd

from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from datetime import timedelta, datetime, timezone, date, time
from pathlib import Path
from context import HvacMode

logger = logging.getLogger(__name__)

api = FastAPI(title="Home Optimizer API")

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@api.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    train: str = None,
    view: str = "hour",
    date_str: str = None,
):
    """
    Dashboard Home. Leest status direct uit Context.
    """
    coordinator = request.app.state.coordinator
    context = coordinator.context

    avg_price = coordinator.config.avg_price
    export_price = coordinator.config.export_price

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
            local_ts = p["time"].astimezone(local_tz)
            p_copy["time"] = local_ts.strftime("%H:%M")
            plan_display.append(p_copy)

    # 1. Grafieken genereren
    plot_html = _get_solar_forecast_plot(request, target_date, plan)
    accuracy_plot_room, accuracy_plot_dhw = _get_accuracy_plots(request, target_date)
    consumption_plot = _get_consumption_plot(request, target_date)
    view_mode = "15min" if view == "15min" else "hour"
    measurements_data = _get_energy_table(request, view_mode, target_date)

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
        realized_pv = 0.0
        realized_self = 0.0
        realized_export = 0.0
        realized_import = 0.0

        # Haal historische data op vanaf het begin van de geselecteerde dag (UTC)
        start_of_day = datetime.combine(target_date, datetime.min.time()).replace(
            tzinfo=local_tz
        )
        database = coordinator.collector.database
        df_hist = database.get_history(start_of_day.astimezone(timezone.utc))

        if not df_hist.empty:
            df_hist["timestamp_local"] = df_hist["timestamp"].dt.tz_convert(local_tz)

            # Filter de historie: vanaf middernacht tot 'nu' (voorkomt dubbel tellen met de toekomst)
            local_now = context.now.astimezone(local_tz)
            cutoff = local_now if is_today else start_of_day + timedelta(days=1)

            mask = (df_hist["timestamp_local"] >= start_of_day) & (
                df_hist["timestamp_local"] < cutoff
            )
            df_past = df_hist[mask].copy()

            if not df_past.empty:
                # Veilige omzetting naar numeriek (voorkom lege of corrupte strings)
                for col in ["pv_actual", "grid_import", "grid_export"]:
                    if col in df_past.columns:
                        df_past[col] = (
                            pd.to_numeric(df_past[col], errors="coerce")
                            .fillna(0.0)
                            .clip(lower=0.0)
                        )

                # Bereken gerealiseerde kWh (kwartierwaarden = kW * 0.25)
                if "pv_actual" in df_past.columns:
                    realized_pv = df_past["pv_actual"].sum() * 0.25
                if "grid_import" in df_past.columns:
                    realized_import = df_past["grid_import"].sum() * 0.25
                if "grid_export" in df_past.columns:
                    realized_export = df_past["grid_export"].sum() * 0.25

                # Gerealiseerd eigen verbruik = Zonne-energie - Teruglevering (per 15 min kwartier bekeken)
                if "pv_actual" in df_past.columns and "grid_export" in df_past.columns:
                    self_kw = (df_past["pv_actual"] - df_past["grid_export"]).clip(
                        lower=0.0
                    )
                    realized_self = self_kw.sum() * 0.25

        # Haal de verwachte rest-van-de-dag (toekomst) op uit je optimizer resultaat
        future_pv = result.get("pv_remaining", 0.0) if is_today else 0.0
        future_self = result.get("solar_self_remaining", 0.0) if is_today else 0.0
        future_export = result.get("export_remaining", 0.0) if is_today else 0.0
        future_import = result.get("grid_remaining", 0.0) if is_today else 0.0

        # Totale dagwaarde = Gerealiseerd (verleden) + Verwacht (toekomst)
        total_pv = realized_pv + future_pv
        total_self = realized_self + future_self
        total_export = realized_export + future_export
        total_import = realized_import + future_import

        # Gerealiseerde kosten (verleden)
        realized_cost = realized_import * avg_price - realized_export * export_price

        # Verwachte kosten (toekomst) komen al uit result
        future_cost = result.get("total_cost_net", 0.0)

        total_day_cost = realized_cost + future_cost
        total_day_gross = (
            realized_import + future_import + realized_self + future_self
        ) * avg_price
        total_day_saving = (realized_self + future_self) * avg_price
        total_day_export_rev = (realized_export + future_export) * export_price

        details.extend(
            [
                {"label": "Modus", "value": result.get("mode", "-")},
                {
                    "label": "Bruto kosten",
                    "value": f"{total_day_gross:.2f}",
                    "unit": "€",
                },
                {
                    "label": "Netto kosten",
                    "value": f"{total_day_cost:.2f}",
                    "unit": "€",
                },
                {
                    "label": "Zon besparing",
                    "value": f"{total_day_saving:.2f}",
                    "unit": "€",
                },
                {
                    "label": "Export opbrengst",
                    "value": f"{total_day_export_rev:.2f}",
                    "unit": "€",
                },
                {
                    "label": "Zon opbrengst",
                    "value": f"{future_pv:.2f}",
                    "total": f"{total_pv:.2f}",
                    "unit": "kWh",
                },
                {
                    "label": "Eigen verbruik",
                    "value": f"{future_self:.2f}",
                    "total": f"{total_self:.2f}",
                    "unit": "kWh",
                },
                {
                    "label": "Import net",
                    "value": f"{future_import:.2f}",
                    "total": f"{total_import:.2f}",
                    "unit": "kWh",
                },
                {
                    "label": "Export net",
                    "value": f"{future_export:.2f}",
                    "total": f"{total_export:.2f}",
                    "unit": "kWh",
                },
            ]
        )

        details.extend(
            [
                {"label": "Kamer Temp", "value": f"{room_t:.1f}", "unit": "°C"},
                {"label": "Boiler Temp", "value": f"{dhw_t:.1f}", "unit": "°C"},
                {
                    "label": "Zon nu",
                    "value": (
                        f"{context.stable_pv:.2f}"
                        if getattr(context, "stable_pv", None) is not None
                        else "-"
                    ),
                    "unit": "kW",
                    # "color": "#FF9100"
                },
                {
                    "label": "Zon Bias",
                    "value": f"{context.solar_bias * 100:.1f}",
                    "unit": "%",
                },
                {
                    "label": "Load nu",
                    "value": (
                        f"{context.stable_load:.2f}"
                        if getattr(context, "stable_load", None) is not None
                        else "-"
                    ),
                    "unit": "kW",
                },
                {
                    "label": "Load Bias",
                    "value": f"{context.load_bias:.2f}",
                    "unit": "kW",
                },
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
                {"label": "Verwachte COP", "value": current_cop, "unit": ""},
            ]
        )

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
            "accuracy_plot_room": accuracy_plot_room,
            "accuracy_plot_dhw": accuracy_plot_dhw,
            "consumption_plot": consumption_plot,
            "details": details,
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
        df_hist_smooth = (
            df_hist.set_index("timestamp_local")
            .sort_index()
            .apply(pd.to_numeric, errors="coerce")
        )

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
                    slot_time = (
                        plan[i]["time"].astimezone(local_tz).replace(tzinfo=None)
                    )
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

    # Zorg dat de kolom als UTC wordt herkend en dan geconverteerd naar lokaal
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(local_tz)

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
    df["wp_actual"] = df.get("wp_actual", 0.0).clip(lower=0.0)
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


def _get_accuracy_plots(request, target_date) -> tuple:
    """
    Genereert vloeiende temperatuurgrafieken voor Kamer en Boiler met Setpoint en Marge.
    """
    coordinator = request.app.state.coordinator
    database = coordinator.collector.database
    local_tz = datetime.now().astimezone().tzinfo

    # 1. Data ophalen
    df_snap = database.get_predictions(target_date)
    start_of_day = datetime.combine(target_date, datetime.min.time()).replace(
        tzinfo=local_tz
    )
    df_hist = database.get_history(start_of_day.astimezone(timezone.utc))

    if df_snap.empty and df_hist.empty:
        return "", ""

    # 2. Data voorbereiden
    if not df_hist.empty:
        df_hist = df_hist.copy()
        df_hist["ts_local"] = (
            df_hist["timestamp"].dt.tz_convert(local_tz).dt.tz_localize(None)
        )
        mask = (df_hist["timestamp"].dt.tz_convert(local_tz) >= start_of_day) & (
            df_hist["timestamp"].dt.tz_convert(local_tz)
            < start_of_day + timedelta(days=1)
        )
        df_hist = df_hist[mask].sort_values("ts_local")

    if not df_snap.empty:
        df_snap = df_snap.copy()
        df_snap["ts_local"] = (
            pd.to_datetime(df_snap["timestamp"], utc=True)
            .dt.tz_convert(local_tz)
            .dt.tz_localize(None)
        )

    # 3. Targets ophalen (Unpack 6 waarden: min, max en ideaal target)
    T = coordinator.optimizer.mpc.horizon + 1
    r_min, r_max, d_min, d_max, r_target, d_target = (
        coordinator.optimizer.mpc._get_targets(start_of_day, T)
    )
    ts_targets = [
        start_of_day.replace(tzinfo=None) + timedelta(minutes=15 * t) for t in range(T)
    ]

    start_ts = start_of_day.replace(tzinfo=None)
    end_ts = start_ts + timedelta(days=1)

    # 4. Gemeenschappelijke layout instellingen (ZONDER yaxis)
    common_layout = dict(
        template="plotly_dark",
        paper_bgcolor="rgb(28, 28, 28)",
        plot_bgcolor="rgb(28, 28, 28)",
        height=320,
        margin=dict(l=40, r=20, t=30, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            range=[start_ts, end_ts],
            fixedrange=True,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            tickformat="%H:%M",
            dtick=7200000,
        ),
    )

    # 5. Helper voor Bandbreedte styling
    def apply_band_style(fig, x, y_min, y_max, y_target, color):
        # Marge (vlak) - Gebruik spline voor vloeiende overgangen
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_max,
                mode="lines",
                line=dict(width=0, shape="spline", smoothing=0.8),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_min,
                mode="lines",
                line=dict(width=0, shape="spline", smoothing=0.8),
                fill="tonexty",
                fillcolor=color.replace("0.1", "0.08"),
                name="Marge",
                showlegend=True,
                hoverinfo="skip",
            )
        )
        # Setpoint (stippellijn)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_target,
                mode="lines",
                line=dict(
                    color=color.replace("0.1", "0.3"),
                    width=1.5,
                    dash="dot",
                    shape="spline",
                    smoothing=0.8,
                ),
                name="Setpoint",
                showlegend=True,
                hoverinfo="skip",
            )
        )

    # --- 6. Kamer Grafiek ---
    fig_room = go.Figure()
    apply_band_style(
        fig_room, ts_targets, r_min, r_max, r_target, "rgba(208, 92, 227, 0.1)"
    )

    if not df_hist.empty:
        fig_room.add_trace(
            go.Scatter(
                x=df_hist["ts_local"],
                y=df_hist["room_temp"],
                name="Actueel",
                line=dict(color="#d05ce3", width=2.5),
                hovertemplate="%{y:.1f} °C<extra></extra>",
            )
        )
    if not df_snap.empty:
        fig_room.add_trace(
            go.Scatter(
                x=df_snap["ts_local"],
                y=df_snap["t_room_pred"],
                name="Voorspelling",
                line=dict(color="#d05ce3", width=1.5, dash="dash"),
                opacity=0.7,
                hovertemplate="%{y:.1f} °C<extra></extra>",
            )
        )
    fig_room.update_layout(**common_layout)
    fig_room.update_yaxes(
        title="Kamer Temp (°C)",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.09)",
        zeroline=False,
    )

    # --- 7. Boiler Grafiek ---
    fig_dhw = go.Figure()

    # Gebruik nu direct de ruwe arrays zonder clipping
    apply_band_style(
        fig_dhw, ts_targets, d_min, d_max, d_target, "rgba(2, 207, 231, 0.1)"
    )

    if not df_hist.empty:
        dhw_act = (df_hist["dhw_top"] + df_hist["dhw_bottom"]) / 2
        fig_dhw.add_trace(
            go.Scatter(
                x=df_hist["ts_local"],
                y=dhw_act,
                name="Actueel",
                line=dict(color="#02cfe7", width=2.5),
                hovertemplate="%{y:.1f} °C<extra></extra>",
            )
        )
    if not df_snap.empty:
        fig_dhw.add_trace(
            go.Scatter(
                x=df_snap["ts_local"],
                y=df_snap["t_dhw_pred"],
                name="Voorspelling",
                line=dict(color="#02cfe7", width=1.5, dash="dash"),
                opacity=0.7,
                hovertemplate="%{y:.1f} °C<extra></extra>",
            )
        )
    fig_dhw.update_layout(**common_layout)
    # Range is verwijderd zodat de as automatisch schaalt (bijv. van 10 naar 55)
    fig_dhw.update_yaxes(
        title="Boiler Temp (°C)",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.09)",
        zeroline=False,
    )

    # --- 8. Export ---
    html_room = pio.to_html(
        fig_room,
        full_html=False,
        include_plotlyjs=True,
        config={"displayModeBar": False},
    )
    html_dhw = pio.to_html(
        fig_dhw,
        full_html=False,
        include_plotlyjs=False,
        config={"displayModeBar": False},
    )

    return html_room, html_dhw


def _get_consumption_plot(request, target_date) -> str:
    """
    Genereert een uurgrafiek (bar chart) van het actuele vs voorspelde warmtepomp verbruik (kWh).
    """
    coordinator = request.app.state.coordinator
    database = coordinator.collector.database
    local_tz = datetime.now().astimezone().tzinfo

    df_snap = database.get_predictions(target_date)
    start_of_day = datetime.combine(target_date, time.min).replace(tzinfo=local_tz)
    end_of_day = start_of_day + timedelta(days=1)

    df_hist = database.get_history(start_of_day.astimezone(timezone.utc))

    # --- 1. Actueel Verbruik ---
    if not df_hist.empty:
        df_hist = df_hist.copy()
        df_hist["ts_local"] = df_hist["timestamp"].dt.tz_convert(local_tz)
        mask = (df_hist["ts_local"] >= start_of_day) & (
            df_hist["ts_local"] < end_of_day
        )
        df_hist = df_hist[mask].copy()

        df_hist["wp_actual"] = pd.to_numeric(
            df_hist.get("wp_actual", 0), errors="coerce"
        ).fillna(0)
        df_hist["hvac_mode"] = (
            pd.to_numeric(df_hist.get("hvac_mode", 0), errors="coerce")
            .fillna(0)
            .round()
        )

        df_hist["act_ufh"] = 0.0
        df_hist.loc[df_hist["hvac_mode"] == 2, "act_ufh"] = df_hist["wp_actual"] * 0.25
        df_hist["act_dhw"] = 0.0
        df_hist.loc[df_hist["hvac_mode"].isin([1, 4]), "act_dhw"] = (
            df_hist["wp_actual"] * 0.25
        )

        # Resample en verwijder direct de tijdzone info van de index
        df_hist_hourly = (
            df_hist.set_index("ts_local")[["act_ufh", "act_dhw"]].resample("1h").sum()
        )
        df_hist_hourly.index = df_hist_hourly.index.tz_localize(None)
    else:
        df_hist_hourly = pd.DataFrame(
            columns=["act_ufh", "act_dhw"], index=pd.DatetimeIndex([])
        )

    # --- 2. Verwacht Verbruik ---
    if not df_snap.empty:
        df_snap = df_snap.copy()
        df_snap["ts_local"] = pd.to_datetime(
            df_snap["timestamp"], utc=True
        ).dt.tz_convert(local_tz)
        mask_snap = (df_snap["ts_local"] >= start_of_day) & (
            df_snap["ts_local"] < end_of_day
        )
        df_snap = df_snap[mask_snap].copy()

        df_snap["pred_ufh"] = (
            pd.to_numeric(df_snap.get("p_el_ufh_pred", 0), errors="coerce").fillna(0)
            * 0.25
        )
        df_snap["pred_dhw"] = (
            pd.to_numeric(df_snap.get("p_el_dhw_pred", 0), errors="coerce").fillna(0)
            * 0.25
        )

        # Resample en verwijder direct de tijdzone info van de index
        df_snap_hourly = (
            df_snap.set_index("ts_local")[["pred_ufh", "pred_dhw"]].resample("1h").sum()
        )
        df_snap_hourly.index = df_snap_hourly.index.tz_localize(None)
    else:
        df_snap_hourly = pd.DataFrame(
            columns=["pred_ufh", "pred_dhw"], index=pd.DatetimeIndex([])
        )

    # --- 3. Samenvoegen ---
    df_plot = pd.concat([df_hist_hourly, df_snap_hourly], axis=1)

    # Maak de volledige 24-uurs index (naive/geen TZ)
    full_idx = pd.date_range(
        start=start_of_day.replace(tzinfo=None), periods=24, freq="1h"
    )

    # Reindex en fix de FutureWarning door infer_objects te gebruiken voor de fillna
    df_plot = df_plot.reindex(full_idx)
    df_plot = df_plot.infer_objects(copy=False).fillna(0.0)

    # --- 4. Opschonen voor Plotly (verberg 0-bars) ---
    for col in ["act_ufh", "act_dhw", "pred_ufh", "pred_dhw"]:
        if col in df_plot.columns:
            # Zet om naar float en vervang kleine waarden door None
            df_plot[col] = (
                df_plot[col].astype(float).apply(lambda x: x if x > 0.001 else None)
            )

    # --- 5. Plotly ---
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df_plot.index,
            y=df_plot["act_ufh"],
            name="Actueel UFH",
            marker_color="#d05ce3",
            hovertemplate="%{y:.2f} kWh<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=df_plot.index,
            y=df_plot["pred_ufh"],
            name="Verwacht UFH",
            marker_color="rgba(208, 92, 227, 0.2)",
            marker_line=dict(color="#d05ce3", width=2),
            hovertemplate="%{y:.2f} kWh<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=df_plot.index,
            y=df_plot["act_dhw"],
            name="Actueel DHW",
            marker_color="#02cfe7",
            hovertemplate="%{y:.2f} kWh<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=df_plot.index,
            y=df_plot["pred_dhw"],
            name="Verwacht DHW",
            marker_color="rgba(2, 207, 231, 0.2)",
            marker_line=dict(color="#02cfe7", width=2),
            hovertemplate="%{y:.2f} kWh<extra></extra>",
        )
    )

    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgb(28, 28, 28)",
        plot_bgcolor="rgb(28, 28, 28)",
        height=320,
        margin=dict(l=40, r=20, t=30, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            title=None,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            tickformat="%H:%M",
            dtick=7200000,
        ),
        yaxis=dict(
            title="Verbruik (kWh)", showgrid=True, gridcolor="rgba(255,255,255,0.1)"
        ),
        bargap=0.15,
        bargroupgap=0.05,
    )

    return pio.to_html(
        fig, full_html=False, include_plotlyjs=True, config={"displayModeBar": False}
    )
