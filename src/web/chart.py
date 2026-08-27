from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from domain.models import BacktestResult, State
from domain.time import to_local_series, to_local_time


def add_series(
    fig: go.Figure,
    name: str,
    points: list,
    line: dict | None = None,
    legendgroup: str | None = None,
    showlegend: bool = True,
    decimal: int | None = 0,
    unit: str | None = "",
    row: int | None = None,
    col: int | None = None,
) -> None:
    if line is None:
        line = dict(width=2)

    fig.add_trace(
        go.Scatter(
            x=[to_local_time(p.time) for p in points],
            y=[p.value for p in points],
            mode="lines",
            name=name,
            line=line,
            legendgroup=legendgroup,
            showlegend=showlegend,
            connectgaps=True,
            hovertemplate=f"%{{y:.{decimal}f}} {unit}<extra>%{{fullData.name}}</extra>",
        ),
        row=row,
        col=col,
    )


def dashboard_chart(state: State) -> str:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Power", "Boiler"),
        row_heights=[0.6, 0.4],
    )

    fig.add_trace(
        go.Scatter(
            x=[to_local_time(p.time) for p in state.forecast.solcast.p10],
            y=[p.value for p in state.forecast.solcast.p10],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            legendgroup="solar",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[to_local_time(p.time) for p in state.forecast.solcast.p90],
            y=[p.value for p in state.forecast.solcast.p90],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(255, 161, 90, 0.08)",
            showlegend=False,
            legendgroup="solar",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    add_series(
        fig,
        "Solar",
        state.measurements.solar.production,
        line=dict(width=2, color="#FFA15A"),
        legendgroup="solar",
        showlegend=False,
        unit="W",
        row=1,
        col=1,
    )

    add_series(
        fig,
        "Solar",
        state.predictions.solar,
        line=dict(width=2, color="#FFA15A"),
        legendgroup="solar",
        unit="W",
        row=1,
        col=1,
    )

    if state.schedule and state.schedule.heat_pump.power:
        add_series(
            fig,
            "Heat pump",
            state.schedule.heat_pump.power,
            unit="W",
            row=1,
            col=1,
            line=dict(width=2, shape="vh", color="#AB63FA"),
        )

    add_series(
        fig,
        "Boiler bottom",
        state.measurements.heat_pump.boiler.bottom_temperature,
        row=2,
        col=1,
        line=dict(width=2, color="#636EFA"),
        unit="°C",
        decimal=1,
    )

    add_series(
        fig,
        "Boiler top",
        state.measurements.heat_pump.boiler.top_temperature,
        row=2,
        col=1,
        line=dict(width=2, color="#19D3F3"),
        unit="°C",
        decimal=1,
    )

    fig.update_layout(
        title=dict(
            text="Dashboard",
            x=0.01,
            y=0.97,
            font=dict(
                size=22,
                color="#eeeeee",
            ),
        ),
        template="plotly_dark",
        paper_bgcolor="#2b2b2b",
        plot_bgcolor="#2b2b2b",
        margin=dict(
            l=70,
            r=20,
            t=60,
            b=10,
        ),
        height=600,
        font=dict(
            color="#cccccc",
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            tickfont=dict(size=12),
        ),
        xaxis2=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="Power (W)",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
        ),
        yaxis2=dict(
            title="Temp (°C)",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            y=-0.15,
            x=0.5,
            xanchor="center",
            font=dict(size=12),
        ),
        hovermode="x unified",
    )

    fig.add_vline(
        x=to_local_time(datetime.now(timezone.utc)).timestamp() * 1000,
        line_width=1,
        line_color="#ffffff",
        layer="above",
    )

    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
    )


def solar_chart(
    state: State,
    capacity: float,
    efficiency: float,
) -> str:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[to_local_time(p.time) for p in state.forecast.solcast.p50],
            y=[p.value for p in state.forecast.solcast.p50],
            mode="lines",
            name="Solcast",
            line=dict(width=2),
            connectgaps=True,
            legendgroup="solcast",
            hovertemplate="%{y:.0f} W<extra>%{fullData.name}</extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[to_local_time(p.time) for p in state.forecast.solcast.p10],
            y=[p.value for p in state.forecast.solcast.p10],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            legendgroup="solcast",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[to_local_time(p.time) for p in state.forecast.solcast.p90],
            y=[p.value for p in state.forecast.solcast.p90],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(99, 110, 250, 0.15)",
            showlegend=False,
            legendgroup="solcast",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[to_local_time(p.time) for p in state.forecast.open_meteo.gti],
            y=[p.value * capacity * efficiency for p in state.forecast.open_meteo.gti],
            mode="lines",
            name="Open-Meteo",
            line=dict(width=2),
            connectgaps=True,
            hovertemplate="%{y:.0f} W<extra>%{fullData.name}</extra>",
        )
    )

    add_series(fig, "PV production", state.measurements.solar.production)
    add_series(fig, "ML prediction", state.predictions.solar)

    fig.update_layout(
        title=dict(
            text="Solar forecast",
            x=0.01,
            y=0.95,
            font=dict(
                size=22,
                color="#eeeeee",
            ),
        ),
        template="plotly_dark",
        paper_bgcolor="#2b2b2b",
        plot_bgcolor="#2b2b2b",
        margin=dict(
            l=70,
            r=20,
            t=60,
            b=10,
        ),
        height=400,
        font=dict(
            color="#cccccc",
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="Power (W)",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            y=-0.15,
            x=0.5,
            xanchor="center",
            font=dict(size=12),
        ),
        hovermode="x unified",
    )

    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
    )


def backtest_chart(result: BacktestResult | None) -> str:
    if result is None:
        return ""

    fig = go.Figure()

    for i, bp in enumerate(result.points):
        df = pd.DataFrame(bp.points)
        df["x_time"] = to_local_series(pd.to_datetime(df["time"]))

        fig.add_trace(
            go.Scatter(
                x=df["x_time"],
                y=df["value"],
                mode="lines",
                name=bp.label,
                legendgroup=bp.group,
                showlegend=not bp.group,
                line=dict(width=1, color=bp.color),
                visible=True,
                connectgaps=True,
                hovertemplate=(
                    f"%{{y:.1f}} {result.unit}<extra>%{{fullData.name}}</extra>"
                ),
            )
        )

    groups: set[str] = set()

    for bp in result.points:
        if bp.group not in groups and bp.group:
            groups.add(bp.group)

            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    name=bp.group,
                    legendgroup=bp.group,
                    showlegend=True,
                    line=dict(width=1, color=bp.color),
                )
            )

    fig.update_layout(
        title=dict(
            text=f"{result.name.capitalize()} backtest - MAE {result.mae:.3f}",
            x=0.01,
            y=0.95,
            font=dict(
                size=22,
                color="#eeeeee",
            ),
        ),
        template="plotly_dark",
        paper_bgcolor="#2b2b2b",
        plot_bgcolor="#2b2b2b",
        margin=dict(
            l=70,
            r=20,
            t=60,
            b=10,
        ),
        height=400,
        font=dict(
            color="#cccccc",
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title=f"{result.label} ({result.unit})",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            y=-0.15,
            x=0.5,
            xanchor="center",
            font=dict(size=12),
            groupclick="togglegroup",
        ),
        hovermode="x unified",
    )

    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
    )
