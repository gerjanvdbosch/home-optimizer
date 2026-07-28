from datetime import datetime

import pandas as pd
import plotly.graph_objects as go

from domain.models.state import OptimizerState
from domain.time import to_local_time


def add_series(
    fig: go.Figure,
    name: str,
    points: list,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=[to_local_time(p.time) for p in points],
            y=[p.value for p in points],
            mode="lines",
            name=name,
            line=dict(width=3),
            connectgaps=True,
            hovertemplate="%{y:.0f} W<extra>%{fullData.name}</extra>",
        )
    )


def solar_forecast_chart(state: OptimizerState) -> str:
    fig = go.Figure()

    for name, points in state.forecast.solar.items():
        add_series(fig, name, points)

    add_series(fig, "PV production", state.measurements.solar.production)

    fig.update_layout(
        title=dict(
            text="Solar forecast",
            x=0.02,
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
            b=80,
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
            font=dict(size=14),
        ),
        hovermode="x unified",
    )

    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
    )


def backtest_chart(results: list[dict]) -> str:
    df = pd.DataFrame(results)

    df["time"] = pd.to_datetime(df["index"]).dt.tz_convert(datetime.now().astimezone().tzinfo)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["actual"],
            mode="lines",
            name="Actual",
            line=dict(width=3),
            connectgaps=True,
            hovertemplate="%{y:.1f} °C<extra>%{fullData.name}</extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["pred"],
            mode="lines",
            name="Prediction",
            line=dict(width=3),
            connectgaps=True,
            hovertemplate="%{y:.1f} °C<extra>%{fullData.name}</extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="Backtest",
            x=0.02,
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
            b=80,
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
            title="Temperature (°C)",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            y=-0.15,
            x=0.5,
            xanchor="center",
            font=dict(size=14),
        ),
        hovermode="x unified",
    )

    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
    )
