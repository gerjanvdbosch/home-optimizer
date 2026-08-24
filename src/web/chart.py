import pandas as pd
import plotly.graph_objects as go

from domain.models import BacktestResult, State
from domain.time import to_local_series, to_local_time


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
            line=dict(width=2),
            connectgaps=True,
            hovertemplate="%{y:.0f} W<extra>%{fullData.name}</extra>",
        )
    )


def solar_forecast_chart(
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

    cloud_layers = [
        # ("Cloud cover low", state.forecast.open_meteo.cloud_cover_low),
        # ("Cloud cover mid", state.forecast.open_meteo.cloud_cover_mid),
        # ("Cloud cover high", state.forecast.open_meteo.cloud_cover_high),
    ]

    for label, points in cloud_layers:
        fig.add_trace(
            go.Scatter(
                x=[to_local_time(p.time) for p in points],
                y=[p.value for p in points],
                mode="lines",
                name=label,
                line=dict(width=1),
                connectgaps=True,
                hovertemplate="%{y:.0f}%<extra>%{fullData.name}</extra>",
                yaxis="y2",
            )
        )

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
        # yaxis2=dict(
        #     title="Cloud cover (%)",
        #     overlaying="y",
        #     side="right",
        #     range=[0, 100],
        #     showgrid=False,
        #     zeroline=False,
        #     tickfont=dict(size=12),
        # ),
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
            font=dict(size=14),
            groupclick="togglegroup",
        ),
        hovermode="x unified",
    )

    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
    )
