import plotly.graph_objects as go

from domain.models import OptimizerState


def add_series(
    fig: go.Figure,
    name: str,
    points: list,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=[p.time for p in points],
            y=[p.watts for p in points],
            mode="lines",
            name=name,
            line=dict(width=3),
            connectgaps=True,
            hovertemplate="%{y:.0f} W<extra>%{fullData.name}</extra>",
        )
    )


def solar_forecast_chart(state: OptimizerState) -> str:
    fig = go.Figure()

    for name, points in state.solar_forecast.items():
        add_series(fig, name, points)

    add_series(fig, "PV production", state.pv_production)

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
