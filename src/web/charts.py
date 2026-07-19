import plotly.graph_objects as go

from domain.models import OptimizerState


def solar_forecast_chart(state: OptimizerState) -> str:
    forecast = state.solar_forecast.model_dump()

    fig = go.Figure()

    for name, points in forecast.items():
        fig.add_trace(
            go.Scatter(
                x=[p["time"] for p in points],
                y=[p["watts"] for p in points],
                mode="lines",
                name=name,
                line=dict(
                    width=2,
                    dash="dot",
                ),
            )
        )

    if state.pv_production:
        fig.add_trace(
            go.Scatter(
                x=[p.time for p in state.pv_production],
                y=[p.watts for p in state.pv_production],
                mode="lines",
                name="PV production",
                line=dict(
                    width=3,
                ),
                connectgaps=True,
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
            t=70,
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
