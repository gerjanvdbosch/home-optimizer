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
            )
        )

    fig.update_layout(
        title="Solar forecast",
        xaxis_title="Time",
        yaxis_title="Power (W)",
    )

    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
    )
