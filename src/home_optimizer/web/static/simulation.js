const simulationBaseUrl = new URL(".", window.location.href);

function simulationApiUrl(path) {
  return new URL(path, simulationBaseUrl).toString();
}

function localInputValue(date) {
  const localDate = new Date(date.getTime() - date.getTimezoneOffset() * 60 * 1000);
  return localDate.toISOString().slice(0, 16);
}

function formatSimulationDisplayDate(date) {
  return new Intl.DateTimeFormat("nl-NL", {
    weekday: "short",
    day: "2-digit",
    month: "short",
  }).format(date);
}

function simulationChartTimestamp(timestamp) {
  const date = new Date(timestamp);
  const localTimestamp = new Date(date.getTime() - date.getTimezoneOffset() * 60 * 1000);
  return localTimestamp.toISOString().slice(0, 19);
}

function simulationLayout(yTitle, y2Title = null) {
  const layout = {
    autosize: true,
    margin: { t: 10, r: 12, b: 36, l: 46 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    xaxis: {
      type: "date",
      tickformat: "%d %H:%M",
      showgrid: true,
      gridcolor: "#eceff1",
      zeroline: false,
      fixedrange: true,
    },
    yaxis: {
      title: { text: yTitle },
      showgrid: true,
      gridcolor: "#eceff1",
      zeroline: false,
      fixedrange: true,
    },
    legend: {
      orientation: "h",
      x: 0,
      y: -0.18,
      font: { size: 12 },
    },
    font: {
      family: 'Roboto, "Noto Sans", "Segoe UI", Arial, sans-serif',
      color: "#212121",
    },
  };

  if (y2Title) {
    layout.yaxis2 = {
      title: { text: y2Title },
      overlaying: "y",
      side: "right",
      showgrid: false,
      zeroline: false,
      fixedrange: true,
    };
  }

  return layout;
}

function simulationTrace(series, label, color, options = {}) {
  return {
    x: series.points.map((point) => simulationChartTimestamp(point.timestamp)),
    y: series.points.map((point) => point.value),
    name: label,
    type: "scatter",
    mode: "lines",
    ...(options.yaxis ? { yaxis: options.yaxis } : {}),
    line: {
      color,
      width: options.width || 2,
      ...(options.dash ? { dash: options.dash } : {}),
    },
    hovertemplate:
      `%{x|%d-%m %H:%M}<br>%{y:.2f} ${series.unit || ""}` + `<extra>${label}</extra>`,
  };
}

const anchorTimeInput = document.getElementById("anchor-time");
const horizonStepsInput = document.getElementById("horizon-steps");
const trainButton = document.getElementById("train-button");
const simulateButton = document.getElementById("simulate-button");
const simulationPreviousDayButton = document.getElementById("simulation-previous-day");
const simulationNextDayButton = document.getElementById("simulation-next-day");
const simulationSelectedDate = document.getElementById("simulation-selected-date");
const simulationStatus = document.getElementById("simulation-status");
const simulationResult = document.getElementById("simulation-result");
const simulationSummary = document.getElementById("simulation-summary");
const simulationModelId = document.getElementById("simulation-model-id");
const roomChart = document.getElementById("simulation-room-chart");
const driverChart = document.getElementById("simulation-driver-chart");

function setSimulationButtonsDisabled(disabled) {
  if (trainButton) {
    trainButton.disabled = disabled;
  }
  if (simulateButton) {
    simulateButton.disabled = disabled;
  }
  if (simulationPreviousDayButton) {
    simulationPreviousDayButton.disabled = disabled;
  }
  if (simulationNextDayButton) {
    simulationNextDayButton.disabled = disabled;
  }
}

function syncSimulationDateLabel(anchorTime) {
  if (simulationSelectedDate) {
    simulationSelectedDate.textContent = formatSimulationDisplayDate(anchorTime);
  }
}

function shiftSimulationDay(days) {
  if (!anchorTimeInput?.value) {
    return;
  }
  const anchorTime = new Date(anchorTimeInput.value);
  anchorTime.setDate(anchorTime.getDate() + days);
  anchorTimeInput.value = localInputValue(anchorTime);
  syncSimulationDateLabel(anchorTime);
  loadSimulation().catch(handleSimulationError);
}

async function runTrain() {
  if (!trainButton) {
    return;
  }

  setSimulationButtonsDisabled(true);
  simulationStatus.textContent = "Model wordt getraind...";
  simulationStatus.className = "status";

  try {
    const response = await fetch(simulationApiUrl("api/train?activate=true"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Modeltraining mislukt.");
    }

    simulationStatus.textContent = `Model getraind: ${payload.model_id}`;
    simulationStatus.className = "status success";
    if (simulationResult) {
      simulationResult.hidden = false;
      simulationResult.textContent = JSON.stringify(payload, null, 2);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "Modeltraining mislukt.";
    simulationStatus.textContent = message;
    simulationStatus.className = "status error";
    if (simulationResult) {
      simulationResult.hidden = false;
      simulationResult.textContent = "Het model kon niet worden getraind.";
    }
  } finally {
    setSimulationButtonsDisabled(false);
  }
}

async function loadSimulation() {
  setSimulationButtonsDisabled(true);
  simulationStatus.textContent = "Simulatie laden...";
  simulationStatus.className = "status";

  try {
    const anchorTime = new Date(anchorTimeInput.value);
    syncSimulationDateLabel(anchorTime);
    const horizonSteps = Number(horizonStepsInput.value || "144");
    const params = new URLSearchParams({
      anchor_time: anchorTime.toISOString(),
      horizon_steps: String(horizonSteps),
    });

    const response = await fetch(simulationApiUrl(`api/simulate/room?${params.toString()}`));
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Simulatie mislukt.");
    }

    Plotly.react(
      roomChart,
      [
        simulationTrace(payload.predicted_room_temperature, "Predicted", "#1e88e5"),
        simulationTrace(payload.actual_room_temperature, "Actual", "#43a047"),
        simulationTrace(payload.room_target_min_temperature, "Target min", "#c62828", {
          dash: "dot",
        }),
        simulationTrace(payload.room_target_max_temperature, "Target max", "#ef6c00", {
          dash: "dot",
        }),
      ],
      simulationLayout("°C"),
      { displayModeBar: false, responsive: true },
    );

    Plotly.react(
      driverChart,
      [
        simulationTrace(payload.outdoor_temperature, "Outdoor", "#455a64"),
        simulationTrace(payload.solar_irradiance, "Solar", "#f9a825", {
          yaxis: "y2",
        }),
      ],
      simulationLayout("°C", "W/m2"),
      { displayModeBar: false, responsive: true },
    );

    simulationSummary.textContent =
      `${payload.horizon_steps} stappen · ${payload.interval_minutes} min`;
    simulationModelId.textContent = payload.model_id;
    simulationStatus.textContent = "Simulatie geladen";
    simulationStatus.className = "status success";
  } finally {
    setSimulationButtonsDisabled(false);
  }
}

function handleSimulationError(error) {
  simulationStatus.textContent = error.message || "Simulatie mislukt";
  simulationStatus.className = "status error";
}

if (anchorTimeInput) {
  anchorTimeInput.value = localInputValue(new Date("2026-05-07T00:00:00Z"));
  syncSimulationDateLabel(new Date(anchorTimeInput.value));
}

if (simulateButton) {
  simulateButton.addEventListener("click", () => {
    loadSimulation().catch(handleSimulationError);
  });
}

trainButton?.addEventListener("click", () => {
  runTrain().catch(handleSimulationError);
});
simulationPreviousDayButton?.addEventListener("click", () => {
  shiftSimulationDay(-1);
});
simulationNextDayButton?.addEventListener("click", () => {
  shiftSimulationDay(1);
});

loadSimulation().catch(handleSimulationError);
