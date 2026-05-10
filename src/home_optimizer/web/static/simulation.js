const simulationBaseUrl = new URL(".", window.location.href);

function simulationApiUrl(path) {
  return new URL(path, simulationBaseUrl).toString();
}

function localInputValue(date) {
  const localDate = new Date(date.getTime() - date.getTimezoneOffset() * 60 * 1000);
  return localDate.toISOString().slice(0, 16);
}

function localDateValue(date) {
  const localDate = new Date(date.getTime() - date.getTimezoneOffset() * 60 * 1000);
  return localDate.toISOString().slice(0, 10);
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
const simulationModelSelect = document.getElementById("simulation-model-select");
const trainButton = document.getElementById("train-button");
const simulateButton = document.getElementById("simulate-button");
const trainStartDateInput = document.getElementById("train-start-date");
const trainEndDateInput = document.getElementById("train-end-date");
const trainActivateInput = document.getElementById("train-activate");
const trainModelTypeSelect = document.getElementById("train-model-type");
const simulationPreviousDayButton = document.getElementById("simulation-previous-day");
const simulationNextDayButton = document.getElementById("simulation-next-day");
const simulationSelectedDate = document.getElementById("simulation-selected-date");
const simulationStatus = document.getElementById("simulation-status");
const trainStatus = document.getElementById("train-status");
const trainResult = document.getElementById("train-result");
const simulationSummary = document.getElementById("simulation-summary");
const simulationModelId = document.getElementById("simulation-model-id");
const roomChart = document.getElementById("simulation-room-chart");
const errorChart = document.getElementById("simulation-error-chart");
const solarChart = document.getElementById("simulation-solar-chart");
const driverChart = document.getElementById("simulation-driver-chart");

function setSimulationButtonsDisabled(disabled) {
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

function setTrainingControlsDisabled(disabled) {
  if (trainButton) {
    trainButton.disabled = disabled;
  }
  if (trainStartDateInput) {
    trainStartDateInput.disabled = disabled;
  }
  if (trainEndDateInput) {
    trainEndDateInput.disabled = disabled;
  }
  if (trainActivateInput) {
    trainActivateInput.disabled = disabled;
  }
  if (trainModelTypeSelect) {
    trainModelTypeSelect.disabled = disabled;
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

function modelLabel(model) {
  const activeSuffix = model.is_active ? " active" : "";
  return `${model.model_type} | ${model.model_id}${activeSuffix}`;
}

function populateSimulationModelSelect(models) {
  if (!simulationModelSelect) {
    return;
  }

  const previousValue = simulationModelSelect.value;
  simulationModelSelect.innerHTML = "";

  if (!models.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "Geen modellen beschikbaar";
    simulationModelSelect.append(option);
    simulationModelSelect.disabled = true;
    return;
  }

  simulationModelSelect.disabled = false;
  const modelsByType = new Map();
  for (const model of models) {
    const entries = modelsByType.get(model.model_type) || [];
    entries.push(model);
    modelsByType.set(model.model_type, entries);
  }

  for (const [modelType, typeModels] of modelsByType.entries()) {
    const group = document.createElement("optgroup");
    group.label = modelType;
    for (const model of typeModels) {
      const option = document.createElement("option");
      option.value = model.model_id;
      option.textContent = modelLabel(model);
      group.append(option);
    }
    simulationModelSelect.append(group);
  }

  const selectedValue =
    models.some((model) => model.model_id === previousValue)
      ? previousValue
      : (models.find((model) => model.is_active)?.model_id || models[0].model_id);
  simulationModelSelect.value = selectedValue;
}

async function refreshRoomModels() {
  const response = await fetch(simulationApiUrl("api/models/room"));
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Kon modellijst niet laden.");
  }
  populateSimulationModelSelect(payload.models || []);
}

async function runTrain() {
  if (!trainButton) {
    return;
  }

  setTrainingControlsDisabled(true);
  if (trainStatus) {
    trainStatus.textContent = "Model wordt getraind...";
    trainStatus.className = "status";
  }

  try {
    const params = new URLSearchParams();
    if (trainStartDateInput?.value) {
      params.set("start_time", `${trainStartDateInput.value}T00:00:00Z`);
    }
    if (trainEndDateInput?.value) {
      params.set("end_time", `${trainEndDateInput.value}T23:59:00Z`);
    }
    if (trainModelTypeSelect?.value) {
      params.set("model_type", trainModelTypeSelect.value);
    }
    params.set("activate", trainActivateInput?.checked ? "true" : "false");

    const response = await fetch(simulationApiUrl(`api/train?${params.toString()}`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Modeltraining mislukt.");
    }

    if (trainStatus) {
      trainStatus.textContent = `Model getraind: ${payload.model_id}`;
      trainStatus.className = "status success";
    }
    if (trainResult) {
      trainResult.hidden = false;
      trainResult.textContent = JSON.stringify(payload, null, 2);
    }
    await refreshRoomModels();
    if (simulationModelSelect && payload.model_id) {
      simulationModelSelect.value = payload.model_id;
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "Modeltraining mislukt.";
    if (trainStatus) {
      trainStatus.textContent = message;
      trainStatus.className = "status error";
    }
    if (trainResult) {
      trainResult.hidden = false;
      trainResult.textContent = "Het model kon niet worden getraind.";
    }
  } finally {
    setTrainingControlsDisabled(false);
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
    if (simulationModelSelect?.value) {
      params.set("model_id", simulationModelSelect.value);
    }

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
      errorChart,
      [
        simulationTrace(payload.prediction_error_c, "Predicted - actual", "#c62828"),
      ],
      simulationLayout("°C"),
      { displayModeBar: false, responsive: true },
    );

    Plotly.react(
      solarChart,
      [
        simulationTrace(payload.solar_irradiance, "Solar", "#f9a825", {
          width: 2,
        }),
        simulationTrace(payload.solar_gain_proxy, "Solar gain proxy", "#6d4c41", {
          dash: "dot",
          width: 2,
        }),
        simulationTrace(payload.shutter_position, "Shutter position", "#1e88e5", {
          yaxis: "y2",
        }),
      ],
      simulationLayout("W/m2", "%"),
      { displayModeBar: false, responsive: true },
    );

    Plotly.react(
      driverChart,
      [
        simulationTrace(payload.outdoor_temperature, "Outdoor", "#455a64"),
        simulationTrace(payload.thermal_output_estimate, "Thermal output", "#43a047", {
          yaxis: "y2",
        }),
      ],
      simulationLayout("°C", "kW"),
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
  anchorTimeInput.value = localInputValue(new Date("2026-05-07T22:00:00Z"));
  syncSimulationDateLabel(new Date(anchorTimeInput.value));
}

if (trainStartDateInput) {
  trainStartDateInput.value = localDateValue(new Date("2026-02-08T00:00:00Z"));
}

if (trainEndDateInput) {
  trainEndDateInput.value = localDateValue(new Date("2026-05-08T23:59:00Z"));
}

if (trainModelTypeSelect) {
  trainModelTypeSelect.value = "room_arx";
}

if (simulateButton) {
  simulateButton.addEventListener("click", () => {
    loadSimulation().catch(handleSimulationError);
  });
}
simulationModelSelect?.addEventListener("change", () => {
  loadSimulation().catch(handleSimulationError);
});

trainButton?.addEventListener("click", () => {
  runTrain().catch(handleSimulationError);
});
simulationPreviousDayButton?.addEventListener("click", () => {
  shiftSimulationDay(-1);
});
simulationNextDayButton?.addEventListener("click", () => {
  shiftSimulationDay(1);
});

refreshRoomModels()
  .then(() => loadSimulation())
  .catch(handleSimulationError);
