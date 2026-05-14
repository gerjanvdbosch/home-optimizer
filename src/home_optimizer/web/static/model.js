const modelBaseUrl = new URL(".", window.location.href);

function modelApiUrl(path) {
  return new URL(path, modelBaseUrl).toString();
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
const modelDetailType = document.getElementById("model-detail-type");
const modelDetailInterval = document.getElementById("model-detail-interval");
const modelDetailSamples = document.getElementById("model-detail-samples");
const modelDetailActive = document.getElementById("model-detail-active");
const modelDetailFitQuality = document.getElementById("model-detail-fit-quality");
const modelDetailFitReasons = document.getElementById("model-detail-fit-reasons");
const modelAggregateMetricsBody = document.getElementById("model-aggregate-metrics-body");
const modelSegmentMetricsBody = document.getElementById("model-segment-metrics-body");
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

function formatMetricValue(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return Number(value).toFixed(digits);
}

function buildMetricRows(metrics) {
  return (metrics || [])
    .map(
      (metric) => `
        <tr>
          <td>${metric.horizon_minutes} min</td>
          <td>${formatMetricValue(metric.mae_c)}</td>
          <td>${formatMetricValue(metric.rmse_c)}</td>
          <td>${formatMetricValue(metric.bias_c)}</td>
          <td>${formatMetricValue(metric.p95_abs_error_c)}</td>
          <td>${metric.sample_count ?? "-"}</td>
        </tr>
      `,
    )
    .join("");
}

function renderModelMetricDetails(payload) {
  if (modelDetailType) {
    modelDetailType.textContent = payload.model_type || "-";
  }
  if (modelDetailInterval) {
    modelDetailInterval.textContent = payload.interval_minutes ? `${payload.interval_minutes} min` : "-";
  }
  if (modelDetailSamples) {
    modelDetailSamples.textContent = String(payload.sample_count ?? "-");
  }
  if (modelDetailActive) {
    modelDetailActive.textContent = payload.is_active ? "active" : "inactive";
  }
  if (modelDetailFitQuality) {
    modelDetailFitQuality.textContent = payload.fit_quality || "-";
  }
  if (modelDetailFitReasons) {
    modelDetailFitReasons.textContent = (payload.fit_quality_reasons || []).join(", ") || "-";
  }

  if (modelAggregateMetricsBody) {
    modelAggregateMetricsBody.innerHTML = "";
    for (const metric of payload.aggregate_metrics || []) {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${metric.horizon_minutes} min</td>
        <td>${formatMetricValue(metric.mae_c)}</td>
        <td>${formatMetricValue(metric.rmse_c)}</td>
        <td>${formatMetricValue(metric.bias_c)}</td>
        <td>${formatMetricValue(metric.p95_abs_error_c)}</td>
        <td>${metric.sample_count ?? "-"}</td>
      `;
      modelAggregateMetricsBody.append(row);
    }
    if (!modelAggregateMetricsBody.children.length) {
      const row = document.createElement("tr");
      row.innerHTML = '<td colspan="6">Geen aggregate metrics beschikbaar</td>';
      modelAggregateMetricsBody.append(row);
    }
  }

  if (modelSegmentMetricsBody) {
    modelSegmentMetricsBody.innerHTML = "";
    for (const segment of payload.segment_metrics || []) {
      const card = document.createElement("section");
      card.className = "panel";
      card.innerHTML = `
        <div class="panel-heading">
          <h2>${segment.segment_name}</h2>
          <span>${segment.description}</span>
        </div>
        <div class="table-wrap">
          <table class="summary-table">
            <thead>
              <tr>
                <th>Horizon</th>
                <th>MAE</th>
                <th>RMSE</th>
                <th>Bias</th>
                <th>P95 abs</th>
                <th>Samples</th>
              </tr>
            </thead>
            <tbody>
              ${buildMetricRows(segment.metrics) || '<tr><td colspan="6">Geen segment metrics beschikbaar</td></tr>'}
            </tbody>
          </table>
        </div>
      `;
      modelSegmentMetricsBody.append(card);
    }
    if (!modelSegmentMetricsBody.children.length) {
      const empty = document.createElement("div");
      empty.textContent = "Geen segment metrics beschikbaar";
      modelSegmentMetricsBody.append(empty);
    }
  }
}

async function loadSelectedModelDetails() {
  if (!simulationModelSelect?.value) {
    return;
  }
  const response = await fetch(modelApiUrl(`api/models/room/${encodeURIComponent(simulationModelSelect.value)}`));
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Kon modeldetails niet laden.");
  }
  renderModelMetricDetails(payload);
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
  const response = await fetch(modelApiUrl("api/models/room"));
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Kon modellijst niet laden.");
  }
  populateSimulationModelSelect(payload.models || []);
  await loadSelectedModelDetails();
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
      params.set("end_time", `${trainEndDateInput.value}T00:00:00Z`);
    }
    if (trainModelTypeSelect?.value) {
      params.set("model_type", trainModelTypeSelect.value);
    }
    params.set("activate", trainActivateInput?.checked ? "true" : "false");

    const response = await fetch(modelApiUrl(`api/train?${params.toString()}`), {
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
    await loadSelectedModelDetails();
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

    const response = await fetch(modelApiUrl(`api/simulate/room?${params.toString()}`));
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
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  anchorTimeInput.value = localInputValue(today);
  syncSimulationDateLabel(today);
}

if (trainStartDateInput) {
  trainStartDateInput.value = localDateValue(new Date("2026-04-16T00:00:00Z"));
}

if (trainEndDateInput) {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  trainEndDateInput.value = localDateValue(today);
}

if (trainModelTypeSelect) {
  trainModelTypeSelect.value = "room_2r2c";
}

if (simulateButton) {
  simulateButton.addEventListener("click", () => {
    loadSimulation().catch(handleSimulationError);
  });
}
simulationModelSelect?.addEventListener("change", () => {
  loadSelectedModelDetails().catch(handleSimulationError);
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
