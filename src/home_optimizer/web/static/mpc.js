const mpcBaseUrl = new URL(".", window.location.href);

function mpcApiUrl(path) {
  return new URL(path, mpcBaseUrl).toString();
}

function mpcLocalInputValue(date) {
  const localDate = new Date(date.getTime() - date.getTimezoneOffset() * 60 * 1000);
  return localDate.toISOString().slice(0, 16);
}

function formatMpcDisplayDate(date) {
  return new Intl.DateTimeFormat("nl-NL", {
    weekday: "short",
    day: "2-digit",
    month: "short",
  }).format(date);
}

function mpcChartTimestamp(timestamp) {
  const date = new Date(timestamp);
  const localTimestamp = new Date(date.getTime() - date.getTimezoneOffset() * 60 * 1000);
  return localTimestamp.toISOString().slice(0, 19);
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return Number(value).toFixed(digits);
}

function lineTrace(x, y, name, color, options = {}) {
  const trace = {
    x,
    y,
    name,
    type: "scatter",
    mode: options.mode || "lines",
    line: {
      color,
      width: options.width || 2,
      ...(options.dash ? { dash: options.dash } : {}),
      ...(options.shape ? { shape: options.shape } : {}),
    },
    yaxis: options.yaxis || "y",
    hovertemplate: options.hovertemplate || `%{x|%d-%m %H:%M}<br>%{y:.2f}<extra>${name}</extra>`,
  };
  if (options.marker) {
    trace.marker = options.marker;
  }
  if (options.fill) {
    trace.fill = options.fill;
  }
  if (options.fillcolor) {
    trace.fillcolor = options.fillcolor;
  }
  return trace;
}

const startTimeInput = document.getElementById("mpc-start-time");
const modelSelect = document.getElementById("mpc-model-select");
const horizonStepsInput = document.getElementById("mpc-horizon-steps");
const heatingKwInput = document.getElementById("mpc-heating-kw");
const planButton = document.getElementById("mpc-plan-button");
const previousDayButton = document.getElementById("mpc-previous-day");
const nextDayButton = document.getElementById("mpc-next-day");
const selectedDateNode = document.getElementById("mpc-selected-date");
const statusNode = document.getElementById("mpc-status");
const chartSummaryNode = document.getElementById("mpc-chart-summary");
const heatingExplanationNode = document.getElementById("mpc-heating-explanation");
const chartNode = document.getElementById("mpc-plan-chart");

const summaryNodes = {
  status: document.getElementById("mpc-summary-status"),
  termination: document.getElementById("mpc-summary-termination"),
  objective: document.getElementById("mpc-summary-objective"),
  objectiveComfortLow: document.getElementById("mpc-summary-objective-comfort-low"),
  objectiveComfortHigh: document.getElementById("mpc-summary-objective-comfort-high"),
  objectiveComfortTotal: document.getElementById("mpc-summary-objective-comfort-total"),
  objectiveTracking: document.getElementById("mpc-summary-objective-tracking"),
  objectiveTerminal: document.getElementById("mpc-summary-objective-terminal"),
  objectiveStart: document.getElementById("mpc-summary-objective-start"),
  objectiveRuntime: document.getElementById("mpc-summary-objective-runtime"),
  objectiveEnergy: document.getElementById("mpc-summary-objective-energy"),
  solveTime: document.getElementById("mpc-summary-solve-time"),
  starts: document.getElementById("mpc-summary-starts"),
  stops: document.getElementById("mpc-summary-stops"),
  violations: document.getElementById("mpc-summary-violations"),
  slack: document.getElementById("mpc-summary-slack"),
  runtime: document.getElementById("mpc-summary-runtime"),
  cost: document.getElementById("mpc-summary-cost"),
};

function setBusy(isBusy) {
  if (planButton) {
    planButton.disabled = isBusy;
  }
  if (previousDayButton) {
    previousDayButton.disabled = isBusy;
  }
  if (nextDayButton) {
    nextDayButton.disabled = isBusy;
  }
  if (modelSelect) {
    modelSelect.disabled = isBusy || modelSelect.options.length <= 1;
  }
}

function syncMpcDateLabel(anchorTime) {
  if (selectedDateNode) {
    selectedDateNode.textContent = formatMpcDisplayDate(anchorTime);
  }
}

function shiftMpcDay(days) {
  if (!startTimeInput?.value) {
    return;
  }
  const startTime = new Date(startTimeInput.value);
  startTime.setDate(startTime.getDate() + days);
  startTimeInput.value = mpcLocalInputValue(startTime);
  syncMpcDateLabel(startTime);
  loadPlan().catch(() => {});
}

function setSummaryValue(node, value) {
  if (node) {
    node.textContent = value;
  }
}

function modelLabel(model) {
  const activeSuffix = model.is_active ? " active" : "";
  return `${model.model_type} | ${model.model_id}${activeSuffix}`;
}

function populateMpcModelSelect(models) {
  if (!modelSelect) {
    return new Map();
  }

  const previousValue = modelSelect.value;
  modelSelect.innerHTML = "";

  if (!models.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "Geen modellen beschikbaar";
    modelSelect.append(option);
    modelSelect.disabled = true;
    return new Map();
  }

  const modelsById = new Map(models.map((model) => [model.model_id, model]));
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
    modelSelect.append(group);
  }

  modelSelect.disabled = false;
  modelSelect.value =
    modelsById.has(previousValue)
      ? previousValue
      : (models.find((model) => model.is_active)?.model_id || models[0].model_id);
  return modelsById;
}

async function refreshMpcModels() {
  const response = await fetch(mpcApiUrl("api/models/room"));
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Kon modellijst niet laden.");
  }
  return populateMpcModelSelect(payload.models || []);
}

function updateSummary(payload) {
  if (heatingExplanationNode) {
    heatingExplanationNode.textContent = payload.heating_explanation || "-";
  }
  setSummaryValue(summaryNodes.status, payload.status);
  setSummaryValue(summaryNodes.termination, payload.termination_condition);
  setSummaryValue(summaryNodes.objective, formatNumber(payload.objective_value, 3));
  setSummaryValue(
    summaryNodes.objectiveComfortLow,
    formatNumber(payload.objective_breakdown?.comfort_low, 3),
  );
  setSummaryValue(
    summaryNodes.objectiveComfortHigh,
    formatNumber(payload.objective_breakdown?.comfort_high, 3),
  );
  setSummaryValue(
    summaryNodes.objectiveComfortTotal,
    formatNumber(payload.objective_breakdown?.comfort_total, 3),
  );
  setSummaryValue(
    summaryNodes.objectiveTracking,
    formatNumber(payload.objective_breakdown?.temperature_tracking, 3),
  );
  setSummaryValue(
    summaryNodes.objectiveTerminal,
    formatNumber(payload.objective_breakdown?.terminal_cost, 3),
  );
  setSummaryValue(
    summaryNodes.objectiveStart,
    formatNumber(payload.objective_breakdown?.start_penalty, 3),
  );
  setSummaryValue(
    summaryNodes.objectiveRuntime,
    formatNumber(payload.objective_breakdown?.runtime, 3),
  );
  setSummaryValue(
    summaryNodes.objectiveEnergy,
    formatNumber(payload.objective_breakdown?.energy_cost, 3),
  );
  setSummaryValue(summaryNodes.solveTime, `${formatNumber(payload.solve_time_seconds, 3)} s`);
  setSummaryValue(summaryNodes.starts, String(payload.summary.start_count));
  setSummaryValue(summaryNodes.stops, String(payload.summary.stop_count));
  setSummaryValue(summaryNodes.violations, String(payload.summary.comfort_violation_count));
  setSummaryValue(summaryNodes.slack, String(payload.summary.slack_usage_count));
  setSummaryValue(summaryNodes.runtime, String(payload.summary.runtime_steps));
  setSummaryValue(summaryNodes.cost, `€ ${formatNumber(payload.summary.estimated_energy_cost_eur, 3)}`);
}

function renderPlanChart(payload) {
  const steps = payload.steps || [];
  const timestamps = steps.map((step) => mpcChartTimestamp(step.timestamp_utc));
  const roomTemps = steps.map((step) => step.predicted_room_temp_c);
  const comfortMin = steps.map((step) => step.temp_min_c);
  const comfortMax = steps.map((step) => step.temp_max_c);
  const hpOnBand = steps.map((step) => (step.hp_on ? 1 : 0));
  const prices = steps.map((step) => step.price_eur_kwh);
  const startMarkers = steps.filter((step) => step.start);
  const stopMarkers = steps.filter((step) => step.stop);

  const traces = [
    lineTrace(timestamps, roomTemps, "Predicted room temp", "#1565c0", {
      width: 3,
      hovertemplate: "%{x|%d-%m %H:%M}<br>%{y:.2f} °C<extra>Predicted room temp</extra>",
    }),
    lineTrace(
      timestamps,
      comfortMin,
      "Comfort min",
      "#c62828",
      {
        dash: "dot",
        hovertemplate: "%{x|%d-%m %H:%M}<br>%{y:.2f} °C<extra>Comfort min</extra>",
      },
    ),
    lineTrace(
      timestamps,
      comfortMax,
      "Comfort max",
      "#ef6c00",
      {
        dash: "dot",
        hovertemplate: "%{x|%d-%m %H:%M}<br>%{y:.2f} °C<extra>Comfort max</extra>",
      },
    ),
    lineTrace(
      timestamps,
      hpOnBand,
      "HP on",
      "#43a047",
      {
        shape: "hv",
        yaxis: "y2",
        fill: "tozeroy",
        fillcolor: "rgba(67, 160, 71, 0.18)",
        hovertemplate: "%{x|%d-%m %H:%M}<br>%{y}<extra>HP on</extra>",
      },
    ),
    lineTrace(
      timestamps,
      prices,
      "Price",
      "#6a1b9a",
      {
        yaxis: "y3",
        hovertemplate: "%{x|%d-%m %H:%M}<br>€ %{y:.3f}/kWh<extra>Price</extra>",
      },
    ),
    {
      x: startMarkers.map((step) => mpcChartTimestamp(step.timestamp_utc)),
      y: startMarkers.map((step) => step.predicted_room_temp_c),
      name: "Start",
      type: "scatter",
      mode: "markers",
      marker: { color: "#2e7d32", size: 10, symbol: "triangle-up" },
      hovertemplate: "%{x|%d-%m %H:%M}<extra>Start</extra>",
    },
    {
      x: stopMarkers.map((step) => mpcChartTimestamp(step.timestamp_utc)),
      y: stopMarkers.map((step) => step.predicted_room_temp_c),
      name: "Stop",
      type: "scatter",
      mode: "markers",
      marker: { color: "#ad1457", size: 10, symbol: "x" },
      hovertemplate: "%{x|%d-%m %H:%M}<extra>Stop</extra>",
    },
  ];

  const layout = {
    autosize: true,
    margin: { t: 10, r: 52, b: 40, l: 48 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    font: {
      family: 'Roboto, "Noto Sans", "Segoe UI", Arial, sans-serif',
      color: "#212121",
    },
    xaxis: {
      type: "date",
      tickformat: "%d %H:%M",
      showgrid: true,
      gridcolor: "#eceff1",
      zeroline: false,
      fixedrange: true,
    },
    yaxis: {
      title: { text: "°C" },
      showgrid: true,
      gridcolor: "#eceff1",
      zeroline: false,
      fixedrange: true,
    },
    yaxis2: {
      title: { text: "HP on" },
      overlaying: "y",
      side: "right",
      range: [-0.1, 1.2],
      tickmode: "array",
      tickvals: [0, 1],
      ticktext: ["off", "on"],
      showgrid: false,
      zeroline: false,
      fixedrange: true,
    },
    yaxis3: {
      title: { text: "€ / kWh" },
      overlaying: "y",
      side: "right",
      anchor: "free",
      position: 1,
      showgrid: false,
      zeroline: false,
      fixedrange: true,
    },
    legend: {
      orientation: "h",
      x: 0,
      y: -0.18,
      font: { size: 12 },
    },
  };

  Plotly.react(chartNode, traces, layout, { displayModeBar: false, responsive: true });
  if (chartSummaryNode) {
    chartSummaryNode.textContent = `${payload.summary.start_count} starts · ${payload.summary.runtime_steps} runtime steps`;
  }
}

async function loadPlan() {
  if (!startTimeInput?.value) {
    return;
  }

  setBusy(true);
  if (statusNode) {
    statusNode.textContent = "MPC-plan laden...";
    statusNode.className = "status";
  }

  try {
    const startTime = new Date(startTimeInput.value);
    syncMpcDateLabel(startTime);
    const params = new URLSearchParams({
      start_time: startTime.toISOString(),
      horizon_steps: String(Number(horizonStepsInput?.value || "144")),
      default_effective_heating_kw: String(Number(heatingKwInput?.value || "3.5")),
    });
    if (modelSelect?.value) {
      params.set("model_id", modelSelect.value);
    }
    const response = await fetch(mpcApiUrl(`api/mpc/space-heating/plan?${params.toString()}`));
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "MPC-plan kon niet worden geladen.");
    }
    updateSummary(payload);
    renderPlanChart(payload);
    if (statusNode) {
      statusNode.textContent = "MPC-plan geladen.";
      statusNode.className = "status success";
    }
  } catch (error) {
    if (statusNode) {
      statusNode.textContent =
        error instanceof Error ? error.message : "MPC-plan kon niet worden geladen.";
      statusNode.className = "status error";
    }
  } finally {
    setBusy(false);
  }
}

function initializeMpcPage() {
  if (startTimeInput) {
    const now = new Date();
    now.setMinutes(0, 0, 0);
    startTimeInput.value = mpcLocalInputValue(now);
    syncMpcDateLabel(now);
  }
  let modelsById = new Map();
  if (planButton) {
    planButton.addEventListener("click", () => {
      loadPlan().catch(() => {});
    });
  }
  if (modelSelect) {
    modelSelect.addEventListener("change", () => {
      loadPlan().catch(() => {});
    });
  }
  if (previousDayButton) {
    previousDayButton.addEventListener("click", () => {
      shiftMpcDay(-1);
    });
  }
  if (nextDayButton) {
    nextDayButton.addEventListener("click", () => {
      shiftMpcDay(1);
    });
  }
  refreshMpcModels()
    .then((nextModelsById) => {
      modelsById = nextModelsById;
      return loadPlan();
    })
    .catch((error) => {
      if (statusNode) {
        statusNode.textContent =
          error instanceof Error ? error.message : "MPC-plan kon niet worden geladen.";
        statusNode.className = "status error";
      }
    });
}

initializeMpcPage();
