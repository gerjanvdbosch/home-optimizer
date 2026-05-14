const backtestBaseUrl = new URL(".", window.location.href);

function backtestApiUrl(path) {
  return new URL(path, backtestBaseUrl).toString();
}

function localInputValue(date) {
  const localDate = new Date(date.getTime() - date.getTimezoneOffset() * 60 * 1000);
  return localDate.toISOString().slice(0, 16);
}

function chartTimestamp(timestamp) {
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

function metricDelta(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return `${value >= 0 ? "+" : ""}${Number(value).toFixed(digits)}`;
}

function chartLayout(yTitle, y2Title = null) {
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

const startTimeInput = document.getElementById("backtest-start-time");
const endTimeInput = document.getElementById("backtest-end-time");
const modelSelect = document.getElementById("backtest-model-select");
const horizonStepsInput = document.getElementById("backtest-horizon-steps");
const runButton = document.getElementById("backtest-run-button");
const statusNode = document.getElementById("backtest-status");
const summaryCaptionNode = document.getElementById("backtest-summary-caption");
const summaryBody = document.getElementById("backtest-summary-body");
const temperatureSummaryNode = document.getElementById("backtest-temp-summary");
const switchSummaryNode = document.getElementById("backtest-switch-summary");
const costSummaryNode = document.getElementById("backtest-cost-summary");
const comfortSummaryNode = document.getElementById("backtest-comfort-summary");
const temperatureChart = document.getElementById("backtest-temperature-chart");
const switchChart = document.getElementById("backtest-switch-chart");
const costChart = document.getElementById("backtest-cost-chart");
const comfortChart = document.getElementById("backtest-comfort-chart");

function setControlsDisabled(disabled) {
  if (runButton) {
    runButton.disabled = disabled;
  }
  if (startTimeInput) {
    startTimeInput.disabled = disabled;
  }
  if (endTimeInput) {
    endTimeInput.disabled = disabled;
  }
  if (modelSelect) {
    modelSelect.disabled = disabled;
  }
  if (horizonStepsInput) {
    horizonStepsInput.disabled = disabled;
  }
}

async function loadModels() {
  const response = await fetch(backtestApiUrl("api/models/room"));
  if (!response.ok) {
    throw new Error("Kon modellijst niet laden");
  }
  const payload = await response.json();
  modelSelect.innerHTML = "";
  for (const model of payload.models || []) {
    const option = document.createElement("option");
    option.value = model.model_id;
    option.textContent = `${model.model_type} | ${model.model_id}${model.is_active ? " active" : ""}`;
    if (model.is_active) {
      option.selected = true;
    }
    modelSelect.append(option);
  }
}

function renderSummary(payload) {
  const rows = [
    ["Energy cost (EUR)", payload.mpc_summary.estimated_energy_cost_eur, payload.historical_summary.estimated_energy_cost_eur, payload.delta.estimated_energy_cost_eur],
    ["Runtime (min)", payload.mpc_summary.runtime_minutes, payload.historical_summary.runtime_minutes, payload.delta.runtime_minutes],
    ["Starts/day", payload.mpc_summary.starts_per_day, payload.historical_summary.starts_per_day, payload.delta.starts_per_day],
    ["Comfort violation (min)", payload.mpc_summary.comfort_violation_minutes, payload.historical_summary.comfort_violation_minutes, payload.delta.comfort_violation_minutes],
    ["Degree-min below", payload.mpc_summary.degree_minutes_below_comfort, payload.historical_summary.degree_minutes_below_comfort, payload.delta.degree_minutes_below_comfort],
    ["Degree-min above", payload.mpc_summary.degree_minutes_above_comfort, payload.historical_summary.degree_minutes_above_comfort, payload.delta.degree_minutes_above_comfort],
    ["Infeasible", payload.mpc_summary.infeasible_count, payload.historical_summary.infeasible_count, payload.delta.infeasible_count],
    ["Avg solve time (s)", payload.mpc_summary.average_solver_runtime_seconds, payload.historical_summary.average_solver_runtime_seconds, payload.delta.average_solver_runtime_seconds],
  ];
  summaryBody.innerHTML = rows.map((row) => `
    <tr>
      <td>${row[0]}</td>
      <td>${formatNumber(row[1])}</td>
      <td>${formatNumber(row[2])}</td>
      <td>${metricDelta(row[3])}</td>
    </tr>
  `).join("");
  summaryCaptionNode.textContent = `${payload.model_type} | ${payload.model_id} | ${payload.interval_minutes} min | ${payload.step_count} stappen`;
}


function renderCharts(payload) {
  const timestamps = payload.steps.map((step) => chartTimestamp(step.timestamp_utc));
  const simulatedTemps = payload.steps.map((step) => step.simulated_next_room_temp_c);
  const historicalTemps = payload.steps.map((step) => step.historical_next_room_temp_c);
  const comfortMin = payload.steps.map((step) => step.temp_min_c);
  const comfortMax = payload.steps.map((step) => step.temp_max_c);
  const mpcHp = payload.steps.map((step) => Number(step.mpc_hp_on));
  const historicalHp = payload.steps.map((step) => Number(step.historical_hp_on));
  const prices = payload.steps.map((step) => step.price_eur_kwh);
  const mpcCost = [];
  const historicalCost = [];
  let mpcRunning = 0;
  let historicalRunning = 0;
  for (const step of payload.steps) {
    mpcRunning += step.estimated_mpc_energy_cost_eur;
    historicalRunning += step.estimated_historical_energy_cost_eur;
    mpcCost.push(mpcRunning);
    historicalCost.push(historicalRunning);
  }
  const comfortBelow = payload.steps.map((step) => Math.max(step.temp_min_c - step.simulated_next_room_temp_c, 0));
  const comfortAbove = payload.steps.map((step) => Math.max(step.simulated_next_room_temp_c - step.temp_max_c, 0));
  const slackLow = payload.steps.map((step) => step.slack_low_c);
  const slackHigh = payload.steps.map((step) => step.slack_high_c);
  const infeasibleTimes = payload.steps.filter((step) => !step.feasible).map((step) => chartTimestamp(step.timestamp_utc));
  const infeasibleY = infeasibleTimes.map(() => 1.05);

  Plotly.react(
    temperatureChart,
    [
      { x: timestamps, y: simulatedTemps, name: "MPC simulated", type: "scatter", mode: "lines", line: { color: "#03a9f4", width: 3 } },
      { x: timestamps, y: historicalTemps, name: "Historical measured", type: "scatter", mode: "lines", line: { color: "#6d4c41", width: 2 } },
      { x: timestamps, y: comfortMin, name: "Comfort min", type: "scatter", mode: "lines", line: { color: "#43a047", dash: "dot" } },
      { x: timestamps, y: comfortMax, name: "Comfort max", type: "scatter", mode: "lines", line: { color: "#fb8c00", dash: "dot" } },
    ],
    chartLayout("Temp (°C)"),
    { responsive: true, displayModeBar: false },
  );

  Plotly.react(
    switchChart,
    [
      { x: timestamps, y: mpcHp, name: "MPC on", type: "scatter", mode: "lines", line: { color: "#03a9f4", shape: "hv" } },
      { x: timestamps, y: historicalHp, name: "Historical on", type: "scatter", mode: "lines", line: { color: "#8e24aa", shape: "hv" } },
      { x: infeasibleTimes, y: infeasibleY, name: "Infeasible", type: "scatter", mode: "markers", marker: { color: "#db4437", size: 8, symbol: "x" } },
    ],
    chartLayout("On/off"),
    { responsive: true, displayModeBar: false },
  );

  Plotly.react(
    costChart,
    [
      { x: timestamps, y: mpcCost, name: "MPC cumulative cost", type: "scatter", mode: "lines", line: { color: "#03a9f4", width: 3 } },
      { x: timestamps, y: historicalCost, name: "Historical cumulative cost", type: "scatter", mode: "lines", line: { color: "#43a047", width: 2 } },
      { x: timestamps, y: prices, name: "Price", type: "scatter", mode: "lines", yaxis: "y2", line: { color: "#fb8c00", dash: "dot" } },
    ],
    chartLayout("Cost (EUR)", "Price (EUR/kWh)"),
    { responsive: true, displayModeBar: false },
  );

  Plotly.react(
    comfortChart,
    [
      { x: timestamps, y: comfortBelow, name: "Below comfort", type: "bar", marker: { color: "#1e88e5" } },
      { x: timestamps, y: comfortAbove, name: "Above comfort", type: "bar", marker: { color: "#e53935" } },
      { x: timestamps, y: slackLow, name: "Slack low", type: "scatter", mode: "lines", line: { color: "#3949ab", width: 2 } },
      { x: timestamps, y: slackHigh, name: "Slack high", type: "scatter", mode: "lines", line: { color: "#f4511e", width: 2 } },
    ],
    chartLayout("Temp gap / slack (°C)"),
    { responsive: true, displayModeBar: false },
  );

  temperatureSummaryNode.textContent = `${formatNumber(payload.delta.degree_minutes_below_comfort)} degree-min delta`;
  switchSummaryNode.textContent = `${formatNumber(payload.delta.starts_per_day)} starts/day delta`;
  costSummaryNode.textContent = `${metricDelta(payload.delta.estimated_energy_cost_eur, 3)} EUR`;
  comfortSummaryNode.textContent = `${payload.mpc_summary.infeasible_count} infeasible solves`;
}

async function loadBacktest() {
  setControlsDisabled(true);
  statusNode.className = "status";
  statusNode.textContent = "Backtest wordt geladen...";
  try {
    const params = new URLSearchParams({
      start_time: new Date(startTimeInput.value).toISOString(),
      end_time: new Date(endTimeInput.value).toISOString(),
      horizon_steps: String(Number(horizonStepsInput.value || 36)),
    });
    if (modelSelect.value) {
      params.set("model_id", modelSelect.value);
    }
    const response = await fetch(backtestApiUrl(`api/backtest/space-heating?${params.toString()}`));
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Backtest request mislukt");
    }
    renderSummary(payload);
    renderCharts(payload);
    statusNode.className = "status success";
    statusNode.textContent = "Backtest geladen";
  } catch (error) {
    statusNode.className = "status error";
    statusNode.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    setControlsDisabled(false);
  }
}

async function initializeBacktestPage() {
  const now = new Date();
  const endTime = new Date(now.getTime() - (now.getMinutes() % 10) * 60 * 1000);
  const startTime = new Date(endTime.getTime() - (24 * 60 * 60 * 1000));
  startTimeInput.value = localInputValue(startTime);
  endTimeInput.value = localInputValue(endTime);
  await loadModels();
  await loadBacktest();
}

if (runButton) {
  runButton.addEventListener("click", () => {
    loadBacktest().catch((error) => {
      statusNode.className = "status error";
      statusNode.textContent = error instanceof Error ? error.message : String(error);
    });
  });
}

initializeBacktestPage().catch((error) => {
  statusNode.className = "status error";
  statusNode.textContent = error instanceof Error ? error.message : String(error);
});
