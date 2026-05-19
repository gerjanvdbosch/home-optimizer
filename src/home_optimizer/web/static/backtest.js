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
const exogenousModeSelect = document.getElementById("backtest-exogenous-mode");
const runButton = document.getElementById("backtest-run-button");
const statusNode = document.getElementById("backtest-status");
const summaryCaptionNode = document.getElementById("backtest-summary-caption");
const summaryBody = document.getElementById("backtest-summary-body");
const objectiveBody = document.getElementById("backtest-objective-body");
const solverObjectiveBody = document.getElementById("backtest-solver-objective-body");
const startStopBody = document.getElementById("backtest-start-stop-body");
const temperatureSummaryNode = document.getElementById("backtest-temp-summary");
const switchSummaryNode = document.getElementById("backtest-switch-summary");
const costSummaryNode = document.getElementById("backtest-cost-summary");
const pvSummaryNode = document.getElementById("backtest-pv-summary");
const pvSurplusSummaryNode = document.getElementById("backtest-pv-surplus-summary");
const pvCaptureSummaryNode = document.getElementById("backtest-pv-capture-summary");
const comfortSummaryNode = document.getElementById("backtest-comfort-summary");
const temperatureChart = document.getElementById("backtest-temperature-chart");
const switchChart = document.getElementById("backtest-switch-chart");
const costChart = document.getElementById("backtest-cost-chart");
const pvChart = document.getElementById("backtest-pv-chart");
const pvSurplusChart = document.getElementById("backtest-pv-surplus-chart");
const pvCaptureChart = document.getElementById("backtest-pv-capture-chart");
const comfortChart = document.getElementById("backtest-comfort-chart");
const previousDayButton = document.getElementById("backtest-previous-day");
const nextDayButton = document.getElementById("backtest-next-day");
const backtestSelectedDate = document.getElementById("backtest-selected-date");

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
  if (exogenousModeSelect) {
    exogenousModeSelect.disabled = disabled;
  }
  if (previousDayButton) {
    previousDayButton.disabled = disabled;
  }
  if (nextDayButton) {
    nextDayButton.disabled = disabled;
  }
}

async function loadModels() {
  const response = await fetch(backtestApiUrl("api/models/room"));
  if (!response.ok) {
    throw new Error("Kon modellijst niet laden");
  }
  const payload = await response.json();
  const models = payload.models || [];
  const previousValue = modelSelect.value;
  modelSelect.innerHTML = "";

  if (!models.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "Geen modellen beschikbaar";
    modelSelect.append(option);
    modelSelect.disabled = true;
    return;
  }

  modelSelect.disabled = false;
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
      option.textContent = `${model.model_type} | ${model.model_id}${model.is_active ? " active" : ""}`;
      group.append(option);
    }
    modelSelect.append(group);
  }

  const selectedValue =
    models.some((m) => m.model_id === previousValue)
      ? previousValue
      : (models.find((m) => m.is_active)?.model_id || models[0].model_id);
  modelSelect.value = selectedValue;
}

function renderSummary(payload) {
  const rows = [
    ["Energy cost (EUR)", payload.mpc_summary.estimated_energy_cost_eur, payload.historical_summary.estimated_energy_cost_eur, payload.delta.estimated_energy_cost_eur],
    ["Runtime (min)", payload.mpc_summary.runtime_minutes, payload.historical_summary.runtime_minutes, payload.delta.runtime_minutes],
    ["Starts/day", payload.mpc_summary.starts_per_day, payload.historical_summary.starts_per_day, payload.delta.starts_per_day],
    ["Comfort violation (min)", payload.mpc_summary.comfort_violation_minutes, payload.historical_summary.comfort_violation_minutes, payload.delta.comfort_violation_minutes],
    ["Degree-min below", payload.mpc_summary.degree_minutes_below_comfort, payload.historical_summary.degree_minutes_below_comfort, payload.delta.degree_minutes_below_comfort],
    ["Degree-min above", payload.mpc_summary.degree_minutes_above_comfort, payload.historical_summary.degree_minutes_above_comfort, payload.delta.degree_minutes_above_comfort],
    ["Active high deg-min", payload.mpc_summary.active_comfort_high_degree_minutes, payload.historical_summary.active_comfort_high_degree_minutes, payload.delta.active_comfort_high_degree_minutes],
    ["Passive high deg-min", payload.mpc_summary.passive_comfort_high_degree_minutes, payload.historical_summary.passive_comfort_high_degree_minutes, payload.delta.passive_comfort_high_degree_minutes],
    ["PV surplus forecast (kWh)", payload.pv_diagnostics.forecast_pv_surplus_kwh, payload.pv_diagnostics.forecast_pv_surplus_kwh, 0],
    ["PV surplus realized (kWh)", payload.pv_diagnostics.realized_pv_surplus_kwh, payload.pv_diagnostics.realized_pv_surplus_kwh, 0],
    ["Preheat budget (kWh)", payload.pv_diagnostics.preheat_budget_electric_kwh, 0, payload.pv_diagnostics.preheat_budget_electric_kwh],
    ["Used preheat budget (kWh)", payload.pv_diagnostics.used_preheat_budget_kwh, 0, payload.pv_diagnostics.used_preheat_budget_kwh],
    ["Missed surplus w/ headroom (kWh)", payload.pv_diagnostics.missed_surplus_with_headroom_kwh, 0, payload.pv_diagnostics.missed_surplus_with_headroom_kwh],
    ["MPC HP energy (kWh)", payload.pv_diagnostics.mpc_hp_energy_kwh, 0, payload.pv_diagnostics.mpc_hp_energy_kwh],
    ["Captured realized PV (kWh)", payload.pv_diagnostics.mpc_realized_pv_surplus_capture_kwh, 0, payload.pv_diagnostics.mpc_realized_pv_surplus_capture_kwh],
    ["Capture ratio realized", payload.pv_diagnostics.mpc_realized_pv_surplus_capture_ratio, 0, payload.pv_diagnostics.mpc_realized_pv_surplus_capture_ratio],
    ["Preheat block count", payload.pv_diagnostics.preheat_block_count, 0, payload.pv_diagnostics.preheat_block_count],
    ["Starts per preheat block", payload.pv_diagnostics.starts_per_preheat_block, 0, payload.pv_diagnostics.starts_per_preheat_block],
    ["Average run duration (min)", payload.pv_diagnostics.average_run_duration_minutes, 0, payload.pv_diagnostics.average_run_duration_minutes],
    ["Short run count", payload.pv_diagnostics.short_run_count, 0, payload.pv_diagnostics.short_run_count],
    ["Authority total starts", payload.invariant_report.total_starts, 0, payload.invariant_report.total_starts],
    ["Authority total stops", payload.invariant_report.total_stops, 0, payload.invariant_report.total_stops],
    ["Starts outside intents", payload.invariant_report.starts_outside_intents, 0, payload.invariant_report.starts_outside_intents],
    ["Emergency starts", payload.invariant_report.emergency_starts, 0, payload.invariant_report.emergency_starts],
    ["External starts", payload.invariant_report.external_starts, 0, payload.invariant_report.external_starts],
    ["Authority violations", payload.invariant_report.start_stop_violation_count, 0, payload.invariant_report.start_stop_violation_count],
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
  const objectiveRows = objectiveBreakdownRows(payload.mpc_objective_breakdown);
  const solverObjectiveRows = objectiveBreakdownRows(payload.solver_objective_breakdown);
  if (objectiveBody) {
    objectiveBody.innerHTML = objectiveRows.map((row) => `
      <tr>
        <td>${row[0]}</td>
        <td>${formatNumber(row[1], 3)}</td>
      </tr>
    `).join("");
  }
  if (solverObjectiveBody) {
    solverObjectiveBody.innerHTML = solverObjectiveRows.map((row) => `
      <tr>
        <td>${row[0]}</td>
        <td>${formatNumber(row[1], 3)}</td>
      </tr>
    `).join("");
  }
  if (startStopBody) {
    const ledgerRows = payload.start_stop_ledger || [];
    startStopBody.innerHTML = ledgerRows.length
      ? ledgerRows.map((entry) => `
        <tr>
          <td>${chartTimestamp(entry.timestamp)}</td>
          <td>${entry.transition}</td>
          <td>${entry.start_reason || entry.stop_reason || "-"}</td>
          <td>${entry.intent_type ? `${entry.intent_type}${entry.intent_id ? ` | ${entry.intent_id}` : ""}` : (entry.intent_id || "-")}</td>
          <td>${entry.active_run_id || "-"}</td>
          <td>${entry.sequencer_mode_before} -> ${entry.sequencer_mode_after}</td>
        </tr>
      `).join("")
      : `
        <tr>
          <td colspan="6">Geen start/stop-overgangen geregistreerd in deze backtest.</td>
        </tr>
      `;
  }
  summaryCaptionNode.textContent = `${payload.model_type} | ${payload.model_id} | ${payload.interval_minutes} min | ${payload.step_count} stappen | ${payload.exogenous_mode}`;
}

function objectiveBreakdownRows(breakdown) {
  return [
    ["Comfort low", breakdown.comfort_low],
    ["Active comfort-high cost", breakdown.active_comfort_high_cost],
    ["Passive comfort-high cost", breakdown.passive_comfort_high_cost],
    ["Comfort high", breakdown.comfort_high],
    ["Comfort total", breakdown.comfort_total],
    ["Tracking under target", breakdown.tracking_under_target],
    ["Tracking over target", breakdown.tracking_over_target],
    ["Target tracking total", breakdown.temperature_tracking],
    ["Energy cost", breakdown.energy_cost],
    ["PV self-consumption reward", breakdown.pv_self_consumption_reward],
    ["Captured PV (kWh)", breakdown.captured_pv_kwh],
    ["Preheat budget shortfall", breakdown.preheat_budget_shortfall],
    ["Unnecessary heating cost", breakdown.unnecessary_heating],
    ["Terminal cost", breakdown.terminal_cost],
    ["Start penalty", breakdown.start_penalty],
    ["Runtime", breakdown.runtime],
    ["Total", breakdown.total],
  ];
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
  const pvForecast = payload.steps.map((step) => step.pv_forecast_kw);
  const pvRealized = payload.steps.map((step) => step.pv_realized_kw);
  const baseLoadForecast = payload.steps.map((step) => step.base_load_forecast_kw);
  const baseLoadRealized = payload.steps.map((step) => step.base_load_realized_kw);
  const pvSurplusForecast = payload.steps.map((step) => step.pv_surplus_forecast_kw);
  const pvSurplusRealized = payload.steps.map((step) => step.pv_surplus_realized_kw);
  const mpcHpPower = payload.steps.map((step) => step.hp_electric_power_kw * Number(step.mpc_hp_on));
  const mpcCost = [];
  const historicalCost = [];
  const cumulativeCapturedRealized = [];
  const cumulativeCapturedForecast = [];
  let mpcRunning = 0;
  let historicalRunning = 0;
  let capturedRealizedRunning = 0;
  let capturedForecastRunning = 0;
  const dtHours = (payload.interval_minutes || 0) / 60;
  const preheatChargePower = payload.steps.map((step) => dtHours > 0 ? step.preheat_charge_kwh / dtHours : 0);
  for (const step of payload.steps) {
    mpcRunning += step.estimated_mpc_energy_cost_eur;
    historicalRunning += step.estimated_historical_energy_cost_eur;
    mpcCost.push(mpcRunning);
    historicalCost.push(historicalRunning);
    capturedRealizedRunning += Math.min(
      step.hp_electric_power_kw * Number(step.mpc_hp_on),
      step.pv_surplus_realized_kw,
    ) * dtHours;
    capturedForecastRunning += Math.min(
      step.hp_electric_power_kw * Number(step.mpc_hp_on),
      step.pv_surplus_forecast_kw,
    ) * dtHours;
    cumulativeCapturedRealized.push(capturedRealizedRunning);
    cumulativeCapturedForecast.push(capturedForecastRunning);
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
    pvChart,
    [
      { x: timestamps, y: pvForecast, name: "PV forecast", type: "scatter", mode: "lines", line: { color: "#f9a825", width: 3 } },
      { x: timestamps, y: pvRealized, name: "PV realized", type: "scatter", mode: "lines", line: { color: "#6d4c41", width: 2 } },
      { x: timestamps, y: baseLoadForecast, name: "Baseload forecast", type: "scatter", mode: "lines", line: { color: "#1e88e5", dash: "dot" } },
      { x: timestamps, y: baseLoadRealized, name: "Baseload realized", type: "scatter", mode: "lines", line: { color: "#43a047", dash: "dot" } },
    ],
    chartLayout("Power (kW)"),
    { responsive: true, displayModeBar: false },
  );

  Plotly.react(
    pvSurplusChart,
    [
      { x: timestamps, y: pvSurplusForecast, name: "PV surplus forecast", type: "scatter", mode: "lines", line: { color: "#fb8c00", width: 3 } },
      { x: timestamps, y: pvSurplusRealized, name: "PV surplus realized", type: "scatter", mode: "lines", line: { color: "#43a047", width: 2 } },
      { x: timestamps, y: mpcHpPower, name: "MPC HP power", type: "scatter", mode: "lines", yaxis: "y2", line: { color: "#03a9f4", shape: "hv" } },
      { x: timestamps, y: preheatChargePower, name: "Preheat charge", type: "scatter", mode: "lines", yaxis: "y2", line: { color: "#8e24aa", width: 2, dash: "dot" } },
    ],
    chartLayout("Surplus (kW)", "HP power (kW)"),
    { responsive: true, displayModeBar: false },
  );

  Plotly.react(
    pvCaptureChart,
    [
      { x: timestamps, y: cumulativeCapturedForecast, name: "Captured forecast surplus", type: "scatter", mode: "lines", line: { color: "#fb8c00", width: 3 } },
      { x: timestamps, y: cumulativeCapturedRealized, name: "Captured realized surplus", type: "scatter", mode: "lines", line: { color: "#43a047", width: 2 } },
    ],
    chartLayout("Captured PV (kWh)"),
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
  pvSummaryNode.textContent = `${formatNumber(payload.pv_diagnostics.forecast_pv_surplus_kwh, 2)} / ${formatNumber(payload.pv_diagnostics.realized_pv_surplus_kwh, 2)} kWh surplus`;
  pvSurplusSummaryNode.textContent = `${formatNumber(payload.pv_diagnostics.mpc_hp_energy_during_forecast_pv_surplus_kwh, 2)} / ${formatNumber(payload.pv_diagnostics.mpc_hp_energy_during_realized_pv_surplus_kwh, 2)} kWh benut`;
  pvCaptureSummaryNode.textContent = `${formatNumber(payload.pv_diagnostics.mpc_realized_pv_surplus_capture_ratio * 100, 1)}% realized capture`;
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
      exogenous_mode: exogenousModeSelect?.value || "perfect_foresight",
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
    statusNode.className = payload.forecast_coverage_ratio < 1 ? "status error" : "status success";
    statusNode.textContent = payload.forecast_coverage_ratio < 1
      ? `Backtest geladen met onvolledige forecastdekking (${formatNumber(payload.forecast_coverage_ratio * 100, 1)}%)`
      : "Backtest geladen";
  } catch (error) {
    statusNode.className = "status error";
    statusNode.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    setControlsDisabled(false);
  }
}

function midnightOf(date) {
  const d = new Date(date);
  d.setHours(0, 0, 0, 0);
  return d;
}

function formatBacktestDisplayDate(startDate) {
  return startDate.toLocaleDateString("nl-NL", { weekday: "short", day: "numeric", month: "short" });
}

function syncBacktestDateLabel(startDate) {
  if (backtestSelectedDate) {
    backtestSelectedDate.textContent = formatBacktestDisplayDate(startDate);
  }
}

function setBacktestDateRange(startDate) {
  const start = midnightOf(startDate);
  const end = new Date(start.getTime() + 24 * 60 * 60 * 1000);
  startTimeInput.value = localInputValue(start);
  endTimeInput.value = localInputValue(end);
  syncBacktestDateLabel(start);
}

function shiftBacktestDay(days) {
  if (!startTimeInput?.value) {
    return;
  }
  const current = midnightOf(new Date(startTimeInput.value));
  current.setDate(current.getDate() + days);
  setBacktestDateRange(current);
  loadBacktest().catch((error) => {
    statusNode.className = "status error";
    statusNode.textContent = error instanceof Error ? error.message : String(error);
  });
}

async function initializeBacktestPage() {
  const yesterday = midnightOf(new Date());
  yesterday.setDate(yesterday.getDate() - 1);
  setBacktestDateRange(yesterday);
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

if (previousDayButton) {
  previousDayButton.addEventListener("click", () => shiftBacktestDay(-1));
}

if (nextDayButton) {
  nextDayButton.addEventListener("click", () => shiftBacktestDay(1));
}

initializeBacktestPage().catch((error) => {
  statusNode.className = "status error";
  statusNode.textContent = error instanceof Error ? error.message : String(error);
});
