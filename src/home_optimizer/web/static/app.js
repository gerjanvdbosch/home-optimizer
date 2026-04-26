const button = document.getElementById("import-button");
const status = document.getElementById("status");
const result = document.getElementById("result");
const selectedDateLabel = document.getElementById("selected-date");
const previousDayButton = document.getElementById("previous-day");
const nextDayButton = document.getElementById("next-day");
const roomSummary = document.getElementById("room-summary");
const dhwSummary = document.getElementById("dhw-summary");
const heatpumpSummary = document.getElementById("heatpump-summary");
const roomChart = document.getElementById("room-chart");
const dhwChart = document.getElementById("dhw-chart");
const heatpumpChart = document.getElementById("heatpump-chart");
const baseUrl = new URL(".", window.location.href);
const heatpumpModeColors = {
  ufh: "#43a047",
  dhw: "#ff9800",
  legionella: "#d81b60",
  cool: "#03a9f4",
  off: "#9e9e9e",
};
const heatpumpStatusStyles = {
  defrost_active: {
    label: "Defrost",
    color: "#1976d2",
    fill: "rgba(25, 118, 210, 0.14)",
  },
  booster_heater_active: {
    label: "Booster heater",
    color: "#c62828",
    fill: "rgba(198, 40, 40, 0.13)",
  },
};

let selectedDate = new Date();

function apiUrl(path) {
  return new URL(path, baseUrl).toString();
}

function formatDate(date) {
  return date.toISOString().slice(0, 10);
}

function formatDisplayDate(date) {
  return new Intl.DateTimeFormat("nl-NL", {
    weekday: "short",
    day: "2-digit",
    month: "short",
  }).format(date);
}

function shiftDate(days) {
  selectedDate = new Date(selectedDate.getTime() + days * 24 * 60 * 60 * 1000);
  loadCharts();
}

async function runImport() {
  if (!button) {
    return;
  }

  button.disabled = true;
  status.className = "status";
  status.textContent = "Import wordt gestart...";

  try {
    const response = await fetch(apiUrl("api/history-import"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Import mislukt.");
    }

    status.textContent = "Import draait...";
    await pollImportJob(payload.job_id);
    await loadCharts();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Import mislukt.";
    status.className = "status error";
    status.textContent = message;
    result.hidden = false;
    result.textContent = "De import kon niet worden uitgevoerd.";
  } finally {
    button.disabled = false;
  }
}

async function pollImportJob(jobId) {
  while (true) {
    const response = await fetch(apiUrl(`api/history-import/jobs/${jobId}`));
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Importstatus ophalen mislukt.");
    }

    result.textContent = JSON.stringify(payload, null, 2);
    result.hidden = false;

    if (payload.status === "succeeded") {
      status.className = "status success";
      status.textContent =
        `Import voltooid: ${payload.total_rows} rijen over ` +
        `${payload.sensor_count} sensoren.`;
      return;
    }

    if (payload.status === "failed") {
      throw new Error(payload.error || "Import mislukt.");
    }

    status.className = "status";
    status.textContent = `Import ${payload.status}...`;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

async function loadCharts() {
  if (!roomChart || !dhwChart || !heatpumpChart || !selectedDateLabel) {
    return;
  }

  selectedDateLabel.textContent = formatDisplayDate(selectedDate);
  const response = await fetch(apiUrl(`api/dashboard/charts?date=${formatDate(selectedDate)}`));
  const payload = await response.json();

  if (!response.ok) {
    throw new Error(payload.detail || "Grafiekdata ophalen mislukt.");
  }

  renderPlot(roomChart, [payload.room_temperature], {
    colors: ["#03a9f4"],
    emptyText: "Geen kamertemperatuur voor deze dag",
    yTitle: payload.room_temperature.unit || "",
  });
  renderPlot(dhwChart, payload.dhw_temperatures, {
    colors: ["#ff9800", "#7e57c2"],
    emptyText: "Geen boilerdata voor deze dag",
    yTitle: payload.dhw_temperatures[0]?.unit || "",
  });
  renderHeatpumpPowerPlot(
    heatpumpChart,
    payload.heatpump_power,
    payload.heatpump_mode,
    payload.heatpump_statuses,
  );

  roomSummary.textContent = summarizeSeries(payload.room_temperature);
  dhwSummary.textContent = summarizeSeries(payload.dhw_temperatures[0]);
  heatpumpSummary.textContent = summarizeHeatpump(
    payload.heatpump_power,
    payload.heatpump_mode,
    payload.heatpump_statuses,
  );
}

function summarizeSeries(series) {
  const values = series.points.map((point) => point.value);
  if (values.length === 0) {
    return "-";
  }
  const latest = values[values.length - 1];
  const unit = series.unit || "";
  return `${latest.toFixed(1)} ${unit}`.trim();
}

function renderPlot(element, seriesList, options) {
  const traces = seriesList.map((series, index) => ({
    x: series.points.map((point) => point.timestamp),
    y: series.points.map((point) => point.value),
    name: series.name,
    type: "scatter",
    mode: "lines",
    line: {
      color: options.colors[index % options.colors.length],
      width: 2,
    },
    hovertemplate: `%{x|%H:%M}<br>%{y:.1f} ${series.unit || ""}<extra>${series.name}</extra>`,
  }));
  const hasPoints = seriesList.some((series) => series.points.length > 0);

  Plotly.react(element, traces, plotLayout(options, hasPoints), {
    displayModeBar: false,
    responsive: true,
  });
}

function renderHeatpumpPowerPlot(element, powerSeries, modeSeries, statusSeriesList) {
  const points = powerSeries.points.map((point) => ({
    ...point,
    mode: modeAtTimestamp(modeSeries.points, point.timestamp),
  }));
  const modes = orderedModes(points.map((point) => point.mode));
  const statusIntervalsByName = heatpumpStatusIntervals(statusSeriesList);
  const traces = modes.map((mode) => ({
    x: points.map((point) => point.timestamp),
    y: points.map((point) => (point.mode === mode ? point.value : null)),
    customdata: points.map((point) => point.mode),
    name: displayMode(mode),
    type: "scatter",
    mode: "lines",
    connectgaps: false,
    line: {
      color: heatpumpModeColors[mode] || "#5c6bc0",
      width: 2,
    },
    hovertemplate:
      `%{x|%H:%M}<br>%{y:.1f} ${powerSeries.unit || ""}` +
      "<br>Mode: %{customdata}<extra></extra>",
  }));
  traces.push(...heatpumpStatusLegendTraces(statusIntervalsByName));
  const hasPoints = powerSeries.points.length > 0;

  Plotly.react(
    element,
    traces,
    plotLayout(
      {
        emptyText: "Geen warmtepompvermogen voor deze dag",
        yTitle: powerSeries.unit || "",
        shapes: heatpumpStatusShapes(statusIntervalsByName),
      },
      hasPoints,
    ),
    {
      displayModeBar: false,
      responsive: true,
    },
  );
}

function heatpumpStatusIntervals(statusSeriesList) {
  return statusSeriesList.map((series) => ({
    name: series.name,
    intervals: activeIntervals(series.points),
  }));
}

function activeIntervals(points) {
  const intervals = [];
  let start = null;
  let previousTimestamp = null;

  for (const point of points) {
    if (isActiveStatusValue(point.value)) {
      start = start || point.timestamp;
      previousTimestamp = point.timestamp;
      continue;
    }

    if (start) {
      intervals.push({ x0: start, x1: point.timestamp });
      start = null;
      previousTimestamp = null;
    }
  }

  if (start && previousTimestamp) {
    intervals.push({ x0: start, x1: addMinutes(previousTimestamp, 1) });
  }

  return intervals;
}

function isActiveStatusValue(value) {
  if (typeof value === "number") {
    return value >= 0.5;
  }
  return ["1", "true", "on", "active"].includes(String(value).trim().toLowerCase());
}

function addMinutes(timestamp, minutes) {
  return new Date(Date.parse(timestamp) + minutes * 60 * 1000).toISOString();
}

function heatpumpStatusShapes(statusIntervalsByName) {
  return statusIntervalsByName.flatMap((series) => {
    const style = heatpumpStatusStyles[series.name];
    if (!style) {
      return [];
    }

    return series.intervals.map((interval) => ({
      type: "rect",
      xref: "x",
      yref: "paper",
      x0: interval.x0,
      x1: interval.x1,
      y0: 0,
      y1: 1,
      fillcolor: style.fill,
      line: { width: 0 },
      layer: "below",
    }));
  });
}

function heatpumpStatusLegendTraces(statusIntervalsByName) {
  return statusIntervalsByName
    .filter((series) => series.intervals.length > 0)
    .map((series) => {
      const style = heatpumpStatusStyles[series.name] || {
        label: series.name,
        color: "#5c6bc0",
      };
      return {
        x: [null],
        y: [null],
        name: style.label,
        type: "scatter",
        mode: "lines",
        line: { color: style.color, width: 8 },
        hoverinfo: "skip",
      };
    });
}

function modeAtTimestamp(modePoints, timestamp) {
  let currentMode = "onbekend";
  const timestampMs = Date.parse(timestamp);

  for (const point of modePoints) {
    if (Date.parse(point.timestamp) > timestampMs) {
      break;
    }
    currentMode = point.value;
  }

  return currentMode;
}

function displayMode(mode) {
  if (mode === "onbekend") {
    return "Onbekend";
  }
  return mode.toUpperCase();
}

function orderedModes(modes) {
  const preferredOrder = ["ufh", "dhw", "legionella", "cool", "off", "onbekend"];
  const uniqueModes = Array.from(new Set(modes));
  return uniqueModes.sort((left, right) => {
    const leftIndex = preferredOrder.indexOf(left);
    const rightIndex = preferredOrder.indexOf(right);
    return modeSortIndex(leftIndex) - modeSortIndex(rightIndex) || left.localeCompare(right);
  });
}

function modeSortIndex(index) {
  return index === -1 ? Number.MAX_SAFE_INTEGER : index;
}

function summarizeHeatpump(powerSeries, modeSeries, statusSeriesList) {
  if (powerSeries.points.length === 0) {
    return "-";
  }
  const latest = powerSeries.points[powerSeries.points.length - 1];
  const mode = modeAtTimestamp(modeSeries.points, latest.timestamp);
  const unit = powerSeries.unit || "";
  const statusLabels = activeStatusLabels(statusSeriesList, latest.timestamp);
  return [`${latest.value.toFixed(1)} ${unit}`, displayMode(mode), ...statusLabels]
    .join(" · ")
    .trim();
}

function activeStatusLabels(statusSeriesList, timestamp) {
  return statusSeriesList
    .filter((series) => isActiveStatusValue(statusAtTimestamp(series.points, timestamp)))
    .map((series) => heatpumpStatusStyles[series.name]?.label || series.name);
}

function statusAtTimestamp(points, timestamp) {
  let currentValue = 0;
  const timestampMs = Date.parse(timestamp);

  for (const point of points) {
    if (Date.parse(point.timestamp) > timestampMs) {
      break;
    }
    currentValue = point.value;
  }

  return currentValue;
}

function plotLayout(options, hasPoints) {
  const annotations = hasPoints
    ? []
    : [
        {
          text: options.emptyText,
          xref: "paper",
          yref: "paper",
          x: 0.5,
          y: 0.5,
          showarrow: false,
          font: { color: "#727272", size: 14 },
        },
      ];

  return {
    autosize: true,
    margin: { t: 10, r: 12, b: 36, l: 46 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    annotations,
    shapes: options.shapes || [],
    xaxis: {
      type: "date",
      tickformat: "%H:%M",
      showgrid: true,
      gridcolor: "#eceff1",
      zeroline: false,
      fixedrange: true,
    },
    yaxis: {
      title: { text: options.yTitle },
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
}

button?.addEventListener("click", runImport);
previousDayButton?.addEventListener("click", () => shiftDate(-1));
nextDayButton?.addEventListener("click", () => shiftDate(1));
window.addEventListener("resize", () => {
  if (roomChart) {
    Plotly.Plots.resize(roomChart);
  }
  if (dhwChart) {
    Plotly.Plots.resize(dhwChart);
  }
  if (heatpumpChart) {
    Plotly.Plots.resize(heatpumpChart);
  }
});
loadCharts().catch((error) => {
  if (roomSummary) {
    roomSummary.textContent = "Fout";
  }
  if (dhwSummary) {
    dhwSummary.textContent = "Fout";
  }
  if (heatpumpSummary) {
    heatpumpSummary.textContent = "Fout";
  }
  console.error(error);
});
