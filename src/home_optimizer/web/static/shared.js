(() => {
  const baseUrl = new URL(".", window.location.href);

  function apiUrl(path) {
    return new URL(path, baseUrl).toString();
  }

  function toDatetimeLocalValue(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    const hours = String(date.getHours()).padStart(2, "0");
    const minutes = String(date.getMinutes()).padStart(2, "0");
    return `${year}-${month}-${day}T${hours}:${minutes}`;
  }

  function localInputToIso(value) {
    const date = new Date(value);
    const offsetMinutes = -date.getTimezoneOffset();
    const sign = offsetMinutes >= 0 ? "+" : "-";
    const absoluteOffset = Math.abs(offsetMinutes);
    const offsetHours = String(Math.floor(absoluteOffset / 60)).padStart(2, "0");
    const offsetRemainder = String(absoluteOffset % 60).padStart(2, "0");
    return `${toDatetimeLocalValue(date)}:00${sign}${offsetHours}:${offsetRemainder}`;
  }

  function formatDate(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    return `${year}-${month}-${day}`;
  }

  function formatDisplayDate(date) {
    return new Intl.DateTimeFormat("nl-NL", {
      weekday: "short",
      day: "2-digit",
      month: "short",
    }).format(date);
  }

  function buildConstantSeries(name, unit, startDate, endDate, intervalMinutes, value) {
    const points = [
      {
        timestamp: localInputToIso(toDatetimeLocalValue(startDate)),
        value,
      },
    ];
    const current = new Date(startDate);
    current.setMinutes(current.getMinutes() + intervalMinutes);

    while (current <= endDate) {
      points.push({
        timestamp: localInputToIso(toDatetimeLocalValue(current)),
        value,
      });
      current.setMinutes(current.getMinutes() + intervalMinutes);
    }

    return { name, unit, points };
  }

  function latestPoint(series) {
    if (!series || !series.points || series.points.length === 0) {
      return null;
    }
    return series.points[series.points.length - 1];
  }

  function formatSigned(value, digits = 2) {
    return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
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

  function chartTimestamp(timestamp) {
    const date = new Date(timestamp);
    const localTimestamp = new Date(date.getTime() - date.getTimezoneOffset() * 60 * 1000);
    return localTimestamp.toISOString().slice(0, 19);
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

    const layout = {
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
        ...(options.xRange ? { range: options.xRange } : {}),
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

    if (options.y2Title) {
      layout.yaxis2 = {
        title: { text: options.y2Title },
        overlaying: "y",
        side: "right",
        showgrid: false,
        zeroline: false,
        fixedrange: true,
      };
    }

    return layout;
  }

  function renderPlot(element, seriesList, options) {
    const traces = seriesList.map((series, index) => {
      const traceOptions = options.traceOptions?.[index] || {};
      const precision = Number.isFinite(traceOptions.precision) ? traceOptions.precision : 1;
      const yaxis = traceOptions.yaxis === "y2" ? "y2" : undefined;
      return {
        x: series.points.map((point) => chartTimestamp(point.timestamp)),
        y: series.points.map((point) => point.value),
        name: traceOptions.label || series.name,
        type: "scatter",
        mode: "lines",
        ...(yaxis ? { yaxis } : {}),
        line: {
          color: traceOptions.color || options.colors?.[index % (options.colors?.length || 1)],
          width: traceOptions.width || 2,
          ...(traceOptions.dash ? { dash: traceOptions.dash } : {}),
          ...(traceOptions.shape ? { shape: traceOptions.shape } : {}),
        },
        hovertemplate:
          `%{x|%H:%M}<br>%{y:.${precision}f} ${series.unit || ""}` +
          `<extra>${traceOptions.label || series.name}</extra>`,
      };
    });
    const hasPoints = seriesList.some((series) => series.points.length > 0);

    Plotly.react(element, traces, plotLayout(options, hasPoints), {
      displayModeBar: false,
      responsive: true,
    });
  }

  window.HomeOptimizer = {
    apiUrl,
    buildConstantSeries,
    chartTimestamp,
    formatDate,
    formatDisplayDate,
    formatSigned,
    latestPoint,
    localInputToIso,
    plotLayout,
    renderPlot,
    summarizeSeries,
    toDatetimeLocalValue,
  };
})();
