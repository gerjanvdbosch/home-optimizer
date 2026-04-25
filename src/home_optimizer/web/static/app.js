const button = document.getElementById("import-button");
const status = document.getElementById("status");
const result = document.getElementById("result");

async function runImport() {
  if (!button) {
    return;
  }

  button.disabled = true;
  status.className = "status";
  status.textContent = "Import wordt uitgevoerd...";

  try {
    const response = await fetch("/api/history-import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Import mislukt.");
    }

    status.className = "status success";
    status.textContent =
      `Import voltooid: ${payload.total_rows} rijen over ` +
      `${payload.sensor_count} sensoren.`;
    result.textContent = JSON.stringify(payload, null, 2);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Import mislukt.";
    status.className = "status error";
    status.textContent = message;
    result.textContent = "De import kon niet worden uitgevoerd.";
  } finally {
    button.disabled = button.dataset.importEnabled !== "true";
  }
}

button?.addEventListener("click", runImport);
