const button = document.getElementById("import-button");
const status = document.getElementById("status");
const result = document.getElementById("result");

async function runImport() {
  if (!button) {
    return;
  }

  button.disabled = true;
  status.className = "status";
  status.textContent = "Import wordt gestart...";

  try {
    const response = await fetch("/api/history-import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Import mislukt.");
    }

    status.textContent = "Import draait...";
    await pollImportJob(payload.job_id);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Import mislukt.";
    status.className = "status error";
    status.textContent = message;
    result.textContent = "De import kon niet worden uitgevoerd.";
  } finally {
    button.disabled = button.dataset.importEnabled !== "true";
  }
}

async function pollImportJob(jobId) {
  while (true) {
    const response = await fetch(`/api/history-import/jobs/${jobId}`);
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Importstatus ophalen mislukt.");
    }

    result.textContent = JSON.stringify(payload, null, 2);

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

button?.addEventListener("click", runImport);
