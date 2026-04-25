from __future__ import annotations

from home_optimizer.web.schemas import DashboardViewModel


def render_dashboard(view_model: DashboardViewModel) -> str:
    button_disabled = "disabled" if not view_model.import_enabled else ""
    button_label = (
        "Import uitgeschakeld in configuratie"
        if not view_model.import_enabled
        else "Importeer geschiedenis"
    )

    return f"""<!DOCTYPE html>
<html lang="nl">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{view_model.title}</title>
    <style>
      :root {{
        color-scheme: light;
        --bg: #f4f7f2;
        --panel: rgba(255, 255, 255, 0.92);
        --panel-border: rgba(20, 62, 43, 0.10);
        --text: #163224;
        --muted: #587164;
        --accent: #1f7a55;
        --accent-strong: #15573c;
        --accent-soft: #e6f3ec;
        --danger: #9f2d2d;
        --shadow: 0 20px 60px rgba(18, 40, 29, 0.12);
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      }}

      * {{
        box-sizing: border-box;
      }}

      body {{
        margin: 0;
        min-height: 100vh;
        color: var(--text);
        background:
          radial-gradient(circle at top left, rgba(112, 191, 145, 0.28), transparent 28%),
          radial-gradient(circle at bottom right, rgba(65, 133, 96, 0.18), transparent 32%),
          linear-gradient(180deg, #eef5ee 0%, var(--bg) 100%);
      }}

      main {{
        width: min(900px, calc(100% - 32px));
        margin: 48px auto;
      }}

      .hero {{
        display: grid;
        gap: 20px;
        padding: 32px;
        border: 1px solid var(--panel-border);
        border-radius: 24px;
        background: var(--panel);
        backdrop-filter: blur(14px);
        box-shadow: var(--shadow);
      }}

      h1 {{
        margin: 0;
        font-size: clamp(2rem, 4vw, 3.4rem);
        line-height: 1.05;
        letter-spacing: -0.04em;
      }}

      p {{
        margin: 0;
        color: var(--muted);
        line-height: 1.6;
      }}

      .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 14px;
      }}

      .stat {{
        padding: 18px;
        border-radius: 18px;
        background: var(--accent-soft);
        border: 1px solid rgba(31, 122, 85, 0.12);
      }}

      .stat strong {{
        display: block;
        font-size: 1.4rem;
        margin-bottom: 6px;
      }}

      .actions {{
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        align-items: center;
      }}

      button {{
        border: none;
        border-radius: 999px;
        padding: 14px 22px;
        font: inherit;
        font-weight: 600;
        color: white;
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%);
        box-shadow: 0 10px 24px rgba(31, 122, 85, 0.24);
        cursor: pointer;
        transition: transform 140ms ease, box-shadow 140ms ease, opacity 140ms ease;
      }}

      button:hover:enabled {{
        transform: translateY(-1px);
        box-shadow: 0 14px 28px rgba(31, 122, 85, 0.28);
      }}

      button:disabled {{
        opacity: 0.58;
        cursor: not-allowed;
        box-shadow: none;
      }}

      .status {{
        min-height: 28px;
        font-weight: 600;
      }}

      .status.error {{
        color: var(--danger);
      }}

      .status.success {{
        color: var(--accent-strong);
      }}

      pre {{
        margin: 0;
        padding: 18px;
        border-radius: 18px;
        border: 1px solid rgba(20, 62, 43, 0.08);
        background: #fbfdfb;
        color: var(--text);
        overflow: auto;
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="hero">
        <div>
          <p>Beheeromgeving</p>
          <h1>{view_model.title}</h1>
        </div>

        <p>
          Start handmatig een historische import vanuit Home Assistant. De import gebruikt de
          bestaande configuratie en schrijft rechtstreeks naar de SQLite-database.
        </p>

        <div class="grid">
          <article class="stat">
            <strong>{view_model.sensor_count}</strong>
            <span>Geconfigureerde sensoren</span>
          </article>
          <article class="stat">
            <strong>{view_model.import_window_days}</strong>
            <span>Dagen terugkijken</span>
          </article>
          <article class="stat">
            <strong>{view_model.chunk_days}</strong>
            <span>Dagen per importblok</span>
          </article>
          <article class="stat">
            <strong>{view_model.api_port}</strong>
            <span>API-poort</span>
          </article>
        </div>

        <p><strong>Database:</strong> {view_model.database_path}</p>

        <div class="actions">
          <button id="import-button" type="button" {button_disabled}>{button_label}</button>
          <div id="status" class="status" aria-live="polite"></div>
        </div>

        <pre id="result">Nog geen import uitgevoerd.</pre>
      </section>
    </main>

    <script>
      const button = document.getElementById("import-button");
      const status = document.getElementById("status");
      const result = document.getElementById("result");

      async function runImport() {{
        button.disabled = true;
        status.className = "status";
        status.textContent = "Import wordt uitgevoerd...";

        try {{
          const response = await fetch("/api/history-import", {{
            method: "POST",
            headers: {{ "Content-Type": "application/json" }}
          }});

          const payload = await response.json();

          if (!response.ok) {{
            throw new Error(payload.detail || "Import mislukt.");
          }}

          status.className = "status success";
          status.textContent =
            `Import voltooid: ${{payload.total_rows}} rijen over ` +
            `${{payload.sensor_count}} sensoren.`;
          result.textContent = JSON.stringify(payload, null, 2);
        }} catch (error) {{
          status.className = "status error";
          status.textContent = error.message;
          result.textContent = "De import kon niet worden uitgevoerd.";
        }} finally {{
          button.disabled = {str(not view_model.import_enabled).lower()};
        }}
      }}

      button?.addEventListener("click", runImport);
    </script>
  </body>
</html>
"""
