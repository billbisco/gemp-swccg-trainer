async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { Accept: "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${text}`);
  }
  return res.json();
}

function fmtRate(v) {
  if (v === null || v === undefined) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function renderStatus(s) {
  document.getElementById("m-status").textContent = s.status || "—";
  document.getElementById("m-mode").textContent = `${s.mode || "—"} → ${s.actual_mode || "?"}`;
  document.getElementById("m-run").textContent = s.run_id || "—";
  document.getElementById("m-games").textContent = String(s.games_completed ?? 0);
  document.getElementById("m-dark").textContent = fmtRate(s.dark_win_rate);
  document.getElementById("m-light").textContent = fmtRate(s.light_win_rate);
  document.getElementById("m-gph").textContent = s.games_per_hour ?? "—";
  document.getElementById("m-runs").textContent = s.run_dir || "(no active run)";
  const notes = document.getElementById("notes");
  notes.innerHTML = "";
  (s.notes || []).slice(-8).forEach((n) => {
    const li = document.createElement("li");
    li.textContent = n;
    notes.appendChild(li);
  });
  if (s.last_error) {
    const li = document.createElement("li");
    li.style.color = "#ff8a80";
    li.textContent = "Error: " + s.last_error;
    notes.appendChild(li);
  }
}

async function refresh() {
  try {
    const s = await api("/api/status");
    renderStatus(s);
  } catch (e) {
    console.error(e);
  }
}

function renderWatch(payload) {
  const panel = document.getElementById("watch-panel");
  panel.hidden = false;
  const summary = payload.summary || { message: payload.message, available: payload.available };
  document.getElementById("watch-summary").textContent = JSON.stringify(summary, null, 2);
  document.getElementById("watch-traces").textContent = payload.traces_path || payload.sample_traces_path || "—";
  document.getElementById("watch-replay").textContent = payload.replay_dir || "(none yet — HeadlessReplayWriter pending)";
  const tbody = document.querySelector("#timeline tbody");
  tbody.innerHTML = "";
  (payload.timeline || []).forEach((row) => {
    if (row.type === "header" || row.type === "outcome") return;
    const tr = document.createElement("tr");
    const side = row.side || "";
    tr.innerHTML = `
      <td>${row.decisionIndex ?? ""}</td>
      <td class="side-${side}">${side}</td>
      <td>${row.phase || ""}</td>
      <td>${row.decisionType || row.type || ""}</td>
      <td>${row.chosen ?? ""}</td>
      <td>${row.darkLF ?? ""}</td>
      <td>${row.lightLF ?? ""}</td>
      <td>${(row.decisionText || "").slice(0, 80)}</td>`;
    tbody.appendChild(tr);
  });
}

document.getElementById("btn-start").addEventListener("click", async () => {
  const mode = document.getElementById("mode").value;
  const s = await api(`/api/start?mode=${encodeURIComponent(mode)}`, { method: "POST" });
  renderStatus(s);
});

document.getElementById("btn-pause").addEventListener("click", async () => {
  const s = await api("/api/pause", { method: "POST" });
  renderStatus(s);
});

document.getElementById("btn-watch").addEventListener("click", async () => {
  const payload = await api("/api/watch");
  renderWatch(payload);
});

document.getElementById("btn-export").addEventListener("click", async () => {
  const result = await api("/api/export", { method: "POST" });
  const panel = document.getElementById("export-panel");
  panel.hidden = false;
  document.getElementById("export-out").textContent = JSON.stringify(result, null, 2);
});

refresh();
setInterval(refresh, 1500);
