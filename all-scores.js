const ALL_SCORE_TASKS = {
  relative_depth: "Relative Depth",
  surface_normals: "Surface Normals",
  semantic_segmentation: "Semantic Segmentation",
  instance_segmentation: "Instance Segmentation",
  panoptic_segmentation: "Panoptic Segmentation",
  edge_detection: "Edge Detection",
  saliency_detection: "Saliency Detection",
  human_pose_2d: "Human Pose 2D",
  face_landmarks: "Face Landmarks",
  object_keypoints: "Object Keypoints",
  colorization: "Colorization",
  denoising: "Denoising",
  deblurring: "Deblurring",
  inpainting: "Inpainting",
  super_resolution: "Super Resolution",
  depth: "Depth",
  normal: "Surface Normals",
  segmentation: "Binary Segmentation",
  panoptic: "Panoptic Segmentation",
};

const PROMPT_COUNT_TASK_ORDER = [
  "relative_depth", "surface_normals", "semantic_segmentation", "instance_segmentation", "panoptic_segmentation",
  "edge_detection", "saliency_detection", "human_pose_2d", "face_landmarks", "object_keypoints",
  "colorization", "denoising", "deblurring", "inpainting", "super_resolution",
];

const PROMPT_COUNT_COLUMNS = [
  { model: "VisualCloze", protocol: "Native P=2" },
  { model: "VisualCloze", protocol: "Native P=4" },
  { model: "LVM", protocol: "Native P=2" },
  { model: "LVM", protocol: "Native P=4" },
];

const RESTORATION_LPIPS_TASKS = [
  "colorization", "denoising", "deblurring", "inpainting", "super_resolution",
];

const RESTORATION_LPIPS_COLUMNS = [
  { model: "VisualCloze", protocol: "Native P=2" },
  { model: "LVM", protocol: "Native P=1" },
];

function parseScoreCsv(text) {
  const table = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        field += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      row.push(field.replace(/\r$/, ""));
      table.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }
  if (field || row.length) {
    row.push(field.replace(/\r$/, ""));
    table.push(row);
  }
  const headers = table.shift();
  return table.filter((cells) => cells.length === headers.length).map((cells) =>
    Object.fromEntries(headers.map((header, index) => [header, cells[index]]))
  );
}

function safeText(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function prettyMetric(metric) {
  const labels = {
    lpips_alex: "LPIPS",
    pck_bbox: "PCK",
    pck_bbox_reject: "PCK",
    nme_bbox: "NME",
    nme_bbox_matched: "NME",
    pq_novoid: "PQ",
    pq: "PQ",
    ois_f: "OIS F-score",
    max_fbeta: "Max Fβ",
    miou: "mIoU",
    delta1_aligned_ssc: "δ₁",
    mean_deg: "MAE (degrees)",
  };
  return labels[metric] || metric.replaceAll("_", " ");
}

function formatScore(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  if (Math.abs(number) >= 100) return number.toFixed(1);
  if (Math.abs(number) >= 10) return number.toFixed(2);
  return number.toFixed(4);
}

function formatExactScore(value) {
  const storedValue = String(value ?? "").trim();
  return storedValue || "—";
}

function renderPromptCountTable(rows) {
  const body = document.querySelector("#prompt-count-body");
  if (!body) return;
  const selected = rows.filter((row) =>
    row.split === "Test" && row.condition === "real" &&
    PROMPT_COUNT_COLUMNS.some((column) => row.model === column.model && row.protocol === column.protocol)
  );
  const lookup = new Map(selected.map((row) => [`${row.model}|${row.protocol}|${row.task}`, row]));

  body.innerHTML = PROMPT_COUNT_TASK_ORDER.map((task) => {
    const taskRows = PROMPT_COUNT_COLUMNS.map((column) => lookup.get(`${column.model}|${column.protocol}|${task}`));
    if (!taskRows.find(Boolean)) return "";
    const cells = taskRows.map((row) => {
      if (!row) return "<td>—</td>";
      const ci = `${formatExactScore(row.ci95_low)} to ${formatExactScore(row.ci95_high)}`;
      const direction = row.direction === "lower" ? "↓" : "↑";
      return `<td><span class="score" title="95% CI ${ci}; n=${safeText(row.n)}"><strong>${formatExactScore(row.mean)}</strong></span><span class="cell-note">${safeText(prettyMetric(row.primary_metric))} ${direction}</span></td>`;
    }).join("");
    return `<tr><td><span class="task-name">${safeText(ALL_SCORE_TASKS[task])}</span></td>${cells}</tr>`;
  }).join("");

  if (!body.innerHTML.trim()) {
    body.innerHTML = '<tr><td colspan="5" class="loading-cell">P=2 and P=4 scores are unavailable.</td></tr>';
  }
}

function renderRestorationLpipsTable(rows) {
  const body = document.querySelector("#restoration-lpips-body");
  if (!body) return;
  const selected = rows.filter((row) =>
    row.split === "Test" && row.condition === "real" && row.primary_metric === "lpips_alex" &&
    RESTORATION_LPIPS_COLUMNS.some((column) => row.model === column.model && row.protocol === column.protocol)
  );
  const lookup = new Map(selected.map((row) => [`${row.model}|${row.protocol}|${row.task}`, row]));

  body.innerHTML = RESTORATION_LPIPS_TASKS.map((task) => {
    const taskRows = RESTORATION_LPIPS_COLUMNS.map((column) => lookup.get(`${column.model}|${column.protocol}|${task}`));
    if (taskRows.some((row) => !row)) return "";
    const cells = taskRows.map((row) => {
      const ci = `${formatExactScore(row.ci95_low)} to ${formatExactScore(row.ci95_high)}`;
      return `<td><span class="score" title="95% CI ${ci}; n=${safeText(row.n)}"><strong>${formatExactScore(row.mean)}</strong></span></td>`;
    }).join("");
    return `<tr><td><span class="task-name">${safeText(ALL_SCORE_TASKS[task])}</span><span class="metric-name">LPIPS ↓</span></td>${cells}</tr>`;
  }).join("");

  if (!body.innerHTML.trim()) {
    body.innerHTML = '<tr><td colspan="3" class="loading-cell">The corrected LPIPS comparison is unavailable.</td></tr>';
  }
}

function populateScoreFilter(select, values) {
  for (const value of [...new Set(values)].sort()) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.append(option);
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  const body = document.querySelector("#all-scores-body");
  if (!body) return;

  const controls = {
    search: document.querySelector("#all-score-search"),
    model: document.querySelector("#all-score-model"),
    protocol: document.querySelector("#all-score-protocol"),
    split: document.querySelector("#all-score-split"),
    condition: document.querySelector("#all-score-condition"),
  };
  const summary = document.querySelector("#all-score-count");

  try {
    const response = await fetch("data/all-eval-scores.csv");
    if (!response.ok) throw new Error(`score table request failed: ${response.status}`);
    const rows = parseScoreCsv(await response.text());
    renderRestorationLpipsTable(rows);
    renderPromptCountTable(rows);

    populateScoreFilter(controls.model, rows.map((row) => row.model));
    populateScoreFilter(controls.protocol, rows.map((row) => row.protocol));
    populateScoreFilter(controls.split, rows.map((row) => row.split));
    populateScoreFilter(controls.condition, rows.map((row) => row.condition));

    const render = () => {
      const query = controls.search.value.trim().toLowerCase();
      const filtered = rows.filter((row) => {
        if (controls.model.value && row.model !== controls.model.value) return false;
        if (controls.protocol.value && row.protocol !== controls.protocol.value) return false;
        if (controls.split.value && row.split !== controls.split.value) return false;
        if (controls.condition.value && row.condition !== controls.condition.value) return false;
        if (!query) return true;
        return [row.model, row.protocol, row.split, row.task, row.condition, row.primary_metric]
          .join(" ").toLowerCase().includes(query);
      });

      body.innerHTML = filtered.map((row) => {
        const task = ALL_SCORE_TASKS[row.task] || row.task.replaceAll("_", " ");
        const direction = row.direction === "lower" ? "↓" : "↑";
        const ci = row.ci95_low && row.ci95_high
          ? `${formatScore(row.ci95_low)}–${formatScore(row.ci95_high)}`
          : "—";
        const subset = row.kind === "q0-ttt-subset" ? '<span class="subset-badge">q0 only</span>' : "";
        return `<tr>
          <td><span class="task-name">${safeText(row.model)}</span><span class="metric-name">${safeText(row.protocol)}</span></td>
          <td><strong>P=${safeText(row.prompt_count)}</strong><span class="cell-note">${safeText(row.prompt_mode)}</span></td>
          <td>${safeText(row.split)} ${subset}<span class="cell-note">${safeText(row.scope)}</span></td>
          <td><span class="task-name">${safeText(task)}</span><span class="metric-name">${safeText(prettyMetric(row.primary_metric))} ${direction}</span></td>
          <td><span class="condition-pill ${safeText(row.condition.toLowerCase().replaceAll("-", "_"))}">${safeText(row.condition)}</span></td>
          <td class="all-score-value">${formatScore(row.mean)}</td>
          <td>${ci}</td>
          <td>${safeText(row.n)}</td>
          <td><a href="${safeText(row.script_href)}" download>Scripts ↓</a></td>
        </tr>`;
      }).join("");

      if (!filtered.length) {
        body.innerHTML = '<tr><td colspan="9" class="loading-cell">No scores match these filters.</td></tr>';
      }
      const protocols = new Set(filtered.map((row) => row.score_source)).size;
      const models = new Set(filtered.map((row) => row.model)).size;
      summary.textContent = `${filtered.length.toLocaleString()} score cells · ${protocols} protocols · ${models} models`;
    };

    Object.values(controls).forEach((control) => control.addEventListener("input", render));
    render();
  } catch (error) {
    body.innerHTML = '<tr><td colspan="9" class="loading-cell">Unable to load the consolidated scores.</td></tr>';
    summary.textContent = "Score table unavailable";
    console.error(error);
  }
});
