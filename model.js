const config = window.MODEL_PAGE;

const taskLabels = {
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
  panoptic: "Panoptic Segmentation"
};

function metricLabel(metric) {
  const labels = {
    lpips_alex: "LPIPS", pck_bbox: "PCK", pck_bbox_reject: "PCK",
    nme_bbox: "NME", nme_bbox_matched: "NME", pq_novoid: "PQ", pq: "PQ",
    ois_f: "OIS F-score", max_fbeta: "Max Fβ", miou: "mIoU",
    delta1_aligned_ssc: "δ₁", mean_deg: "MAE (degrees)",
  };
  return labels[metric] || metric.replaceAll("_", " ");
}

function parseCsv(text) {
  const [headerLine, ...lines] = text.trim().split(/\r?\n/);
  const headers = headerLine.split(",");
  return lines.map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  });
}

function groupResults(rows) {
  const grouped = new Map();
  rows.forEach((row) => {
    if (!grouped.has(row.task)) {
      grouped.set(row.task, {
        task: row.task,
        metric: row.primary_metric,
        direction: row.direction,
        copy: Number(row.copy_input_mean),
        conditions: {}
      });
    }
    grouped.get(row.task).conditions[row.condition] = {
      mean: Number(row.mean),
      low: Number(row.ci95_low),
      high: Number(row.ci95_high),
      n: Number(row.n),
      nan: Number(row.n_nan)
    };
  });
  return [...grouped.values()].map((row) => {
    const gap = row.direction === "higher"
      ? row.conditions.real.mean - row.conditions.scrambled.mean
      : row.conditions.scrambled.mean - row.conditions.real.mean;
    return { ...row, gap };
  });
}

function formatScore(value) {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 10) return value.toFixed(2);
  return value.toFixed(4).replace(/^0(?=\.)/, "");
}

function scoreCell(condition, extraClass = "") {
  const count = condition.nan ? `n=${condition.n}, ${condition.nan} invalid` : `n=${condition.n}`;
  const title = `95% CI ${formatScore(condition.low)} to ${formatScore(condition.high)}; ${count}`;
  return `<span class="score ${extraClass}" tabindex="0" title="${title}"><strong>${formatScore(condition.mean)}</strong><small>${formatScore(condition.low)}–${formatScore(condition.high)}</small></span>`;
}

function resultRow(row) {
  const arrow = row.direction === "higher" ? "↑" : "↓";
  const gapClass = Math.abs(row.gap) < 0.0005 ? "neutral" : row.gap > 0 ? "positive" : "negative";
  const prefix = row.gap > 0 ? "+" : "";
  return `<tr>
    <td><span class="task-name">${taskLabels[row.task] || row.task}</span><span class="metric-name">${metricLabel(row.metric)} ${arrow}</span></td>
    <td>${scoreCell(row.conditions.real, "real")}</td>
    <td>${scoreCell(row.conditions.scrambled)}</td>
    <td>${scoreCell(row.conditions.grey)}</td>
    <td>${scoreCell(row.conditions.oracle)}</td>
    <td><span class="score"><strong>${formatScore(row.copy)}</strong></span></td>
    <td><span class="gap-badge ${gapClass}">${prefix}${formatScore(row.gap)}</span></td>
  </tr>`;
}

let testRows = [];

function renderTestRows() {
  const query = document.querySelector("#task-search").value.trim().toLowerCase();
  const sort = document.querySelector("#sort-results").value;
  const rows = testRows.filter((row) => `${taskLabels[row.task] || row.task} ${row.metric} ${metricLabel(row.metric)}`.toLowerCase().includes(query));
  rows.sort((a, b) => {
    if (sort === "gap-desc") return b.gap - a.gap;
    if (sort === "real-desc") return b.conditions.real.mean - a.conditions.real.mean;
    return (taskLabels[a.task] || a.task).localeCompare(taskLabels[b.task] || b.task);
  });
  document.querySelector("#results-body").innerHTML = rows.map(resultRow).join("");
  document.querySelector("#empty-state").hidden = rows.length > 0;
}

async function loadTable(url, bodySelector, interactive = false) {
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error("Results request failed");
    const rows = groupResults(parseCsv(await response.text()));
    if (interactive) {
      testRows = rows;
      renderTestRows();
    } else {
      document.querySelector(bodySelector).innerHTML = rows.map(resultRow).join("");
    }
  } catch (error) {
    document.querySelector(bodySelector).innerHTML = '<tr><td colspan="7" class="loading-cell">Results could not be loaded. The downloadable CSV is still available.</td></tr>';
  }
}

let sheetIndex = 0;
const dialog = document.querySelector("#image-dialog");
const dialogImage = document.querySelector("#dialog-image");

function renderSheet(index) {
  sheetIndex = (index + config.sheets.length) % config.sheets.length;
  const sheet = config.sheets[sheetIndex];
  const image = document.querySelector("#sample-image");
  image.src = sheet.image;
  image.alt = sheet.alt;
  document.querySelector("#sheet-title").textContent = sheet.title;
  document.querySelector("#sheet-kicker").textContent = `Sheet ${String(sheetIndex + 1).padStart(2, "0")} · ${sheet.count} tasks`;
  document.querySelector("#sheet-count").textContent = `${sheetIndex + 1} / ${config.sheets.length}`;
  document.querySelectorAll("#sheet-tabs button").forEach((button, buttonIndex) => {
    button.setAttribute("aria-selected", buttonIndex === sheetIndex ? "true" : "false");
    button.tabIndex = buttonIndex === sheetIndex ? 0 : -1;
  });
}

config.sheets.forEach((sheet, index) => {
  const button = document.createElement("button");
  button.type = "button";
  button.role = "tab";
  button.textContent = sheet.tasks;
  button.addEventListener("click", () => renderSheet(index));
  document.querySelector("#sheet-tabs").append(button);
});

document.querySelector("#previous-sheet").addEventListener("click", () => renderSheet(sheetIndex - 1));
document.querySelector("#next-sheet").addEventListener("click", () => renderSheet(sheetIndex + 1));
document.querySelector("#open-image").addEventListener("click", () => {
  const sheet = config.sheets[sheetIndex];
  dialogImage.src = sheet.image;
  dialogImage.alt = sheet.alt;
  document.querySelector("#dialog-title").textContent = sheet.tasks;
  dialog.showModal();
});
document.querySelector("#open-validation-image").addEventListener("click", () => {
  dialogImage.src = config.validationImage;
  dialogImage.alt = `Representative ${config.name} validation outputs`;
  document.querySelector("#dialog-title").textContent = `${config.name} validation outputs`;
  dialog.showModal();
});
document.querySelector("#close-dialog").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });

document.querySelector("#task-search").addEventListener("input", renderTestRows);
document.querySelector("#sort-results").addEventListener("change", renderTestRows);

renderSheet(0);
loadTable(config.testCsv, "#results-body", true);
loadTable(config.validationCsv, "#validation-results-body");
