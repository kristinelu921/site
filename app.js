const sheets = [
  {
    image: "assets/samples-01.png",
    title: "Geometry & parsing",
    tasks: "Relative Depth · Surface Normals · Semantic Segmentation",
    alt: "Sample outputs for Relative Depth, Surface Normals, and Semantic Segmentation"
  },
  {
    image: "assets/samples-02.png",
    title: "Masks & boundaries",
    tasks: "Instance Segmentation · Panoptic Segmentation · Edge Detection",
    alt: "Sample outputs for Instance Segmentation, Panoptic Segmentation, and Edge Detection"
  },
  {
    image: "assets/samples-03.png",
    title: "People & attention",
    tasks: "Saliency · Human Pose · Face Landmarks",
    alt: "Sample outputs for Saliency Detection, Human Pose 2D, and Face Landmarks"
  },
  {
    image: "assets/samples-04.png",
    title: "Keypoints & restoration",
    tasks: "Object Keypoints · Colorization · Denoising",
    alt: "Sample outputs for Object Keypoints, Colorization, and Denoising"
  },
  {
    image: "assets/samples-05.png",
    title: "Image restoration",
    tasks: "Deblurring · Inpainting · Super Resolution",
    alt: "Sample outputs for Deblurring, Inpainting, and Super Resolution"
  }
];

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

let currentSheet = 0;
let taskRows = [];

const sampleImage = document.querySelector("#sample-image");
const dialog = document.querySelector("#image-dialog");
const dialogImage = document.querySelector("#dialog-image");

function renderSheet(index) {
  currentSheet = (index + sheets.length) % sheets.length;
  const sheet = sheets[currentSheet];
  sampleImage.src = sheet.image;
  sampleImage.alt = sheet.alt;
  dialogImage.src = sheet.image;
  dialogImage.alt = sheet.alt;
  document.querySelector("#sheet-title").textContent = sheet.title;
  document.querySelector("#sheet-kicker").textContent = `Sheet ${String(currentSheet + 1).padStart(2, "0")} · 3 tasks`;
  document.querySelector("#sheet-count").textContent = `${currentSheet + 1} / ${sheets.length}`;
  document.querySelector("#dialog-title").textContent = sheet.tasks;

  document.querySelectorAll("#sheet-tabs button").forEach((button, buttonIndex) => {
    button.setAttribute("aria-selected", buttonIndex === currentSheet ? "true" : "false");
    button.tabIndex = buttonIndex === currentSheet ? 0 : -1;
  });
}

const tabs = document.querySelector("#sheet-tabs");
sheets.forEach((sheet, index) => {
  const button = document.createElement("button");
  button.type = "button";
  button.role = "tab";
  button.textContent = sheet.tasks;
  button.addEventListener("click", () => renderSheet(index));
  tabs.append(button);
});

document.querySelector("#previous-sheet").addEventListener("click", () => renderSheet(currentSheet - 1));
document.querySelector("#next-sheet").addEventListener("click", () => renderSheet(currentSheet + 1));
document.querySelector("#open-image").addEventListener("click", () => {
  const sheet = sheets[currentSheet];
  dialogImage.src = sheet.image;
  dialogImage.alt = sheet.alt;
  document.querySelector("#dialog-title").textContent = sheet.tasks;
  dialog.showModal();
});
document.querySelector("#open-validation-image").addEventListener("click", () => {
  dialogImage.src = "assets/lineage-a-samples.png";
  dialogImage.alt = "Representative VisualCloze outputs for the four validation tasks";
  document.querySelector("#dialog-title").textContent = "Validation outputs";
  dialog.showModal();
});
document.querySelector("#close-dialog").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", (event) => {
  if (event.target === dialog) dialog.close();
});

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
    const signedGap = row.direction === "higher"
      ? row.conditions.real.mean - row.conditions.scrambled.mean
      : row.conditions.scrambled.mean - row.conditions.real.mean;
    return { ...row, gap: signedGap };
  });
}

function formatScore(value) {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 10) return value.toFixed(2);
  return value.toFixed(4).replace(/^0(?=\.)/, "");
}

function formatGap(value) {
  const prefix = value > 0 ? "+" : "";
  return `${prefix}${formatScore(value)}`;
}

function scoreCell(condition, extraClass = "") {
  const count = condition.nan ? `n=${condition.n}, ${condition.nan} invalid` : `n=${condition.n}`;
  const title = `95% CI ${formatScore(condition.low)} to ${formatScore(condition.high)}; ${count}`;
  return `<span class="score ${extraClass}" tabindex="0" title="${title}"><strong>${formatScore(condition.mean)}</strong><small>${formatScore(condition.low)}–${formatScore(condition.high)}</small></span>`;
}

function resultRow(row) {
  const directionArrow = row.direction === "higher" ? "↑" : "↓";
  const gapClass = Math.abs(row.gap) < 0.0005 ? "neutral" : row.gap > 0 ? "positive" : "negative";
  return `<tr>
    <td><span class="task-name">${taskLabels[row.task]}</span><span class="metric-name">${metricLabel(row.metric)} ${directionArrow}</span></td>
    <td>${scoreCell(row.conditions.real, "real")}</td>
    <td>${scoreCell(row.conditions.scrambled)}</td>
    <td>${scoreCell(row.conditions.grey)}</td>
    <td>${scoreCell(row.conditions.oracle)}</td>
    <td><span class="score"><strong>${formatScore(row.copy)}</strong></span></td>
    <td><span class="gap-badge ${gapClass}">${formatGap(row.gap)}</span></td>
  </tr>`;
}

function renderResults() {
  const query = document.querySelector("#task-search").value.trim().toLowerCase();
  const sort = document.querySelector("#sort-results").value;
  const filtered = taskRows.filter((row) => {
    const haystack = `${taskLabels[row.task]} ${row.metric} ${metricLabel(row.metric)}`.toLowerCase();
    return haystack.includes(query);
  });

  filtered.sort((a, b) => {
    if (sort === "gap-desc") return b.gap - a.gap;
    if (sort === "real-desc") return b.conditions.real.mean - a.conditions.real.mean;
    return taskLabels[a.task].localeCompare(taskLabels[b.task]);
  });

  const body = document.querySelector("#results-body");
  body.innerHTML = filtered.map(resultRow).join("");

  document.querySelector("#empty-state").hidden = filtered.length > 0;
}

async function loadValidationResults() {
  try {
    const response = await fetch("data/lineage-a-results.csv");
    if (!response.ok) throw new Error("Validation results request failed");
    const rows = groupResults(parseCsv(await response.text()));
    document.querySelector("#validation-results-body").innerHTML = rows.map(resultRow).join("");
  } catch (error) {
    document.querySelector("#validation-results-body").innerHTML = '<tr><td colspan="7" class="loading-cell">Validation results could not be loaded. The downloadable CSV is still available.</td></tr>';
  }
}

async function loadResults() {
  try {
    const response = await fetch("data/results.csv");
    if (!response.ok) throw new Error("Results request failed");
    taskRows = groupResults(parseCsv(await response.text()));
    renderResults();
  } catch (error) {
    document.querySelector("#results-body").innerHTML = '<tr><td colspan="7" class="loading-cell">Results could not be loaded. The downloadable CSV is still available.</td></tr>';
  }
}

document.querySelector("#task-search").addEventListener("input", renderResults);
document.querySelector("#sort-results").addEventListener("change", renderResults);

renderSheet(0);
loadResults();
loadValidationResults();
