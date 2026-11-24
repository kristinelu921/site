const comparisonModels = [
  { slug: "visualcloze", name: "VisualCloze", page: "#test", test: "data/results.csv", validation: "data/lineage-a-results.csv" },
  { slug: "lvm", name: "LVM", page: "lvm/", test: "data/lvm-test.csv", validation: "data/lvm-validation.csv" },
  { slug: "painter", name: "Painter", page: "painter/", test: "data/painter-test.csv", validation: "data/painter-validation.csv" },
  { slug: "genlv", name: "GenLV", page: "genlv/", test: "data/genlv-test.csv", validation: "data/genlv-validation.csv" },
  { slug: "prompt-diffusion", name: "Prompt Diffusion", page: "prompt-diffusion/", test: "data/prompt-diffusion-test.csv", validation: "data/prompt-diffusion-validation.csv" },
  { slug: "improv", name: "IMProv", page: "improv/", test: "data/improv-test.csv", validation: "data/improv-validation.csv" },
  { slug: "relation-adapter", name: "RelationAdapter", page: "relation-adapter/", test: "data/relation-adapter-test.csv", validation: "data/relation-adapter-validation.csv" }
];

const comparisonTaskOrder = {
  test: ["relative_depth", "surface_normals", "semantic_segmentation", "instance_segmentation", "panoptic_segmentation", "edge_detection", "saliency_detection", "human_pose_2d", "face_landmarks", "object_keypoints", "colorization", "denoising", "deblurring", "inpainting", "super_resolution"],
  validation: ["depth", "normal", "segmentation", "panoptic"]
};

const comparisonLabels = {
  relative_depth: "Relative Depth", surface_normals: "Surface Normals", semantic_segmentation: "Semantic Segmentation",
  instance_segmentation: "Instance Segmentation", panoptic_segmentation: "Panoptic Segmentation", edge_detection: "Edge Detection",
  saliency_detection: "Saliency Detection", human_pose_2d: "Human Pose 2D", face_landmarks: "Face Landmarks",
  object_keypoints: "Object Keypoints", colorization: "Colorization", denoising: "Denoising", deblurring: "Deblurring",
  inpainting: "Inpainting", super_resolution: "Super Resolution", depth: "Depth", normal: "Surface Normals",
  segmentation: "Binary Segmentation", panoptic: "Panoptic Segmentation"
};

let comparisonSplit = "test";
let comparisonCondition = "real";
const comparisonData = { test: new Map(), validation: new Map() };

function parseComparisonCsv(text) {
  const [headerLine, ...lines] = text.trim().split(/\r?\n/);
  const headers = headerLine.split(",");
  return lines.map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  });
}

function formatComparisonScore(value) {
  if (!Number.isFinite(value)) return "—";
  return value.toFixed(6).replace(/(\.\d*?[1-9])0+$|\.0+$/, "$1");
}

function comparisonMetricLabel(metric) {
  const labels = {
    lpips_alex: "LPIPS", pck_bbox: "PCK", pck_bbox_reject: "PCK",
    nme_bbox: "NME", nme_bbox_matched: "NME", pq_novoid: "PQ", pq: "PQ",
    ois_f: "OIS F-score", max_fbeta: "Max Fβ", miou: "mIoU",
    delta1_aligned_ssc: "δ₁", mean_deg: "MAE (degrees)",
  };
  return labels[metric] || metric.replaceAll("_", " ");
}

function comparisonHeader() {
  return `<tr><th scope="col">Task / primary metric</th>${comparisonModels.map((model) => `
    <th scope="col"><a href="${model.page}">${model.name}</a><a class="header-script-link" href="scripts/${model.slug}/evaluation-scripts.zip" download>Scripts</a></th>`).join("")}</tr>`;
}

function renderComparison() {
  const head = document.querySelector("#comparison-head");
  const body = document.querySelector("#comparison-body");
  head.innerHTML = comparisonHeader();

  const tasks = comparisonTaskOrder[comparisonSplit];
  body.innerHTML = tasks.map((task) => {
    const rows = comparisonModels.map((model) => comparisonData[comparisonSplit].get(model.slug)?.get(`${task}@${comparisonCondition}`));
    const first = rows.find(Boolean);
    if (!first) return "";
    const metricSignatures = new Set(rows.filter(Boolean).map((row) => `${comparisonMetricLabel(row.primary_metric)}|${row.direction}`));
    const comparable = metricSignatures.size === 1;
    const values = rows.map((row) => Number(row?.mean));
    const finite = values.filter(Number.isFinite);
    const best = first.direction === "higher" ? Math.max(...finite) : Math.min(...finite);
    const arrow = first.direction === "higher" ? "↑" : "↓";
    const cells = rows.map((row, index) => {
      if (!row) return "<td>—</td>";
      const value = Number(row.mean);
      const isBest = comparable && Number.isFinite(value) && Math.abs(value - best) <= Math.max(1e-12, Math.abs(best) * 1e-12);
      const title = `95% CI ${formatComparisonScore(Number(row.ci95_low))} to ${formatComparisonScore(Number(row.ci95_high))}; n=${row.n}`;
      const rowArrow = row.direction === "higher" ? "↑" : "↓";
      const metric = comparable ? "" : `<span class="cell-note">${comparisonMetricLabel(row.primary_metric)} ${rowArrow}</span>`;
      return `<td class="${isBest ? "best-score" : ""}"><a href="${comparisonModels[index].page}#${comparisonSplit === "test" ? "results" : "validation"}" title="${title}">${formatComparisonScore(value)}</a>${metric}${isBest ? "<span>Best</span>" : ""}</td>`;
    }).join("");
    const metricLabel = comparable ? `${comparisonMetricLabel(first.primary_metric)} ${arrow}` : "model-specific metrics";
    return `<tr><td><span class="task-name">${comparisonLabels[task]}</span><span class="metric-name">${metricLabel}</span></td>${cells}</tr>`;
  }).join("");

  const splitLabel = comparisonSplit === "test" ? "15-task test" : "four-task validation";
  document.querySelector("#comparison-caption").textContent = `${splitLabel} · ${comparisonCondition} condition`;
}

async function loadComparison() {
  try {
    const requests = comparisonModels.flatMap((model) => [
      fetch(model.test).then((response) => response.ok ? response.text() : Promise.reject(new Error(model.test))).then((text) => ["test", model.slug, text]),
      fetch(model.validation).then((response) => response.ok ? response.text() : Promise.reject(new Error(model.validation))).then((text) => ["validation", model.slug, text])
    ]);
    const responses = await Promise.all(requests);
    responses.forEach(([split, slug, text]) => {
      const rows = parseComparisonCsv(text);
      comparisonData[split].set(slug, new Map(rows.map((row) => [`${row.task}@${row.condition}`, row])));
    });
    renderComparison();
  } catch (error) {
    document.querySelector("#comparison-body").innerHTML = '<tr><td colspan="8" class="loading-cell">The combined results could not be loaded. Individual model tables remain available above.</td></tr>';
  }
}

document.querySelector("#comparison-split").addEventListener("change", (event) => {
  comparisonSplit = event.target.value;
  renderComparison();
});

document.querySelectorAll("[data-comparison-condition]").forEach((button) => {
  button.addEventListener("click", () => {
    comparisonCondition = button.dataset.comparisonCondition;
    document.querySelectorAll("[data-comparison-condition]").forEach((item) => item.setAttribute("aria-pressed", item === button ? "true" : "false"));
    renderComparison();
  });
});

loadComparison();
