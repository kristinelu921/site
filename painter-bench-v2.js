const painterBenchTaskLabels = {
  colorization: "Colorization",
  deblurring: "Deblurring",
  denoising: "Denoising",
  edge_detection: "Edge Detection",
  face_landmarks: "Face Landmarks",
  human_pose_2d: "Human Pose 2D",
  inpainting: "Inpainting",
  instance_segmentation: "Instance Segmentation",
  relative_depth: "Relative Depth",
  saliency_detection: "Saliency Detection",
  semantic_segmentation: "Semantic Segmentation",
  super_resolution: "Super Resolution",
  surface_normals: "Surface Normals",
};

const painterBenchMetricLabels = {
  delta1: "δ1",
  lpips: "LPIPS",
  max_fbeta: "max Fβ",
  mean_deg: "Mean degrees",
  miou: "mIoU",
  nme: "NME",
  ois_f: "Mean per-query OIS F",
  "pck_torso_0.2": "PCK@0.2",
  pq: "PQ",
};

function painterBenchScore(value, metric) {
  if (metric === "mean_deg") return value.toFixed(3);
  return value.toFixed(6);
}

function renderPainterBench(data) {
  const body = document.querySelector("#painter-bench-v2-body");
  const edge = document.querySelector("#painter-edge-readout");
  if (!body || !edge) return;
  body.innerHTML = data.rows.map((row) => {
    const direction = row.lower_is_better ? "↓" : "↑";
    const won = row.improvement > 0;
    const tied = Math.abs(row.improvement) < 1e-12;
    const result = tied ? "Tie" : won ? "Better" : "Worse";
    return `<tr>
      <th scope="row">
        <span class="bench-v2-task-name">${painterBenchTaskLabels[row.task] || row.task}</span>
        <span class="bench-v2-metric-name">${painterBenchMetricLabels[row.metric] || row.metric} ${direction}</span>
      </th>
      <td class="bench-v2-model-score">${painterBenchScore(row.painter, row.metric)}</td>
      <td>${painterBenchScore(row.copy, row.metric)}</td>
      <td class="${won ? "bench-v2-positive" : tied ? "" : "bench-v2-negative"}">${row.improvement > 0 ? "+" : ""}${painterBenchScore(row.improvement, row.metric)}</td>
      <td><span class="bench-v2-result ${won ? "is-better" : tied ? "is-tie" : "is-worse"}">${result}</span></td>
    </tr>`;
  }).join("");

  const pooled = data.edge_pooled;
  edge.innerHTML = ["ods", "ois", "ap"].map((metric) => `<article>
    <span>${metric.toUpperCase()}</span>
    <strong>${pooled.painter[metric].toFixed(6)}</strong>
    <small>copy ${pooled.copy[metric].toFixed(6)}</small>
  </article>`).join("");
}

fetch("../data/painter-bench-v2-validation.json")
  .then((response) => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  })
  .then(renderPainterBench)
  .catch((error) => {
    const body = document.querySelector("#painter-bench-v2-body");
    if (body) body.innerHTML = `<tr><td colspan="5" class="loading-cell">Bench-v2 scores could not be loaded: ${error.message}</td></tr>`;
  });
