const benchV2TaskLabels = {
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

const benchV2MetricLabels = {
  delta1: "δ1",
  lpips: "LPIPS",
  max_fbeta: "max Fβ",
  mean_deg: "Mean degrees",
  miou: "mIoU",
  nme: "NME",
  "pck_torso_0.2": "PCK@0.2",
  pooled_ois: "Pooled OIS",
  pq: "PQ",
};

function benchV2Score(value, metric) {
  if (metric === "mean_deg") return value.toFixed(3);
  return value.toFixed(6);
}

function renderBenchV2(data) {
  const body = document.querySelector("#genlv-bench-v2-body");
  const edge = document.querySelector("#genlv-edge-readout");
  if (!body || !edge) return;
  body.innerHTML = "";

  data.rows.forEach((row) => {
    const tr = document.createElement("tr");
    const direction = row.lower_is_better ? "↓" : "↑";
    const won = row.improvement > 0;
    const tied = Math.abs(row.improvement) < 1e-12;
    const result = tied ? "Tie" : won ? "Better" : "Worse";
    tr.innerHTML = `
      <th scope="row">
        <span class="bench-v2-task-name">${benchV2TaskLabels[row.task] || row.task}</span>
        <span class="bench-v2-metric-name">${benchV2MetricLabels[row.metric] || row.metric} ${direction}</span>
      </th>
      <td class="bench-v2-model-score">${benchV2Score(row.genlv, row.metric)}</td>
      <td>${benchV2Score(row.copy, row.metric)}</td>
      <td class="${won ? "bench-v2-positive" : tied ? "" : "bench-v2-negative"}">${row.improvement > 0 ? "+" : ""}${benchV2Score(row.improvement, row.metric)}</td>
      <td><span class="bench-v2-result ${won ? "is-better" : tied ? "is-tie" : "is-worse"}">${result}</span></td>
    `;
    body.appendChild(tr);
  });

  const e = data.edge_pooled;
  edge.innerHTML = `
    <article><span>ODS</span><strong>${e.genlv.ods.toFixed(6)}</strong><small>copy ${e.copy.ods.toFixed(6)}</small></article>
    <article><span>OIS</span><strong>${e.genlv.ois.toFixed(6)}</strong><small>copy ${e.copy.ois.toFixed(6)}</small></article>
    <article><span>AP</span><strong>${e.genlv.ap.toFixed(6)}</strong><small>copy ${e.copy.ap.toFixed(6)}</small></article>
  `;
}

fetch("../data/genlv-bench-v2-validation.json")
  .then((response) => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  })
  .then(renderBenchV2)
  .catch((error) => {
    const body = document.querySelector("#genlv-bench-v2-body");
    if (body) body.innerHTML = `<tr><td colspan="5" class="loading-cell">Bench-v2 scores could not be loaded: ${error.message}</td></tr>`;
  });
